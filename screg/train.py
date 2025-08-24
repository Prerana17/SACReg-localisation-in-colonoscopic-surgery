from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import argparse
import os
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .datasets import SimCol3DDataset, simcol3d_collate_fn
from .model import SCRegNet, PointEmbed

# -----------------------------------------------------------------------------
# φ-encoding helper (vectorised torch) – matches PointEmbed 38-D scheme
# -----------------------------------------------------------------------------
F1_DEFAULT = 0.017903170262351338
GAMMA_DEFAULT = 2.884031503126606
F_HARMONICS = 6  # ⇒ 38-D (2 + 3*F*2)


def encode_phi38(xyz: torch.Tensor, f1: float = F1_DEFAULT, gamma: float = GAMMA_DEFAULT) -> torch.Tensor:
    """Compute 38-D φ-encoding for xyz map.

    Parameters
    ----------
    xyz : (B,3,H,W) tensor in metres
    returns : (B,38,H,W) tensor
    """
    B, _, H, W = xyz.shape
    device = xyz.device
    dtype = xyz.dtype

    # (1, F)
    freqs = torch.tensor([f1 * (gamma ** i) for i in range(F_HARMONICS)], dtype=dtype, device=device)[None]

    # Harmonic encoding xyz → (B, 3*2F, H, W)
    xyz_exp = xyz.unsqueeze(2)  # (B,3,1,H,W)
    ang = xyz_exp * freqs.view(1, 1, F_HARMONICS, 1, 1)
    cos_part = torch.cos(ang)
    sin_part = torch.sin(ang)
    enc = torch.cat([cos_part, sin_part], dim=2)  # (B,3,2F,H,W)
    enc = enc.flatten(1, 2)  # (B, 3*2F, H, W)

    # Normalised uv grid (B,2,H,W)
    u_lin = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    v_lin = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v_lin, u_lin, indexing="ij")
    uv = torch.stack([u_grid, v_grid], dim=0).expand(B, -1, -1, -1)  # (B,2,H,W)

    return torch.cat([uv, enc], dim=1)  # (B,38,H,W)


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

def pixelwise_confidence_loss(pred: torch.Tensor, target: torch.Tensor, tau: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """τ-weighted L1 loss with −log τ term.

    pred  : (B,38,H,W)
    target: (B,38,H,W)
    tau   : (B,H,W) or (B,1,H,W)
    mask  : (B,1,H,W) or None
    """
    if tau.dim() == 3:
        tau = tau.unsqueeze(1)
    tau = tau.clamp_min(1e-6)  # ensure positive

    # Pairwise ℓ2 normalisation on harmonic components of prediction (cos,sin)
    # Skip first 2 channels (u,v), normalise 18 (cos,sin) pairs per xyz axis.
    if pred.shape[1] != 38:
        raise ValueError("Expected 38-D φ vector")
    uv_pred = pred[:, :2]
    enc_pred = pred[:, 2:].reshape(pred.shape[0], 18, 2, *pred.shape[2:])  # (B,18,2,H,W)
    norm = enc_pred.norm(dim=2, keepdim=True).clamp_min(1e-6)
    enc_pred = enc_pred / norm
    pred_norm = torch.cat([uv_pred, enc_pred.reshape(pred.shape[0], 36, *pred.shape[2:])], dim=1)

    diff = (pred_norm - target).abs()  # (B,38,H,W)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
        diff = diff.sum() / denom
    else:
        diff = diff.mean()

    return (tau * diff - torch.log(tau)).mean()


# -----------------------------------------------------------------------------
# Optimiser & scheduler
# -----------------------------------------------------------------------------

def build_optimizer(params, lr: float = 1e-4):
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.05)


def build_scheduler(optimizer: torch.optim.Optimizer):
    def lr_lambda(epoch):
        return (epoch + 1) / 10.0 if epoch < 10 else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_one_epoch(model: SCRegNet, loader: DataLoader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        # Move tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        query_img = batch["query_img"]  # (B,3,H,W)
        db_img = batch["db_img"]
        uvxyz = batch["uvxyz"]  # (B,K,5)
        xyz_gt = batch["query_xyz"]  # (B,3,h0,w0)

        B, _, H_tgt, W_tgt = query_img.shape

        # Resize xyz to image resolution
        xyz_gt_rs = F.interpolate(xyz_gt, size=(H_tgt, W_tgt), mode="bilinear", align_corners=False)

        # Compute φ38 target on-the-fly
        phi_gt = encode_phi38(xyz_gt_rs)

        with autocast():
            outputs = model(query_img, db_img, uvxyz)
            phi_pred = outputs["phi38"]  # (B,38,H,W)
            tau = outputs["conf"]  # (B,H,W)

            loss = pixelwise_confidence_loss(phi_pred, phi_gt, tau, mask=None)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item() * B
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_train(data_root: str | Path = "data/processed/SimCol3D", epochs: int = 100, batch_size: int = 16, num_workers: int = 4, device: str | torch.device = "cuda", ckpt_path: str | None = None, out_dir: str = "checkpoints", *, freeze_encoder=False, freeze_decoder=False, freeze_mixer=False, freeze_head=False):
    device = torch.device(device)

    ds = SimCol3DDataset(top_k=1, n_samples=256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=simcol3d_collate_fn)

    # Create model and optionally load pretrained weights
    if ckpt_path:
        print(f"[train] Loading pretrained weights from {ckpt_path}")
        model = SCRegNet.load_from_dust3r_weight(ckpt_path, head_type="dpt", output_mode="pts3d", show_mismatch=True).to(device)
    else:
        model = SCRegNet(has_conf=True).to(device)

    print("[before freeze] Model parameters")
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable = {n_train}/{n_total}  ({n_train/n_total:.1%})")
    # Freeze modules as requested
    model.set_freeze(encoder=freeze_encoder, decoder=freeze_decoder, mixer=freeze_mixer, head=freeze_head)
    print("[after freeze] Model parameters")
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable = {n_train}/{n_total}  ({n_train/n_total:.1%})")
    # Build optimizer with trainable params only
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(trainable_params)
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler()

    # Prepare output dirs
    os.makedirs(out_dir, exist_ok=True)
    log_path = Path(out_dir) / "train_log.txt"

    for epoch in range(epochs):
        loss = train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch)
        log_str = f"{datetime.now().isoformat()}\tEpoch {epoch:03d}\tloss={loss:.6f}"
        print(log_str)
        with open(log_path, "a", encoding="utf-8") as f_log:
            f_log.write(log_str + "\n")

        # Save checkpoint every epoch
        ckpt_file = Path(out_dir) / f"epoch_{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, ckpt_file)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SCRegNet with φ38 + τ loss")
    parser.add_argument("--data_root", type=str, default="data/processed/SimCol3D", help="root folder of dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:0 ...")
    # Freeze flags
    parser.add_argument("--freeze_encoder", action="store_true", help="freeze ViT encoder")
    parser.add_argument("--freeze_decoder", action="store_true", help="freeze cross decoder")
    parser.add_argument("--freeze_mixer", action="store_true", help="freeze 3D mixer module")
    parser.add_argument("--freeze_head", action="store_true", help="freeze prediction head")
    parser.add_argument("--pretrained", type=str, default=None, help="path to Dust3r weight to load")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="folder to save checkpoints & log")
    cli_args = parser.parse_args()

    run_train(
        data_root=cli_args.data_root,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        device=cli_args.device,
        ckpt_path=cli_args.pretrained,
        out_dir=cli_args.out_dir,
        freeze_encoder=cli_args.freeze_encoder,
        freeze_decoder=cli_args.freeze_decoder,
        freeze_mixer=cli_args.freeze_mixer,
        freeze_head=cli_args.freeze_head,
    )