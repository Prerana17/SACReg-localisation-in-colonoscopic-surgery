from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import argparse
import os
from datetime import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # for error heatmap visualization
import plotly.graph_objects as go  # interactive 3-D visualisation
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .datasets import SimCol3DDataset, simcol3d_collate_fn
from .model import SCRegNet, PointEmbed
# --- Switched to direct 3-D regression ---
from .losses import Regr3D_World, ConfLoss_World, L21
from .utils import create_normalized_uv_grid_torch, phi_encode_xyz_torch, compute_phi_frequencies

# -----------------------------------------------------------------------------
# φ-encoding helper (vectorised torch) – matches PointEmbed 38-D scheme
# (Deprecated: now regress XYZ directly; kept for reference)
# -----------------------------------------------------------------------------
F1_DEFAULT = 125.6637
GAMMA_DEFAULT = 2.1867
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

    # 使用统一的频率计算
    freqs_np = compute_phi_frequencies(f1, gamma, F_HARMONICS)
    freqs = torch.from_numpy(freqs_np).to(device=device, dtype=dtype)

    # 统一的UV网格生成
    uv = create_normalized_uv_grid_torch(H, W, device=device, dtype=dtype)  # (2,H,W)
    uv = uv.expand(B, -1, -1, -1)  # (B,2,H,W)

    # φ编码 XYZ坐标
    xyz_permuted = xyz.permute(0, 2, 3, 1)  # (B,H,W,3)
    xyz_encoded = phi_encode_xyz_torch(xyz_permuted, freqs)  # (B,H,W,36)
    xyz_encoded = xyz_encoded.permute(0, 3, 1, 2)  # (B,36,H,W)

    return torch.cat([uv, xyz_encoded], dim=1)  # (B,38,H,W)


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


def build_scheduler(optimizer: torch.optim.Optimizer, *, warmup_steps: int = 500, total_steps: int | None = None):
    """Return *batch-level* scheduler: linear warm-up then cosine decay.

    Parameters
    ----------
    optimizer : torch Optimizer
    warmup_steps : number of steps for linear warm-up
    total_steps : total training steps (warmup+cosine). If *None*, cosine is disabled.
    """
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    # --- Linear warm-up ---
    warmup_sched = LinearLR(optimizer, start_factor=1.0 / max(1, warmup_steps), end_factor=1.0, total_iters=warmup_steps)

    # --- Cosine decay ---
    if total_steps is None or total_steps <= warmup_steps:
        # Only warm-up
        return warmup_sched

    cosine_iters = total_steps - warmup_steps
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_iters)

    # Sequential: warm-up → cosine
    sched = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    return sched


# -----------------------------------------------------------------------------
# PnP pose evaluation
# -----------------------------------------------------------------------------

def pose_eval(pts3d_pred: torch.Tensor, conf: torch.Tensor, K: torch.Tensor, pose_gt: torch.Tensor, max_samples: int = 4096):
    """Return translation & rotation error (m, deg) using cv2 PnP.

    pts3d_pred : (H,W,3) world coords tensor (cpu)
    conf       : (H,W) confidence 0-1 tensor (cpu)
    K          : (3,3) intrinsics tensor or ndarray
    pose_gt    : (4,4) world→cam ground-truth (numpy)
    """
    H, W = pts3d_pred.shape[:2]
    
    # Convert inputs to numpy
    if isinstance(K, torch.Tensor):
        K = K.numpy()
    if isinstance(pose_gt, torch.Tensor):
        pose_gt = pose_gt.numpy()
    
    pts3d_np = pts3d_pred.numpy()
    conf_np = conf.numpy()
    
    # Create pixel coordinates grid (standard OpenCV convention)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pixels = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)  # (H*W, 2) as (x, y)
    
    # Flatten 3D points and confidence
    pts3d_flat = pts3d_np.reshape(-1, 3)  # (H*W, 3)
    conf_flat = conf_np.reshape(-1)       # (H*W,)
    
    # Filter valid points (finite coordinates)
    finite_mask = (
        np.isfinite(pts3d_flat).all(axis=1) & 
        np.isfinite(conf_flat)
    )
    
    if finite_mask.sum() < 6:  # Need at least 6 points for PnP
        # print(f"[pose_eval] Insufficient finite points: {finite_mask.sum()}/6")
        return float('nan'), float('nan'), None
    
    # Apply finite mask first
    finite_pts3d = pts3d_flat[finite_mask]
    finite_pixels = pixels[finite_mask]
    finite_conf = conf_flat[finite_mask]
    
    # SACReg strategy: filter out points below confidence median
    conf_median = np.median(finite_conf)
    high_conf_mask = finite_conf >= conf_median
    
    if high_conf_mask.sum() < 6:  # Need at least 6 points for PnP
        # print(f"[pose_eval] Insufficient high-conf points: {high_conf_mask.sum()}/6 (median conf: {conf_median:.4f})")
        return float('nan'), float('nan'), None
    
    # Select high confidence points
    high_conf_pts3d = finite_pts3d[high_conf_mask]
    high_conf_pixels = finite_pixels[high_conf_mask]
    high_conf_values = finite_conf[high_conf_mask]
    
    # Random sampling from high confidence points (not weighted)
    n_high_conf = len(high_conf_pts3d)
    if n_high_conf > max_samples:
        # Random sampling without replacement
        indices = np.random.choice(n_high_conf, size=max_samples, replace=False)
        valid_pts3d = high_conf_pts3d[indices]
        valid_pixels = high_conf_pixels[indices]
    else:
        valid_pts3d = high_conf_pts3d
        valid_pixels = high_conf_pixels
    
    try:
        # Solve PnP using RANSAC for robustness
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            valid_pts3d.astype(np.float32),  # 3D world points
            valid_pixels.astype(np.float32), # 2D image points
            K.astype(np.float32),            # camera intrinsics
            None,                            # no distortion
            iterationsCount=2000,            # Increased iterations
            reprojectionError=5.0,           # More tolerant error threshold
            confidence=0.95                  # Slightly lower confidence requirement
        )
        
        if not success or inliers is None or len(inliers) < 4:  # Reduced from 6 to 4 (minimum for PnP)
            inlier_count = len(inliers) if inliers is not None else 0
            # print(f"[pose_eval] PnP RANSAC failed: success={success}, inliers={inlier_count}/4")
            return float('nan'), float('nan'), None
        
        # Convert rotation vector to rotation matrix
        R_pnp, _ = cv2.Rodrigues(rvec)
        t_pnp = tvec.flatten()
        
        # OpenCV PnP returns camera pose in world coords (cam→world transform)
        # But our GT pose is world→cam transform, so we need to invert PnP result
        T_pnp_world2cam = np.eye(4)
        T_pnp_world2cam[:3, :3] = R_pnp.T  # Invert rotation
        T_pnp_world2cam[:3, 3] = -R_pnp.T @ t_pnp  # Invert translation
        
        # Extract ground truth world→cam transform
        R_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:3, 3]
        
        # Calculate translation error (camera center positions in world coords)
        # Camera center: c = -R^T @ t (for world→cam transform)
        c_gt = -R_gt.T @ t_gt
        c_est = -T_pnp_world2cam[:3, :3].T @ T_pnp_world2cam[:3, 3]
        trans_err = np.linalg.norm(c_est - c_gt)
        
        # Calculate rotation error (angle between rotation matrices)
        R_est = T_pnp_world2cam[:3, :3]
        R_rel = R_gt.T @ R_est
        # Rotation angle from trace: θ = arccos((tr(R) - 1) / 2)
        trace = np.trace(R_rel)
        # Clamp to valid range for arccos to avoid numerical issues
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        rot_err = np.degrees(np.arccos(cos_angle))
        
        return float(trans_err), float(rot_err), T_pnp_world2cam
        
    except cv2.error as e:
        print(f"[pose_eval] OpenCV error: {e}")
        return float('nan'), float('nan'), None
    except Exception as e:
        print(f"[pose_eval] Unexpected error: {e}")
        return float('nan'), float('nan'), None


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_one_epoch(model: SCRegNet, loader: DataLoader, optimizer, scheduler, scaler, device, epoch, *, writer: SummaryWriter, val_interval: int, out_dir: Path, step_offset: int = 0, val_loader: DataLoader | None = None):
    """Train model for one epoch and log to tensorboard.

    Returns final global step index and last (trans_err, rot_err).
    """
    # Instantiate 3-D regression + confidence loss (once per epoch)
    loss_fn = ConfLoss_World(Regr3D_World(L21))
    model.train()
    total_loss = 0.0
    # Prepare iterator over validation loader if provided
    val_iter = iter(val_loader) if val_loader is not None else None
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    step = step_offset
    last_trans_err = float('nan')
    last_rot_err = float('nan')
    interval_steps = max(1, len(loader)//10)
    # Directory to save visualisations
    viz_dir = Path(out_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(pbar):
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

        # ------------------------------------------------------------------
        # Deprecated φ38 pipeline ⇒ switch to direct XYZ regression
        # phi_gt = encode_phi38(xyz_gt_rs)
        # ------------------------------------------------------------------

        with autocast():
            outputs = model(query_img, db_img, uvxyz)
            pts_pred = outputs["pts3d"]  # (B,H,W,3)
            tau = outputs["conf"]        # (B,H,W)

            # Build GT / pred dicts for 3-D loss
            gt_dict = {
                "pts3d": xyz_gt_rs.permute(0, 2, 3, 1),      # (B,H,W,3)
                "valid_mask": torch.isfinite(xyz_gt_rs).all(dim=1),
            }
            pred_dict = {"pts3d": pts_pred, "conf": tau}

            loss, _ = loss_fn(gt_dict, pred_dict)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item() * B
        writer.add_scalar("train/L21_loss", loss.item(), global_step=step)

        # Validation using external val_loader every val_interval steps
        if (step % val_interval == 0) and val_iter is not None:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)

                # Move tensors to device
                for k, v in val_batch.items():
                    if torch.is_tensor(v):
                        val_batch[k] = v.to(device, non_blocking=True)

                v_query_img = val_batch["query_img"]
                v_db_img = val_batch["db_img"]
                v_uvxyz = val_batch["uvxyz"]

                v_outputs = model(v_query_img, v_db_img, v_uvxyz)
                v_pts_pred = v_outputs["pts3d"]
                v_tau = v_outputs["conf"]

                B_val = v_pts_pred.shape[0]
                n_eval = min(1, B_val)
                sel_idx = torch.randperm(B_val, device=v_pts_pred.device)[:n_eval]
                trans_err_list, rot_err_list = [], []
                saved_viz = False  # only save once per epoch
                for idx in sel_idx:
                    pts3d_cpu = v_pts_pred[idx].detach().cpu()
                    conf_cpu = v_tau[idx].detach().cpu()
                    if "K" not in val_batch or "query_pose" not in val_batch:
                        continue
                    K_cpu = val_batch["K"][idx].cpu()
                    # print("[val] K_scaled:\n", K_cpu.numpy())
                    pose_gt = val_batch["query_pose"][idx]
                    if isinstance(pose_gt, torch.Tensor):
                        pose_gt = pose_gt.cpu().numpy()
                    t_err, r_err, pose_pred = pose_eval(pts3d_cpu, conf_cpu, K_cpu, pose_gt)

                    # —— Baseline: use ground-truth xyz itself ——
                    xyz_gt_full = val_batch["query_xyz"][idx]  # (3,H,W)
                    H_gt, W_gt = xyz_gt_full.shape[1:]
                    xyz_gt_rs = F.interpolate(xyz_gt_full.unsqueeze(0), size=(pts3d_cpu.shape[0], pts3d_cpu.shape[1]), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
                    xyz_gt_cpu = xyz_gt_rs.detach().cpu()
                    # --- Debug: compare predicted vs GT xyz ranges ---
                    # print("pred xyz [min,max,z+]:", pts3d_cpu.min().item(),
                        #   pts3d_cpu.max().item(), pts3d_cpu[..., 2].min().item())
                    # print("gt xyz   [min,max,z+]:", xyz_gt_cpu.min().item(),
                        #   xyz_gt_cpu.max().item(), xyz_gt_cpu[..., 2].min().item())
                    ones_conf = torch.ones((xyz_gt_cpu.shape[0], xyz_gt_cpu.shape[1]), dtype=torch.float32)
                    t_err_gt, r_err_gt, pose_gtxyz = pose_eval(xyz_gt_cpu, ones_conf, K_cpu, pose_gt)

                    # Print results
                    np.set_printoptions(precision=4, suppress=True)
                    # print("[val] GT pose (world→cam):\n", pose_gt)
                    # print("[val] Pred pose (world→cam):\n", pose_pred)
                    # print("[val] PnP(pred)  trans_err={:.4f} m, rot_err={:.2f}°".format(t_err, r_err))
                    # print("[val] PnP(GT xyz) trans_err={:.4f} m, rot_err={:.2f}°".format(t_err_gt, r_err_gt))

                    # ---- Save visualisations once ----
                    if not saved_viz:
                        # Query RGB
                        rgb_t = v_query_img[idx].detach().cpu()  # (3,H,W)
                        rgb_np = (rgb_t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                        cv2.imwrite(str(viz_dir / f"epoch_{epoch:03d}_rgb.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

                        def save_xyz_img(xyz_tensor, fname):
                            arr = xyz_tensor.detach().cpu().numpy()
                            mn, mx = arr.min(), arr.max()
                            if mx - mn < 1e-6:
                                mx = mn + 1e-6
                            arr = (arr - mn) / (mx - mn)
                            img = (arr * 255).astype(np.uint8)
                            cv2.imwrite(str(viz_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                        save_xyz_img(pts3d_cpu, f"epoch_{epoch:03d}_pred_xyz.png")
                        save_xyz_img(xyz_gt_cpu, f"epoch_{epoch:03d}_gt_xyz.png")

                        # ---- 3-D camera pose visualisation ----
                        def cam_center_from_T(T):
                            R = T[:3, :3]
                            t = T[:3, 3]
                            return (-R.T @ t).reshape(3)

                        poses = {
                            "GT pose": pose_gt,
                            "PnP(pred)": pose_pred,
                            "PnP(GT xyz)": pose_gtxyz,
                        }
                        fig = go.Figure()
                        colors = {"GT pose": "green", "PnP(pred)": "red", "PnP(GT xyz)": "blue"}
                        scale = 0.05  # arrow length in metres
                        for name, T in poses.items():
                            if T is None or np.isnan(T).any():
                                continue
                            c = cam_center_from_T(T)
                            R = T[:3, :3]
                            forward = R.T @ np.array([0, 0, -scale])  # -Z axis
                            end = c + forward
                            fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers", marker=dict(size=4, color=colors[name]), name=name))
                            fig.add_trace(go.Scatter3d(x=[c[0], end[0]], y=[c[1], end[1]], z=[c[2], end[2]], mode="lines", line=dict(color=colors[name]), showlegend=False))

                        fig.update_layout(scene=dict(aspectmode="data"), title=f"Epoch {epoch}: Camera Poses")
                        fig.write_html(str(viz_dir / f"epoch_{epoch:03d}_poses.html"))

                        # ---- Confidence map ----
                        conf_np = conf_cpu.numpy()
                        # Normalise for visualisation
                        mn_c, mx_c = conf_np.min(), conf_np.max()
                        if mx_c - mn_c < 1e-6:
                            mx_c = mn_c + 1e-6
                        conf_vis = (conf_np - mn_c) / (mx_c - mn_c)
                        # Optional: print stats once
                        # print(f"[val] conf range: {mn_c:.4f} – {mx_c:.4f}")
                        plt.figure(figsize=(4, 4), dpi=200)
                        plt.imshow(conf_vis, cmap="viridis")
                        plt.colorbar(label="Pred confidence")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(str(viz_dir / f"epoch_{epoch:03d}_conf.png"))
                        plt.close()

                        # ---- Pixel-wise error heatmap ----
                        err_np = np.linalg.norm(pts3d_cpu.numpy() - xyz_gt_cpu.numpy(), axis=2)
                        plt.figure(figsize=(4, 4), dpi=200)
                        plt.imshow(err_np, cmap="inferno")
                        plt.colorbar(label="XYZ error (m)")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(str(viz_dir / f"epoch_{epoch:03d}_err_heatmap.png"))
                        plt.close()
                        saved_viz = True
                    trans_err_list.append(t_err)
                    rot_err_list.append(r_err)
                if trans_err_list:
                    last_trans_err = float(np.nanmean(trans_err_list))
                    last_rot_err = float(np.nanmean(rot_err_list))
                    writer.add_scalar("val/trans_err", last_trans_err, global_step=step)
                    writer.add_scalar("val/rot_err", last_rot_err, global_step=step)
            # restore training mode
            if was_training:
                model.train()

        # Update progress bar postfix with loss and, if available, validation PnP(pred) errors
        postfix = {"loss": f"{loss.item():.4f}"}
        if not math.isnan(last_trans_err):
            postfix["t_err(m)"] = f"{last_trans_err:.3f}"
            postfix["r_err(°)"] = f"{last_rot_err:.2f}"
        pbar.set_postfix(**postfix)

        # periodic checkpoint (every 10% progress) or last batch
        # if ((batch_idx + 1) % interval_steps == 0) or (batch_idx + 1 == len(loader)):
        #     ckpt_name = f"epoch_{epoch:03d}_step_{batch_idx+1:06d}.pth"
        #     torch.save({
        #         "epoch": epoch,
        #         "step": batch_idx + 1,
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "scheduler": scheduler.state_dict(),
        #     }, Path(out_dir) / ckpt_name)

        step += 1

    # log last values after epoch
    if not math.isnan(last_trans_err):
        writer.add_scalar("val/trans_err", last_trans_err, global_step=step)
        writer.add_scalar("val/rot_err", last_rot_err, global_step=step)

    return step, total_loss / len(loader.dataset), last_trans_err, last_rot_err


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_train(data_root: str | Path = "data/processed/SimCol3D", epochs: int = 100, batch_size: int = 16, num_workers: int = 4, device: str | torch.device = "cuda", ckpt_path: str | None = None, out_dir: str = "checkpoints", val_interval: int = 50, log_dir: str | None = None, note: str | None = None, *, freeze_encoder=False, freeze_decoder=False, freeze_mixer=False, freeze_head=False, resume: str | None = None):
    device = torch.device(device)

    # Auto-create unique experiment directory
    out_dir = Path(out_dir)
    if out_dir.name == "checkpoints":
        timestamp = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        name = timestamp
        if note:
            safe_note = "_" + "_".join(note.strip().split())
            name += safe_note
        out_dir = out_dir / name
    os.makedirs(out_dir, exist_ok=True)

    ds = SimCol3DDataset(top_k=1, n_samples=1024, variants=("SyntheticColon_I", "SyntheticColon_II"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=simcol3d_collate_fn)

    # Validation dataset from SyntheticColon_III
    val_ds = SimCol3DDataset(top_k=1, n_samples=1024, variants=("SyntheticColon_III",))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=simcol3d_collate_fn)

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
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    scheduler = build_scheduler(optimizer, warmup_steps=200, total_steps=total_steps)
    scaler = GradScaler()

    # --- Resume from checkpoint (our own saved .pth) ---
    start_epoch = 0
    if resume is not None and Path(resume).is_file():
        print(f"[train] Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        # Model
        if "model" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            if missing:
                print(f"[resume] Missing keys: {len(missing)} (showing first 10)\n  {missing[:10]}")
            if unexpected:
                print(f"[resume] Unexpected keys: {len(unexpected)} (showing first 10)\n  {unexpected[:10]}")
        # Optimizer / scheduler
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[resume] Optimizer state not loaded: {e}")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[resume] Scheduler state not loaded: {e}")
        # Epoch
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
            print(f"[resume] Starting from epoch {start_epoch}")

    # Prepare output dirs
    os.makedirs(out_dir, exist_ok=True)
    log_path = Path(out_dir) / "train_log.txt"

    # Build TensorBoard log directory with key init params
    tb_suffix = f"tb_bs{batch_size}_fe{int(freeze_encoder)}_fd{int(freeze_decoder)}_fm{int(freeze_mixer)}_fh{int(freeze_head)}"
    tb_dir = Path(log_dir) / tb_suffix if log_dir else (Path(out_dir) / tb_suffix)
    writer = SummaryWriter(tb_dir)
    # Record a human-readable summary of important hyper-parameters
    hparams_txt = (
        f"batch_size: {batch_size}\n"
        f"epochs: {epochs}\n"
        f"val_interval: {val_interval}\n"
        f"freeze_encoder: {freeze_encoder}\n"
        f"freeze_decoder: {freeze_decoder}\n"
        f"freeze_mixer: {freeze_mixer}\n"
        f"freeze_head: {freeze_head}\n"
        f"pretrained: {ckpt_path}\n"
        f"resume: {resume}\n"
        f"out_dir: {out_dir}\n"
        f"device: {device}"
    )
    writer.add_text("hparams/summary", hparams_txt, global_step=0)

    global_step = start_epoch * steps_per_epoch
    last_trans_err = float('nan')
    last_rot_err = float('nan')
    for epoch in range(start_epoch, epochs):
        global_step, avg_loss, last_trans_err, last_rot_err = train_one_epoch(
            model, loader, optimizer, scheduler, scaler, device, epoch,
            writer=writer, val_interval=val_interval, out_dir=Path(out_dir), step_offset=global_step, val_loader=val_loader)
        log_str = f"{datetime.now().isoformat()}\tEpoch {epoch:03d}\tloss={avg_loss:.6f}"
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
            "avg_loss": avg_loss,
            "val_trans_err": last_trans_err,
            "val_rot_err": last_rot_err,
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
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training from")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="folder to save checkpoints & log")
    parser.add_argument("--val_interval", type=int, default=50, help="steps between validation pose error eval")
    parser.add_argument("--log_dir", type=str, default=None, help="tensorboard log directory")
    parser.add_argument("--note", type=str, default=None, help="experiment note to append to folder name")
    cli_args = parser.parse_args()

    run_train(
        data_root=cli_args.data_root,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        device=cli_args.device,
        ckpt_path=cli_args.pretrained,
        resume=cli_args.resume,
        out_dir=cli_args.out_dir,
        val_interval=cli_args.val_interval,
        log_dir=cli_args.log_dir,
        note=cli_args.note,
        freeze_encoder=cli_args.freeze_encoder,
        freeze_decoder=cli_args.freeze_decoder,
        freeze_mixer=cli_args.freeze_mixer,
        freeze_head=cli_args.freeze_head,
    )