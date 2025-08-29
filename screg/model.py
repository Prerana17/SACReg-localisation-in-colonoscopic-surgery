"""SCRegNet
===============
Minimal skeleton implementation building on croco's `CroCoNet`.
Only required plumbing is provided; algorithmic details (φ‐encoding, full
3DMixer attention schedule, etc.) are left as TODO so that training can start
and imports work.
"""
from __future__ import annotations

from itertools import chain
from typing import List, Tuple

import torch
import torch.nn as nn
import sys
from pathlib import Path
import re
import ast

# ---------------------------------------------------------------------------
# Ensure croco submodule root is on sys.path so that top-level "models" import
# statements inside croco code resolve correctly.
# ---------------------------------------------------------------------------
_croco_root = Path(__file__).resolve().parent.parent / "croco"
if _croco_root.is_dir() and str(_croco_root) not in sys.path:
    sys.path.append(str(_croco_root))

# CroCo
from croco.models.croco import CroCoNet
from croco.models.blocks import DecoderBlock

# Local heads
from .heads import head_factory
from .utils import compute_phi_frequencies


class PointEmbed(nn.Module):
    """Point encoding module implementing the φ-encoding described in SACReg.

    The module takes input points (u, v, x, y, z) and:
      1. Applies harmonic cosine/sine encoding to the xyz coordinates with
         frequencies  f_i = f1 * gamma^(i-1)  for i = 1..F.
      2. Concatenates the raw (u, v) coordinates giving a 38-D vector
         (uv 2 + xyz encoding 36).
      3. Projects the 38-D vector through a two-layer MLP to obtain a 256-D
         token suitable for the 3D-Mixer.

    No normalisation or rescaling is applied to inputs.
    """

    def __init__(
        self,
        f1: float = 125.6637,
        gamma: float = 2.1867,
        F: int = 6,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        # 使用统一的频率计算函数
        # Computed for 20cm level scene size and 1mm regression precision according paper
        freqs_np = compute_phi_frequencies(f1, gamma, F)
        freqs = torch.from_numpy(freqs_np)
        self.register_buffer("frequencies", freqs)

        input_dim = 2 + 3 * F * 2  # 38 when F=6
        # Two-layer MLP 38 -> 256 -> 256 with ReLU
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, pts2d3d: torch.Tensor) -> torch.Tensor:
        """Encode (u,v,x,y,z) points into 256-D tokens.

        Parameters
        ----------
        pts2d3d : Tensor (B, N, 5)
            Input points with columns (u, v, x, y, z).

        Returns
        -------
        Tensor (B, N, 256)
            Encoded point tokens.
        """
        B, N, _ = pts2d3d.shape
        if N == 0:
            # Gracefully handle empty correspondence sets.
            return torch.empty(B, 0, self.mlp[-1].out_features, device=pts2d3d.device, dtype=pts2d3d.dtype)

        uv = pts2d3d[..., 0:2]  # (B, N, 2)
        xyz = pts2d3d[..., 2:5]  # (B, N, 3)

        # 使用统一的φ编码函数
        from .utils import phi_encode_xyz_torch
        enc_xyz = phi_encode_xyz_torch(xyz, self.frequencies)  # (B, N, 36)

        # Concatenate (u, v)
        feat = torch.cat([uv, enc_xyz], dim=-1)  # (B, N, 38)

        return self.mlp(feat)


# ---------------------------------------------------------------------------
# Standard transformer-based decoder blocks (image & point) for 3D-Mixer
# ---------------------------------------------------------------------------
class StandardImageDecoderBlock(nn.Module):
    """Classic Transformer decoder block with self-attention on image tokens
    followed by cross-attention with point tokens.
    """
    def __init__(self, dim: int = 1024, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # Cross-attention (keys/values from mem)
        x = x + self.cross_attn(self.ln2(x), mem, mem)[0]
        # Feed-forward
        x = x + self.mlp(self.ln3(x))
        return x  # (B, S, C)


class StandardPointDecoderBlock(nn.Module):
    """Point-level block: only cross-attention (no self-attention)."""
    def __init__(self, dim_pt: int = 256, dim_img: int = 1024, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim_pt)
        self.cross_attn = nn.MultiheadAttention(dim_pt, num_heads, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(dim_pt)
        hidden_dim = int(dim_pt * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim_pt, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_pt),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Cross-attention only (query = point tokens, key/value = image tokens)
        q = q + self.cross_attn(self.ln_q(q), kv, kv)[0]
        q = q + self.mlp(self.ln_ff(q))
        return q  # (B, N, dim_pt)


# ---------------------------------------------------------------------------
# Existing croco-based PointLevelDecoderBlock remains for backward compatibility
# ---------------------------------------------------------------------------
class PointLevelDecoderBlock(nn.Module):
    """DecoderBlock variant without self-attention (SA)."""

    def __init__(self, dim: int, num_heads: int = 12):
        super().__init__()
        # remove SA, keep cross-attention + MLP
        self.cross_attn_block = DecoderBlock(dim, num_heads, norm_mem=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor, xpos, ypos):
        # we simply ignore the self-attn path inside DecoderBlock by feeding zeros
        dummy = torch.zeros_like(x, requires_grad=False)
        out, _ = self.cross_attn_block(dummy, y, xpos, ypos)  # SA uses dummy, effectively no-op
        # return updated x (point tokens)
        return out


class ThreeDMixer(nn.Module):
    """Alternate image-level and point-level decoder blocks to fuse features."""

    def __init__(self,
                 n_img_blocks: int = 3,
                 n_pt_blocks: int = 2,
                 embed_dim_img: int = 1024,
                 embed_dim_pt: int = 256,
                 num_heads: int = 12):
        super().__init__()
        print("\n[ThreeDMixer] constructor kwargs ↓")
        print(f"n_img_blocks: {n_img_blocks}")
        print(f"n_pt_blocks: {n_pt_blocks}")
        print(f"embed_dim_img: {embed_dim_img}")
        print(f"embed_dim_pt: {embed_dim_pt}")
        print(f"num_heads: {num_heads}")
        assert n_img_blocks == n_pt_blocks + 1, "Need one more image block than point blocks (I/P alternation)."

        # Shared linear projections
        self.up_pt = nn.Linear(embed_dim_pt, embed_dim_img)   # 256 → 1024
        self.down_pt = nn.Linear(embed_dim_img, embed_dim_pt)  # 1024 → 256

        # Use classic Transformer implementations
        self.img_blocks = nn.ModuleList([StandardImageDecoderBlock(embed_dim_img, 8) for _ in range(n_img_blocks)])
        self.pt_blocks = nn.ModuleList([StandardPointDecoderBlock(embed_dim_pt, embed_dim_img, 8) for _ in range(n_pt_blocks)])

    def forward(self,
                img_tok: torch.Tensor,   # (B,S,1024)
                img_pos: torch.Tensor,   # (B,S,2)
                pt_tok: torch.Tensor     # (B,N,256)
                ) -> torch.Tensor:
        B = img_tok.shape[0]
        img_seq = img_tok
        pt_seq = pt_tok

        for img_block, pt_block in zip(self.img_blocks[:-1], self.pt_blocks):
            # Image-level decoder block (self + cross-attention)
            img_seq = img_block(img_seq, self.up_pt(pt_seq))
            # Point-level block (cross only)
            pt_seq = pt_block(pt_seq, self.down_pt(img_seq))
        # Final image block
        img_seq = self.img_blocks[-1](img_seq, self.up_pt(pt_seq))
        return img_seq  # (B,S,1024)


class SCRegNet(CroCoNet):
    """SCRegNet built on CroCoNet with 3DMixer and custom heads."""

    def __init__(self,
                 output_mode: str = 'pts3d',
                 head_type: str = 'dpt',
                 depth_mode: Tuple[str, float, float] = ('linear', -float('inf'), float('inf')),
                 conf_mode: Tuple[str, float, float] = ('exp', 1, float('inf')),
                 has_conf: bool = True,
                 **croco_kwargs):
        # Force compatible dims with ViT-Base
        defaults = dict(enc_embed_dim=1024,
                         dec_embed_dim=768,
                         enc_num_heads=12,
                         dec_num_heads=12)
        defaults.update(croco_kwargs)
        super().__init__(**defaults)

        # Store inference modes
        self.output_mode = output_mode
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        # Extra modules
        self.point_embed = PointEmbed(embed_dim=256)
        self.mixer = ThreeDMixer()

        # Downstream head
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=has_conf)
        self.return_all_layers = getattr(self.downstream_head1, 'return_all_layers', False)

        # ------------------------------------------------------------------
        # Log structure hyper-parameters for debugging
        # ------------------------------------------------------------------
        struct_params = {
            'enc_depth': defaults.get('enc_depth', len(getattr(self, 'enc_blocks', []))),
            'dec_depth': defaults.get('dec_depth', len(getattr(self, 'dec_blocks', []))),
            'enc_embed_dim': defaults.get('enc_embed_dim'),
            'dec_embed_dim': defaults.get('dec_embed_dim'),
            'enc_num_heads': defaults.get('enc_num_heads'),
            'dec_num_heads': defaults.get('dec_num_heads'),
            'pos_embed': defaults.get('pos_embed', croco_kwargs.get('pos_embed', 'sincos')),
            'patch_embed_cls': defaults.get('patch_embed_cls', croco_kwargs.get('patch_embed_cls', 'PatchEmbed')),
            'img_size': defaults.get('img_size', croco_kwargs.get('img_size', (224, 224))),
            'has_conf': has_conf,
        }
        print('[SCRegNet] Structure:', ', '.join(f"{k}={v}" for k, v in struct_params.items()))
    
    def _set_prediction_head(self, dec_embed_dim, patch_size):
        # 不创建 prediction_head，或者创建一个占位符
        pass

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------
    @classmethod
    def load_from_dust3r_weight(cls,
                        ckpt_path: str,
                        map_location: str | torch.device = 'cpu',
                        strict: bool = False,
                        head_type: str = 'dpt',
                        output_mode: str = 'pts3d',
                        show_mismatch: bool = True,
                        **kwargs):
        """Instantiate *and* load weights from a checkpoint.

        The checkpoint can be either:
        1. a plain *state_dict* (mapping parameter names → tensors)
        2. a dict containing key ``'state_dict'`` or ``'model'`` holding the state_dict.
        """
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        else:
            state = ckpt

        # --------------------------------------------------------------
        # Auto-configure SCRegNet according to checkpoint args, if any
        # --------------------------------------------------------------
        extra = {}
        croco_conf = {}
        if isinstance(ckpt, dict) and 'args' in ckpt:
            args_obj = ckpt['args']
            if hasattr(args_obj, '__dict__'):
                args_dict = vars(args_obj)
            elif isinstance(args_obj, dict):
                args_dict = args_obj
            else:
                args_dict = {}
            # Parse "model" string like "AsymmetricCroCo3DStereo(enc_depth=24, dec_depth=12, ...)"
            model_str = args_dict.get('model', '')
            m_kv = re.findall(r"(\w+)=([^,()]+)", model_str)
            for k, v in m_kv:
                try:
                    args_dict[k] = ast.literal_eval(v)
                except Exception:
                    args_dict[k] = v.strip("'\"")
            # Special-case img_size tuple not captured correctly above
            if 'img_size' not in args_dict:
                m_sz = re.search(r"img_size=\((\d+)\s*,\s*(\d+)\)", model_str)
                if m_sz:
                    args_dict['img_size'] = (int(m_sz.group(1)), int(m_sz.group(2)))
            # CroCo structural keys to forward
            for k in ('enc_depth','dec_depth','enc_embed_dim','dec_embed_dim','enc_num_heads','dec_num_heads','pos_embed','patch_embed_cls','img_size'):
                if k in args_dict:
                    croco_conf[k] = args_dict[k]
            # SCRegNet head / mode keys
            for k in ('head_type','output_mode','depth_mode','conf_mode'):
                if k in args_dict:
                    extra[k] = args_dict[k]
        # fallback to provided kwargs overrides
        # remove keys not accepted by CroCoNet
        valid_croco_keys = {'img_size','patch_size','mask_ratio','enc_embed_dim','enc_depth','enc_num_heads','dec_embed_dim','dec_depth','dec_num_heads','mlp_ratio','norm_layer','norm_im2_in_dec','pos_embed'}
        croco_conf = {k:v for k,v in croco_conf.items() if k in valid_croco_keys}
        croco_conf.update(kwargs)
        # Ensure img_size is tuple, not list (CroCo expects tuple)
        if 'img_size' in croco_conf and isinstance(croco_conf['img_size'], list):
            croco_conf['img_size'] = tuple(croco_conf['img_size'])

        # --- DEBUG log ---
        if 'img_size' in croco_conf:
            print('[DEBUG] croco_conf.img_size', croco_conf['img_size'])
        else:
            print('[DEBUG] img_size missing in croco_conf')
        # Detect whether checkpoint head outputs confidence channel (4 vs 3)
        has_conf_detect = None
        # 1) DPT head
        for k, v in state.items():
            if k.endswith('.dpt.head.4.weight') and isinstance(v, torch.Tensor):
                has_conf_detect = (v.shape[0] == 4)
                break
        # 2) Linear head
        if has_conf_detect is None:
            for k, v in state.items():
                if k.endswith('.proj.weight') and isinstance(v, torch.Tensor):
                    out_features = v.shape[0]
                    # try to guess patch_size from input dim later if needed; assume 16
                    for p in (16, 8):
                        if out_features == 4 * p * p:
                            has_conf_detect = True
                            break
                        if out_features == 3 * p * p:
                            has_conf_detect = False
                            break
                    if has_conf_detect is not None:
                        break
        if has_conf_detect is None:
            has_conf_detect = False

        # import pprint
        # print("\n[SCRegNet] constructor kwargs ↓")
        # pprint.pprint(croco_conf)
        # pprint.pprint(extra)
        model = cls(head_type=extra.get('head_type', head_type),
                    output_mode=extra.get('output_mode', output_mode),
                    depth_mode=extra.get('depth_mode', ("exp", -float('inf'), float('inf'))),
                    conf_mode=extra.get('conf_mode', ("exp",1,float('inf'))),
                    has_conf=has_conf_detect,
                    **croco_conf)
        # ------------------------------------------------------------------
        # Filter out parameters whose shape does not match current model (e.g., head).
        # ------------------------------------------------------------------
        model_state = model.state_dict()
        compatible_state = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        if show_mismatch:
            if missing:
                print(f"[SCRegNet] Missing params ({len(missing)}):")
                for n in missing:
                    print("  -", n)
            if unexpected:
                print(f"[SCRegNet] Unexpected params ({len(unexpected)}):")
                for n in unexpected:
                    print("  -", n)
        return model

    # backward compatibility alias
    from_pretrained = load_from_dust3r_weight

    def load_weights(self, ckpt_path: str, map_location: str | torch.device = 'cpu', strict: bool = False):
        """Load model weights into existing instance."""
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        else:
            state = ckpt
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            print(f"[SCRegNet] Missing params: {len(missing)}")
        if unexpected:
            print(f"[SCRegNet] Unexpected params: {len(unexpected)}")
        return missing, unexpected

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    # def _cross_decoder(self,
    #                    q_feat: torch.Tensor, q_pos: torch.Tensor,
    #                    kv_feat: torch.Tensor, kv_pos: torch.Tensor,
    #                    return_all: bool = False):
    #     """Run CroCo decoder where Q comes from q_feat and K/V from kv_feat."""
    #     # Inherit CroCo decoder but feed kv as memory (feat2) and ignore masks.
    #     return self._decoder(q_feat, q_pos, None, kv_feat, kv_pos, return_all_blocks=return_all)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self,
                query_img: torch.Tensor,
                retrieved_img: torch.Tensor,
                correspondences: torch.Tensor,
                ) -> dict:
        """Compute 3-D prediction.

        Parameters
        ----------
        query_img : Tensor (B,3,H,W)
        retrieved_img : Tensor (B,3,H,W)
        correspondences : Tensor (B,N,5) – (u,v,x,y,z)
        """
        # ViT encoders (no masking)

        # print("\n[Q Image Shape 1 ] constructor kwargs ↓")
        # print(query_img.shape)
        # print("\n[B Image Shape 2 ] constructor kwargs ↓")
        # print(retrieved_img.shape)
        feat_q, pos_q, _ = self._encode_image(query_img, do_mask=False)
        feat_b, pos_b, _ = self._encode_image(retrieved_img, do_mask=False)
        # print("\n[Q Image Shape 2 ] constructor kwargs ↓")
        # print(feat_q.shape)
        # print("\n[B Image Shape 2 ] constructor kwargs ↓")
        # print(feat_b.shape)

        # 3-D point embedding
        # print("\n[ correspondences  Shape] constructor kwargs ↓")
        # print(correspondences.shape)
        pt_tok = self.point_embed(correspondences)  # (B,N,256)

        # print("\n[embedded correspondences  Shape] constructor kwargs ↓")
        # print(pt_tok.shape)


        # 3DMixer fusion → enhanced image tokens
        feat_b_mix = self.mixer(feat_b, pos_b, pt_tok)  # (B,S,768)
        # print("\n[feat_b_mix  Shape]  kwargs ↓")
        # print(feat_b_mix.shape)


        # feat_b_mix = self.decoder_embed(feat_b_mix)

        # Cross-attention decoder
        decout = self._decoder(feat_q, pos_q, None, feat_b_mix, pos_b, return_all_blocks=self.return_all_layers)
        
        # Combine encoder + decoder features for DPT head (matches dust3r expectation)
        # DPT expects: [encoder_feat, decoder_layer_6, decoder_layer_9, decoder_layer_11]
        combined_features = [feat_q] + decout  # feat_q is 1024-dim encoder output

        # Head
        H, W = query_img.shape[-2:]
        result = self.downstream_head1(combined_features, (H, W))
        return result

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def set_freeze(self, *, encoder: bool = False, decoder: bool = False, head: bool = False, mixer: bool = False):
        """Freeze/unfreeze sub-modules on demand.

        Passing True freezes the corresponding part (requires_grad=False). False unfreezes.
        """
        def _toggle(module, freeze_flag: bool):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = not freeze_flag

        # Encoder = patch embedding + encoder blocks + pos_embed param
        if hasattr(self, 'patch_embed'):
            _toggle(self.patch_embed, encoder)
        if hasattr(self, 'enc_blocks'):
            _toggle(self.enc_blocks, encoder)
        if hasattr(self, 'pos_embed') and isinstance(self.pos_embed, torch.nn.Parameter):
            self.pos_embed.requires_grad = not encoder

        # Decoder blocks
        if hasattr(self, 'dec_blocks'):
            _toggle(self.dec_blocks, decoder)

        # 3D Mixer
        _toggle(self.mixer, mixer)

        # Prediction head
        _toggle(self.downstream_head1, head)

        # Optionally log
        print(f"[set_freeze] encoder={encoder}, decoder={decoder}, mixer={mixer}, head={head}")


