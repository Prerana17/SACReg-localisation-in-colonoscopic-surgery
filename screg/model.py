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


class PointEmbed(nn.Module):
    """Placeholder φ-encoding + MLP to map 3-D points to 256-D tokens."""

    def __init__(self, in_dim: int = 3, embed_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, pts2d3d: torch.Tensor) -> torch.Tensor:  # (B,N,5)
        # TODO: replace with φ encoding (frequency features, etc.)
        xyz = pts2d3d[..., 2:5]  # use xyz only for now
        return self.fc(xyz)


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
                 n_img_blocks: int = 4,
                 n_pt_blocks: int = 3,
                 embed_dim_img: int = 768,
                 embed_dim_pt: int = 256,
                 num_heads: int = 12):
        super().__init__()
        assert n_img_blocks == n_pt_blocks + 1, "Need one more image block than point blocks (I/P alternation)."

        # Shared linear projections
        self.up_pt = nn.Linear(embed_dim_pt, embed_dim_img)   # 256 → 768
        self.down_pt = nn.Linear(embed_dim_img, embed_dim_pt)  # 768 → 256

        self.img_blocks = nn.ModuleList([DecoderBlock(embed_dim_img, num_heads) for _ in range(n_img_blocks)])
        self.pt_blocks = nn.ModuleList([PointLevelDecoderBlock(embed_dim_pt, num_heads) for _ in range(n_pt_blocks)])

    def forward(self,
                img_tok: torch.Tensor,   # (B,S,768)
                img_pos: torch.Tensor,   # (B,S,2)
                pt_tok: torch.Tensor     # (B,N,256)
                ) -> torch.Tensor:
        B = img_tok.shape[0]
        img_seq = img_tok
        pt_seq = pt_tok

        for img_block, pt_block in zip(self.img_blocks[:-1], self.pt_blocks):
            # Image-level decoder block (self+cross with point as mem)
            img_seq, _ = img_block(img_seq, self.up_pt(pt_seq), img_pos, img_pos)  # treat pt as y after up-proj
            # Point-level block (cross only)
            pt_seq = pt_block(pt_seq, self.down_pt(img_seq), img_pos, img_pos)
        # Final image block
        img_seq, _ = self.img_blocks[-1](img_seq, self.up_pt(pt_seq), img_pos, img_pos)
        return img_seq  # (B,S,768)


class SCRegNet(CroCoNet):
    """SCRegNet built on CroCoNet with 3DMixer and custom heads."""

    def __init__(self,
                 output_mode: str = 'pts3d',
                 head_type: str = 'dpt',
                 depth_mode: Tuple[str, float, float] = ('exp', -float('inf'), float('inf')),
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
        self.point_embed = PointEmbed(in_dim=3, embed_dim=256)
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

        model = cls(head_type=extra.get('head_type', head_type),
                    output_mode=extra.get('output_mode', output_mode),
                    depth_mode=extra.get('depth_mode', ("exp", -float('inf'), float('inf'))),
                    conf_mode=extra.get('conf_mode', ("exp",1,float('inf'))),
                    has_conf=has_conf_detect,
                    **croco_conf)
        missing, unexpected = model.load_state_dict(state, strict=strict)
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
    def _cross_decoder(self,
                       q_feat: torch.Tensor, q_pos: torch.Tensor,
                       kv_feat: torch.Tensor, kv_pos: torch.Tensor,
                       return_all: bool = False):
        """Run CroCo decoder where Q comes from q_feat and K/V from kv_feat."""
        # Inherit CroCo decoder but feed kv as memory (feat2) and ignore masks.
        return self._decoder(q_feat, q_pos, None, kv_feat, kv_pos, return_all_blocks=return_all)

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
        feat_q, pos_q, _ = self._encode_image(query_img, do_mask=False)
        feat_b, pos_b, _ = self._encode_image(retrieved_img, do_mask=False)

        # 3-D point embedding
        pt_tok = self.point_embed(correspondences)  # (B,N,256)

        # 3DMixer fusion → enhanced image tokens
        feat_b_mix = self.mixer(feat_b, pos_b, pt_tok)  # (B,S,768)

        # Cross-attention decoder
        decout = self._cross_decoder(feat_q, pos_q, feat_b_mix, pos_b, return_all=self.return_all_layers)

        # Head
        H, W = query_img.shape[-2:]
        result = self.head(decout, (H, W))  # type: ignore[arg-type]
        return result
