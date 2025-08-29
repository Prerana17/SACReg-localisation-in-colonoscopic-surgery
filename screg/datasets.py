"""Datasets utilities for SCRegNet.

Provides a generic parent class `BasePairDataset` and a concrete implementation
`SimCol3DDataset` that reads the SimCol3D pre-processed structure produced by
`scripts/preprocess_SimCol3D.py`.

Each sample consists of a (query RGB image, database RGB image) pair plus a set
of randomly sampled 2-D/3-D correspondences from the database image.  A
per-pixel 38-D φ-encoding of the query image xyz map is also returned as the
supervision target.
"""
from __future__ import annotations

import random
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Sequence, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image

# Re-use harmonic frequencies from PointEmbed for consistency
from .model import PointEmbed
from .utils import create_normalized_uv_grid_numpy, phi_encode_xyz_numpy, compute_phi_frequencies

__all__ = ["BasePairDataset", "SimCol3DDataset", "simcol3d_collate_fn"]


class BasePairDataset(Dataset):
    """Abstract dataset that yields image pairs plus correspondence samples.

    Sub-classes must fill `self.pairs` in `__init__` as a list of tuples
    ``(query_rgb_path, db_rgb_path)`` **absolute** paths.
    """

    def __init__(
        self,
        pair_paths: Sequence[Tuple[Path, Path]],
        n_samples: int = 256,
        img_size: Tuple[int, int] = (512, 512),
        xyz_suffix: str = ".xyz.npy",
        rng: Optional[random.Random] = None,
        camera_K: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.pairs: List[Tuple[Path, Path]] = list(pair_paths)
        self.n_samples = n_samples
        self.img_size = img_size
        self.xyz_suffix = xyz_suffix
        self.camera_K = camera_K
        self.rng = rng or random.Random()

        # Report stats
        qs = {q for q, _ in self.pairs}
        print(
            f"[BasePairDataset] pairs={len(self.pairs)}, unique queries={len(qs)}, "
            f"samples/DB_img={self.n_samples}"
        )

    # ------------------------------------------------------------------
    # Abstract helpers (can be overridden by subclasses)
    # ------------------------------------------------------------------
    def _load_rgb(self, path: Path) -> torch.Tensor:
        """Load RGB image as float32 tensor in [0,1] of shape (C,H,W)."""
        img = Image.open(path).convert("RGB")
        return to_tensor(img)  # already float32 / 0-1

    def _load_pose(self, rgb_path: Path) -> np.ndarray:
        """Return 4×4 world→camera pose for *rgb_path*."""
        pose_path = Path(str(rgb_path).replace('.rgb.png', '.pose.txt'))
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        vals = np.fromstring(pose_path.read_text(), sep=" ", dtype=np.float32)
        if vals.size != 7:
            raise ValueError(f"Unexpected pose format: {pose_path}")
        # Positions are already in right-handed world coords (cm) → convert to metres
        t = vals[:3] * 0.01  # cm→m
        q = vals[3:]
        Rm = R.from_quat(q).as_matrix().astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = Rm
        pose[:3, 3] = t
        # print(rgb_path,pose)
        # print()
        return pose

    def _load_xyz(self, rgb_path: Path) -> np.ndarray:
        """Load xyz world coordinates matching *rgb_path*.
        Shape (H,W,3), dtype float32.
        """
        xyz_path = Path(str(rgb_path).replace('.rgb.png', '.xyz.npy'))
        if not xyz_path.exists():
            raise FileNotFoundError(xyz_path)
        return np.load(xyz_path).astype(np.float32)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _random_uv_indices(self, h: int, w: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample *k* unique (u,v) pixel indices within image size (h,w)."""
        if k > h * w:
            k = h * w
        flat_idx = self.rng.sample(range(h * w), k)
        vv, uu = np.divmod(flat_idx, w)  # v=row, u=col
        return np.asarray(uu), np.asarray(vv)

    def _normalise_uv(self, uu: np.ndarray, vv: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        u_n = 2.0 * uu / (w - 1) - 1.0
        v_n = 2.0 * vv / (h - 1) - 1.0
        return u_n.astype(np.float32), v_n.astype(np.float32)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        q_rgb_path, db_rgb_path = self.pairs[idx]

        # --- Load images & resize to model input size ---
        q_img = torch.nn.functional.interpolate(self._load_rgb(q_rgb_path).unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False).squeeze(0)
        db_img = torch.nn.functional.interpolate(self._load_rgb(db_rgb_path).unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False).squeeze(0)

        # --- Sample correspondences from DB image ---
        xyz_db = self._load_xyz(db_rgb_path)
        h, w = xyz_db.shape[:2]
        uu, vv = self._random_uv_indices(h, w, self.n_samples)
        u_n, v_n = self._normalise_uv(uu, vv, h, w)
        xyz_samples = xyz_db[vv, uu]  # (k,3)
        uvxyz = np.column_stack([u_n, v_n, xyz_samples.reshape(-1, 3)])  # (k,5)
        uvxyz_t = torch.from_numpy(uvxyz)

        # --- Query xyz target (no φ38 precompute) ---
        xyz_q = self._load_xyz(q_rgb_path)  # (H_orig,W_orig,3)
        xyz_q_t = torch.from_numpy(xyz_q).permute(2, 0, 1)  # (3,H,W)

        pose_q = self._load_pose(q_rgb_path)
        sample = {
            "query_img": q_img,
            "db_img": db_img,
            "uvxyz": uvxyz_t,  # (k,5)
            "query_xyz": xyz_q_t,  # (3,H,W)
            "query_path": str(q_rgb_path),
            "db_path": str(db_rgb_path),
            "query_pose": pose_q,  # (4,4)
        }
        # Lazy K load if not set
        if self.camera_K is None:
            cam_path = Path(q_rgb_path).parent.parent / "cam.txt"
            if cam_path.exists():
                vals = [float(v) for v in cam_path.read_text().strip().replace("\n", " ").split()]
                if len(vals) >= 6:
                    fx, fy, cx, cy = vals[0], vals[4], vals[2], vals[5]
                    self.camera_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if self.camera_K is not None:
            # Scale intrinsics to match the resized image resolution
            h_orig, w_orig = xyz_q.shape[:2]
            sx = self.img_size[1] / w_orig
            sy = self.img_size[0] / h_orig
            K_scaled = self.camera_K.copy()
            K_scaled[0, 0] *= sx
            K_scaled[0, 2] *= sx
            K_scaled[1, 1] *= sy
            K_scaled[1, 2] *= sy
            sample["K"] = torch.from_numpy(K_scaled).float()
        return sample



# -----------------------------------------------------------------------------
# SimCol3D dataset -------------------------------------------------------------
# -----------------------------------------------------------------------------
class SimCol3DDataset(BasePairDataset):
    """SimCol3D implementation reading `pairs_topK.txt` from colon I & II."""

    ROOT = Path("data/processed/SimCol3D")

    def __init__(
        self,
        top_k: int = 50,
        n_samples: int = 256,
        variants: Sequence[str] = ("SyntheticColon_I", "SyntheticColon_II"),
        rng: Optional[random.Random] = None,
    ) -> None:
        pair_paths: List[Tuple[Path, Path]] = []
        for vname in variants:
            # Always read the full pairs_top50.txt generated during preprocessing
            pairs_file = self.ROOT / vname / "pairs_top50.txt"
            if not pairs_file.exists():
                print(f"[SimCol3DDataset] Missing {pairs_file}, skipping.")
                continue
            with pairs_file.open() as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) < 2:
                        continue
                    q_rel = parts[0]
                    db_list = parts[1:top_k + 1] if top_k else parts[1:]  # slice to requested K
                    # Build absolute query path once
                    q_abs = self.ROOT / q_rel
                    for db_rel in db_list:
                        db_abs = self.ROOT / db_rel
                        pair_paths.append((q_abs, db_abs))
        if not pair_paths:
            raise RuntimeError("No pairs found for SimCol3D dataset.")
        # Try load intrinsics from first variant that has cam.txt
        K = None
        for vname in variants:
            cam_path = self.ROOT / vname / "database" / "cam.txt"
            if cam_path.exists():
                vals = [float(v) for v in cam_path.read_text().strip().replace("\n", " ").split()]
                if len(vals) >= 6:
                    fx, fy, cx, cy = vals[0], vals[4], vals[2], vals[5]
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                    break
        super().__init__(pair_paths, n_samples=n_samples, rng=rng, camera_K=K)


# -----------------------------------------------------------------------------
# Helper φ-encoder producing 38-D features (pre-MLP)
# -----------------------------------------------------------------------------
class _Phi38Encoder:
    """统一的φ-编码实现 (NumPy版本)"""

    def __init__(self, f1: float = 31.4159, gamma: float = 2.884031503126606, F: int = 6):
        # 使用统一的频率计算函数
        self.frequencies = compute_phi_frequencies(f1, gamma, F)

    def encode_np(self, xyz: np.ndarray) -> np.ndarray:
        """Return φ-encoding channels first: (38,H,W) given xyz (H,W,3)."""
        h, w, _ = xyz.shape
        
        # 使用统一的UV网格生成
        uv = create_normalized_uv_grid_numpy(h, w)  # (2,H,W)

        # 使用统一的φ编码函数
        xyz_encoded = phi_encode_xyz_numpy(xyz, self.frequencies)  # (H,W,36)
        xyz_encoded = xyz_encoded.transpose(2, 0, 1)  # (36,H,W)

        return np.concatenate([uv, xyz_encoded], axis=0).astype(np.float32)  # (38,H,W)


# -----------------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------------

def simcol3d_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate to handle variable uvxyz list lengths if needed."""
    # In current design N is fixed, but keep flexible
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals) if vals[0].dim() != 2 else torch.stack(vals)  # treat (N,5) also stack
        else:
            out[k] = vals
    return out
