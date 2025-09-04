"""preprocess_SimCol3D.py
Dataset initialization pipeline for SimCol3D.

This script contains four main steps:
1. split_dataset: Split raw data into database and query sets according to text lists.
2. parse_and_rename: Standardize filenames / paths if needed.
3. generate_world_coords: Generate dense xyz world coordinate .npy for each depth image.
4. generate_embeddings: Produce feature embeddings for each frame (placeholder).
5. create_pairs: Build positive / negative image pairs for training (placeholder).

Implementation will be filled in later. Currently each step provides
only a minimal placeholder so that the pipeline can be called without error.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import numpy as np
import imageio.v2 as imageio
from dreamsim import dreamsim
from PIL import Image
from tqdm import tqdm
import torch
from scipy.spatial.transform import Rotation as R

# Base directories
RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "SimCol3D"
PROCESSED_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed" / "SimCol3D"
MISC_ROOT = RAW_ROOT / "misc"


def split_dataset() -> None:
    """Split raw SyntheticColon_* sequences into database/query subsets.

    The text files ``database_file_*.txt`` and ``query_file_*.txt`` list the
    target *frame folders* that should belong to each subset.

    For now this function simply copies the corresponding ``Frames_*`` folders
    and related *txt* metadata files (``SavedPosition_*``,
    ``SavedRotationQuaternion_*`` and ``cam.txt``) from ``data/raw`` into the
    new ``data/processed`` tree, preserving the original names.

    TODO: replace naive copy with a smarter link/processing strategy.
    """

    import itertools

    # Map variant letter to its list file suffix
    variants = {
        "I": "SyntheticColon_I",
        "II": "SyntheticColon_II",
        "III": "SyntheticColon_III",
    }

    for suffix, colon_name in variants.items():
        db_list_file = MISC_ROOT / f"database_file_{suffix}.txt"
        q_list_file = MISC_ROOT / f"query_file_{suffix}.txt"

        if not db_list_file.exists() or not q_list_file.exists():
            print(f"[split_dataset] Warning: list files missing for {colon_name}, skipped.")
            continue

        with db_list_file.open() as f:
            db_lines = [ln.strip().rstrip("/") for ln in f if ln.strip()]
        with q_list_file.open() as f:
            q_lines = [ln.strip().rstrip("/") for ln in f if ln.strip()]

        assignments = ((db_lines, "database"), (q_lines, "query"))

        for lines, subset in assignments:
            subset_root = PROCESSED_ROOT / colon_name / subset
            subset_root.mkdir(parents=True, exist_ok=True)

            # Copy cam.txt once per subset
            cam_src = RAW_ROOT / colon_name / "cam.txt"
            cam_dst = subset_root / "cam.txt"
            if cam_src.exists() and not cam_dst.exists():
                shutil.copy2(cam_src, cam_dst)

            for rel_path in lines:
                # rel_path example: SyntheticColon_I/Frames_S1
                parts = Path(rel_path).parts
                if len(parts) < 2:
                    continue
                frame_folder = parts[1]  # Frames_S1
                seq_id = frame_folder.split("_")[-1]  # S1

                src_frames = RAW_ROOT / colon_name / frame_folder
                dst_frames = subset_root / frame_folder

                if not src_frames.exists():
                    print(f"[split_dataset] Missing source {src_frames}, skipping.")
                    continue

                # Copy frames folder (overwrite if exists)
                if dst_frames.exists():
                    shutil.rmtree(dst_frames)
                shutil.copytree(src_frames, dst_frames)

                # Copy metadata files for this sequence
                meta_files = [
                    f"SavedPosition_{seq_id}.txt",
                    f"SavedRotationQuaternion_{seq_id}.txt",
                ]
                for mf in meta_files:
                    src_meta = RAW_ROOT / colon_name / mf
                    if src_meta.exists():
                        shutil.copy2(src_meta, subset_root / mf)
                    else:
                        print(f"[split_dataset] Missing metadata {src_meta}")


def parse_and_rename() -> None:
    """Placeholder for parsing metadata files and renaming frames.

    Expected tasks (to be implemented later):
    - Read trajectory/pose information.
    - Rename images or folders into a unified naming convention.
    - Update metadata accordingly.
    """

    # Iterate over processed colon variants and subsets
    variants = ["SyntheticColon_I", "SyntheticColon_II", "SyntheticColon_III"]
    subsets = ["database", "query"]

    for colon_name in variants:
        for subset in subsets:
            subset_root = PROCESSED_ROOT / colon_name / subset
            if not subset_root.exists():
                continue

            # Find all Frames_* folders within this subset
            for frames_dir in subset_root.glob("Frames_*"):
                if not frames_dir.is_dir():
                    continue

                seq_id = frames_dir.name.split("_")[-1]  # e.g. S1 / B3 / O2

                pos_file = subset_root / f"SavedPosition_{seq_id}.txt"
                quat_file = subset_root / f"SavedRotationQuaternion_{seq_id}.txt"

                if not pos_file.exists() or not quat_file.exists():
                    print(f"[parse_and_rename] Missing pose files for {frames_dir}, skipping.")
                    continue

                with pos_file.open() as fp:
                    pos_lines = [ln.strip() for ln in fp if ln.strip()]
                with quat_file.open() as fq:
                    quat_lines = [ln.strip() for ln in fq if ln.strip()]

                if len(pos_lines) != len(quat_lines):
                    print(f"[parse_and_rename] Mismatch lines in position/quat for {seq_id} (pos={len(pos_lines)}, quat={len(quat_lines)})")
                    min_len = min(len(pos_lines), len(quat_lines))
                else:
                    min_len = len(pos_lines)

                for idx in range(min_len):
                    pos_vals = pos_lines[idx]
                    quat_vals = quat_lines[idx]
                    pose_line = f"{pos_vals} {quat_vals}\n"
                    out_name = f"{idx:04d}.pose.txt"
                    (frames_dir / out_name).write_text(pose_line)

                # Rename image files
                for img_path in frames_dir.glob("Depth_*.png"):
                    num = img_path.stem.split("_")[-1]
                    new_path = frames_dir / f"{num}.depth.png"
                    if not new_path.exists():
                        img_path.rename(new_path)
                for img_path in frames_dir.glob("FrameBuffer_*.png"):
                    num = img_path.stem.split("_")[-1]
                    new_path = frames_dir / f"{num}.rgb.png"
                    if not new_path.exists():
                        img_path.rename(new_path)

                print(f"[parse_and_rename] Processed {frames_dir}")

    # After renaming and pose generation, also produce xyz world npy files
    generate_world_coords()

def _load_intrinsics(cam_txt: Path) -> tuple[float, float, float, float]:
    vals = [float(v) for v in cam_txt.read_text().strip().replace("\n", " ").split()]
    if len(vals) < 6:
        raise ValueError(f"Unexpected cam.txt format: {vals}")
    fx, fy, cx, cy = vals[0], vals[4], vals[2], vals[5]
    return fx, fy, cx, cy


def _depth_to_cam(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    zs = depth.astype(np.float32)
    xs_cam = (xs - cx) / fx * zs
    ys_cam = (ys - cy) / fy * zs
    xyz = np.stack([xs_cam, ys_cam, zs], axis=2)
    xyz[zs <= 0] = np.nan
    return xyz


def _cam_to_world(xyz_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    orig_shape = xyz_cam.shape
    pts = xyz_cam.reshape(-1, 3)
    Rm = T[:3, :3]
    t = T[:3, 3]
    pts_w = pts @ Rm.T + t
    return pts_w.reshape(orig_shape)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()


def _is_left_handed(Rm: np.ndarray) -> bool:
    return np.linalg.det(Rm) < 0


def _convert_left_to_right(T: np.ndarray) -> np.ndarray:
    S = np.diag([1, 1, -1, 1]).astype(T.dtype)
    return S @ T @ S


def generate_world_coords() -> None:
    """Generate dense xyz world coordinate .npy for each depth image."""

    variants = ["SyntheticColon_I", "SyntheticColon_II", "SyntheticColon_III"]
    subsets = ["database", "query"]

    for colon_name in variants:
        for subset in subsets:
            subset_root = PROCESSED_ROOT / colon_name / subset
            if not subset_root.exists():
                continue
            cam_file = subset_root / "cam.txt"
            if not cam_file.exists():
                print(f"[generate_world_coords] Missing {cam_file}")
                continue
            fx, fy, cx, cy = _load_intrinsics(cam_file)

            for frames_dir in subset_root.glob("Frames_*"):
                pose_files = sorted(frames_dir.glob("*.pose.txt"))
                if not pose_files:
                    continue

                for pose_path in pose_files:
                    idx = pose_path.stem.split(".")[0]  # 0000
                    depth_path = frames_dir / f"{idx}.depth.png"
                    if not depth_path.exists():
                        continue
                    xyz_out = frames_dir / f"{idx}.xyz.npy"
                    if xyz_out.exists():
                        xyz_out.unlink()  # overwrite existing

                    # Read depth image and convert to metres (0–0.20 m range).
                    depth_raw = imageio.imread(depth_path)
                    if depth_raw.dtype == np.uint16:
                        depth_norm = depth_raw.astype(np.float32) / 65535.0  # 0–1
                    else:  # fallback for 8-bit legacy images
                        depth_norm = depth_raw.astype(np.float32) / 255.0
                    depth_m = depth_norm * 0.20

                    xyz_cam = _depth_to_cam(depth_m, fx, fy, cx, cy)

                    # Load pose
                    vals = np.fromstring(pose_path.read_text(), sep=" ")
                    if vals.size != 7:
                        print(f"[generate_world_coords] Unexpected pose format: {pose_path}")
                        continue
                    # SavedPosition translations are in centimetres → convert to metres.
                    t_vec = vals[:3] * 0.01
                    q = vals[3:]
                    Rm = _quat_to_rotmat(q)
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = Rm
                    T[:3, 3] = t_vec
                    if _is_left_handed(Rm):
                        T = _convert_left_to_right(T)

                    xyz_world = _cam_to_world(xyz_cam, T)
                    np.save(xyz_out, xyz_world)

                print(f"[generate_world_coords] Finished {frames_dir}")


def generate_embeddings(device: str = "cuda") -> None:
    """Generate DreamSim Dino embeddings for each RGB image and save as .embed.npy"""

    model, preprocess = dreamsim(pretrained=True, dreamsim_type="dino_vitb16")
    model.to(device)
    model.eval()

    variants = ["SyntheticColon_I", "SyntheticColon_II", "SyntheticColon_III"]
    subsets = ["database", "query"]

    for colon_name in variants:
        for subset in subsets:
            subset_root = PROCESSED_ROOT / colon_name / subset
            if not subset_root.exists():
                continue
            for frames_dir in subset_root.glob("Frames_*"):
                rgb_files = sorted(frames_dir.glob("*.rgb.png"))
                if not rgb_files:
                    continue
                for rgb_path in rgb_files:
                    # Remove the trailing '.rgb' from the stem for cleaner naming
                    name_no_rgb = rgb_path.stem.replace(".rgb", "")
                    embed_out = rgb_path.with_name(f"{name_no_rgb}.embed.npy")
                    if embed_out.exists():
                        continue  # skip existing
                    img = Image.open(rgb_path).convert("RGB")
                    inp = preprocess(img).to(device)  # preprocess already adds batch dim
                    with torch.no_grad():
                        emb = model.embed(inp).cpu().squeeze(0).numpy()  # (768,)
                    np.save(embed_out, emb)
                print(f"[generate_embeddings] Finished {frames_dir}")


def create_pairs(top_k: int = 50) -> None:
    """Generate top-k retrieval mapping for each SyntheticColon variant.

    For every query RGB image, find the *top_k* most similar database images
    by cosine similarity of DreamSim embeddings and write one mapping file
    per variant: ``pairs_top<k>.txt``.

    Line format::
        <query_rel_path> <db_rel_path_1> ... <db_rel_path_k>

    Paths are relative to ``data/processed/SimCol3D`` so they can be joined
    easily later.
    """

    variants = [
        "SyntheticColon_I",
        "SyntheticColon_II",
        "SyntheticColon_III",
    ]
    processed_root = PROCESSED_ROOT

    for colon_name in variants:
        q_root = processed_root / colon_name / "query"
        db_root = processed_root / colon_name / "database"
        if not q_root.exists() or not db_root.exists():
            continue

        # Load database embeddings once into memory
        db_embs: list[np.ndarray] = []
        db_paths: list[Path] = []
        for embed_path in db_root.glob("Frames_*/*.embed.npy"):
            emb = np.load(embed_path).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            db_embs.append(emb / norm)
            # Convert *.embed.npy back to the original RGB image path
            db_paths.append(Path(str(embed_path).replace(".embed.npy", ".rgb.png")).relative_to(processed_root))

        if not db_embs:
            print(f"[create_pairs] No database embeddings for {colon_name}")
            continue

        db_matrix = np.stack(db_embs, axis=0)  # (N_db, 768)

        out_path = processed_root / colon_name / f"pairs_top{top_k}.txt"
        with out_path.open("w") as fout:
            query_embed_paths = list(q_root.glob("Frames_*/*.embed.npy"))
            for q_embed_path in tqdm(query_embed_paths, desc=f"{colon_name} queries"):
                q_emb = np.load(q_embed_path).astype(np.float32)
                q_norm = np.linalg.norm(q_emb)
                if q_norm == 0:
                    continue
                q_emb /= q_norm
                sims = db_matrix @ q_emb  # cosine similarity
                top_idx = np.argsort(-sims)[:top_k]
                top_rel_paths = [str(db_paths[i]) for i in top_idx]
                q_rel_path = str(Path(str(q_embed_path).replace(".embed.npy", ".rgb.png")).relative_to(processed_root))
                fout.write(" ".join([q_rel_path, *top_rel_paths]) + "\n")

        print(f"[create_pairs] Wrote {out_path}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def main(steps: List[str] | None = None) -> None:
    """Run selected preprocessing steps.

    Parameters
    ----------
    steps : list[str] | None
        Names of steps to run. If *None*, run all in order.
    """

    pipeline = {
        "split": split_dataset,
        "parse_rename": parse_and_rename,
        "embed": generate_embeddings,
        "pairs": create_pairs,
    }

    if steps is None:
        steps = list(pipeline.keys())

    for step in steps:
        func = pipeline.get(step)
        if func is None:
            print(f"[main] Unknown step: {step}")
            continue
        print(f"[main] Running step: {step}")
        func()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SimCol3D preprocessing pipeline")
    parser.add_argument("steps", nargs="*", help="Steps to execute (split, parse_rename, pairs)")
    args = parser.parse_args()

    main(args.steps or None)
