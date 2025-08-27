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


def parse_and_rename(debug: bool = False) -> None:
    """Placeholder for parsing metadata files and renaming frames.

    Expected tasks (to be implemented later):
    - Read trajectory/pose information.
    - Rename images or folders into a unified naming convention.
    - Update metadata accordingly.
    
    Parameters
    ----------
    debug : bool
        If True, only process first 20 objects in each frame folder.
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

                # Limit processing to first 20 objects if debug mode
                process_len = min(20, min_len) if debug else min_len
                
                for idx in range(process_len):
                    pos_vals = pos_lines[idx]
                    quat_vals = quat_lines[idx]
                    pose_line = f"{pos_vals} {quat_vals}\n"
                    out_name = f"{idx:04d}.pose.txt"
                    (frames_dir / out_name).write_text(pose_line)

                # Rename image files (limit to first 20 if debug mode)
                depth_files = sorted(frames_dir.glob("Depth_*.png"))
                rgb_files = sorted(frames_dir.glob("FrameBuffer_*.png"))
                
                if debug:
                    depth_files = depth_files[:20]
                    rgb_files = rgb_files[:20]
                
                for img_path in depth_files:
                    num = img_path.stem.split("_")[-1]
                    new_path = frames_dir / f"{num}.depth.png"
                    if not new_path.exists():
                        img_path.rename(new_path)
                for img_path in rgb_files:
                    num = img_path.stem.split("_")[-1]
                    new_path = frames_dir / f"{num}.rgb.png"
                    if not new_path.exists():
                        img_path.rename(new_path)

                print(f"[parse_and_rename] Processed {frames_dir}")

    # After renaming and pose generation, also produce xyz world npy files
    generate_world_coords(debug=debug)

def _load_intrinsics(cam_txt: Path) -> tuple[float, float, float, float]:
    vals = [float(v) for v in cam_txt.read_text().strip().replace("\n", " ").split()]
    if len(vals) < 6:
        raise ValueError(f"Unexpected cam.txt format: {vals}")
    fx, fy, cx, cy = vals[0], vals[4], vals[2], vals[5]
    return fx, fy, cx, cy


def set_id_grid(depth: torch.Tensor) -> torch.Tensor:
    """Generate pixel coordinate grid."""
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)
    return torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def pixel2cam(depth: torch.Tensor, intrinsics_inv: torch.Tensor) -> torch.Tensor:
    """Convert depth map to camera coordinates using inverse intrinsics."""
    b, h, w = depth.size()
    pixel_coords = set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def _cam_to_world(cam_coords_flat: np.ndarray, rot_mat: np.ndarray, trans_vec: np.ndarray) -> np.ndarray:
    """Transform camera coordinates to world coordinates."""
    return rot_mat @ cam_coords_flat + trans_vec


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()


def _is_left_handed(Rm: np.ndarray) -> bool:
    return np.linalg.det(Rm) < 0


def _convert_left_to_right(T: np.ndarray) -> np.ndarray:
    S = np.diag([1, 1, -1, 1]).astype(T.dtype)
    return S @ T @ S


def generate_world_coords(debug: bool = False) -> None:
    """Generate dense xyz world coordinate .npy for each depth image.
    
    Parameters
    ----------
    debug : bool
        If True, only process first 20 objects in each frame folder.
    """

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
                
                # Limit to first 20 files if debug mode
                if debug:
                    pose_files = pose_files[:20]

                for pose_path in pose_files:
                    idx = pose_path.stem.split(".")[0]  # 0000
                    depth_path = frames_dir / f"{idx}.depth.png"
                    if not depth_path.exists():
                        continue
                    xyz_out = frames_dir / f"{idx}.xyz.npy"
                    if xyz_out.exists():
                        xyz_out.unlink()  # overwrite existing

                    # Read depth image and convert to metres (0–0.20 m range)
                    depth_raw = np.array(Image.open(depth_path))
                    depth_m = depth_raw / 256 / 255 * 0.20  # 0–0.20 m

                    # Convert to torch tensor for processing
                    depth_tensor = torch.tensor(depth_m.reshape((1, *depth_m.shape))).float()
                    
                    # Build intrinsics matrix
                    intrinsics = np.eye(3, dtype=np.float32)
                    intrinsics[0, 0] = fx
                    intrinsics[0, 2] = cx
                    intrinsics[1, 1] = fy
                    intrinsics[1, 2] = cy
                    intrinsics_inv = torch.tensor(np.linalg.inv(intrinsics)).float()
                    
                    # Convert to camera coordinates
                    cam_coords = pixel2cam(depth_tensor, intrinsics_inv)
                    cam_coords_flat = cam_coords.reshape(1, 3, -1).numpy()

                    # Load pose
                    vals = np.fromstring(pose_path.read_text(), sep=" ")
                    if vals.size != 7:
                        print(f"[generate_world_coords] Unexpected pose format: {pose_path}")
                        continue
                    
                    # Pose: translation in cm, rotation quaternion (world→camera)
                    t_cm = vals[:3]
                    rotations = vals[3:]

                    # Convert translation to metres
                    t_m = t_cm * 0.01

                    # Rotation matrix (world→camera)
                    R_wc = _quat_to_rotmat(rotations)

                    # Invert to get camera→world
                    R_cw = R_wc.T
                    t_cw = -R_cw @ t_m.reshape(3, 1)

                    # Apply left-handed→right-handed conversion
                    TM = np.diag([1, -1, 1], k=0).astype(np.float32)
                    R_cw = TM @ R_cw @ TM
                    t_cw = TM @ t_cw

                    rot_gt = R_cw
                    tr_gt = t_cw
                    
                    # Transform to world coordinates
                    cloud_world = _cam_to_world(cam_coords_flat[0], rot_gt, tr_gt)
                    
                    # Reshape back to image format (already in metres)
                    h, w = depth_m.shape
                    xyz_world = cloud_world.T.reshape(h, w, 3)
                    np.save(xyz_out, xyz_world)

                print(f"[generate_world_coords] Finished {frames_dir}")


def generate_embeddings(device: str = "cuda", debug: bool = False) -> None:
    """Generate DreamSim Dino embeddings for each RGB image and save as .embed.npy
    
    Parameters
    ----------
    device : str
        Device to run the model on (cuda or cpu).
    debug : bool
        If True, only process first 20 objects in each frame folder.
    """

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
                
                # Limit to first 20 files if debug mode
                if debug:
                    rgb_files = rgb_files[:20]
                    
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


def create_pairs(top_k: int = 50, debug: bool = False) -> None:
    """Generate top-k retrieval mapping for each SyntheticColon variant.

    For every query RGB image, find the *top_k* most similar database images
    by cosine similarity of DreamSim embeddings and write one mapping file
    per variant: ``pairs_top<k>.txt``.

    Line format::
        <query_rel_path> <db_rel_path_1> ... <db_rel_path_k>

    Paths are relative to ``data/processed/SimCol3D`` so they can be joined
    easily later.
    
    Parameters
    ----------
    top_k : int
        Number of top similar images to retrieve.
    debug : bool
        If True, only process first 20 objects in each frame folder.
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
        embed_files = list(db_root.glob("Frames_*/*.embed.npy"))
        
        # Limit to first 20 files per frame folder if debug mode
        if debug:
            embed_files_by_frame = {}
            for embed_path in embed_files:
                frame_name = embed_path.parent.name
                if frame_name not in embed_files_by_frame:
                    embed_files_by_frame[frame_name] = []
                embed_files_by_frame[frame_name].append(embed_path)
            
            embed_files = []
            for frame_name, files in embed_files_by_frame.items():
                embed_files.extend(sorted(files)[:20])
        
        for embed_path in embed_files:
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
            
            # Limit to first 20 files per frame folder if debug mode
            if debug:
                query_files_by_frame = {}
                for q_embed_path in query_embed_paths:
                    frame_name = q_embed_path.parent.name
                    if frame_name not in query_files_by_frame:
                        query_files_by_frame[frame_name] = []
                    query_files_by_frame[frame_name].append(q_embed_path)
                
                query_embed_paths = []
                for frame_name, files in query_files_by_frame.items():
                    query_embed_paths.extend(sorted(files)[:20])
            
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

def main(steps: List[str] | None = None, debug: bool = False) -> None:
    """Run selected preprocessing steps.

    Parameters
    ----------
    steps : list[str] | None
        Names of steps to run. If *None*, run all in order.
    debug : bool
        If True, only process first 20 objects in each frame folder.
    """

    pipeline = {
        "split": split_dataset,
        "parse_rename": lambda: parse_and_rename(debug=debug),
        "embed": lambda: generate_embeddings(debug=debug),
        "pairs": lambda: create_pairs(debug=debug),
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
    parser.add_argument("steps", nargs="*", help="Steps to execute (split, parse_rename, embed, pairs)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: only process first 20 objects per frame folder")
    args = parser.parse_args()

    main(args.steps or None, debug=args.debug)
