"""
Voxel Cache 생성 스크립트
- train/val scene의 voxel grid를 미리 계산해서 저장
- 이후 generate_dataset.py에서 load해서 사용

Usage:
    python src/utils/create_voxels.py \
    --multiscan-dir data/multiscan_instsegm/object_instance_segmentation \
    --output-dir    data/voxel_cache_010 \
    --voxel-size    0.01
"""

import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# Voxelize
# ─────────────────────────────────────────────

def voxelize(points, voxel_size):
    """
    포인트를 voxel index 배열로 변환
    set 대신 numpy array로 저장 (파일 저장/로드 용이)

    return: [M, 3] int32 array of occupied voxel indices
    """
    indices = np.floor(points / voxel_size).astype(np.int32)
    # 중복 제거
    indices = np.unique(indices, axis=0)
    return indices


def voxels_to_set(voxel_indices):
    """numpy array → set (충돌 검사용)"""
    return set(map(tuple, voxel_indices))


# ─────────────────────────────────────────────
# 씬 로드
# ─────────────────────────────────────────────

def load_scene_xyz(pth_path):
    data  = torch.load(pth_path, map_location="cpu", weights_only=False)
    xyz   = data["xyz"].astype(np.float32)
    valid = data["instance_ids"] != -1
    return xyz[valid]


# ─────────────────────────────────────────────
# 캐시 생성
# ─────────────────────────────────────────────

def precompute_split(scene_files, output_dir, voxel_size):
    os.makedirs(output_dir, exist_ok=True)

    for pth in tqdm(scene_files, desc=f"Voxelizing → {os.path.basename(output_dir)}"):
        scene_name = os.path.splitext(os.path.basename(pth))[0]
        out_path   = os.path.join(output_dir, f"{scene_name}.npy")

        if os.path.exists(out_path):
            continue  # 이미 생성된 경우 skip

        xyz          = load_scene_xyz(pth)
        voxel_indices = voxelize(xyz, voxel_size)
        np.save(out_path, voxel_indices)

    print(f"  저장 완료: {len(glob.glob(os.path.join(output_dir, '*.npy')))} scenes")


# ─────────────────────────────────────────────
# 캐시 로드 (generate_dataset.py에서 사용)
# ─────────────────────────────────────────────

def load_voxel_cache(pth_path, cache_dir, voxel_size):
    """
    캐시가 있으면 로드, 없으면 직접 계산

    return: set of (ix, iy, iz) tuples
    """
    scene_name = os.path.splitext(os.path.basename(pth_path))[0]

    # split 자동 감지 (train/val/test)
    if "train" in pth_path:
        split = "train"
    elif "val" in pth_path:
        split = "val"
    elif "test" in pth_path:
        split = "test"
    else:
        split = "train"

    cache_path = os.path.join(cache_dir, split, f"{scene_name}.npy")

    if os.path.exists(cache_path):
        voxel_indices = np.load(cache_path)
    else:
        print(f"  [cache miss] {scene_name} → 직접 계산")
        xyz           = load_scene_xyz(pth_path)
        voxel_indices = voxelize(xyz, voxel_size)

    return voxels_to_set(voxel_indices)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiscan-dir", type=str,
        default="data/multiscan_instsegm/object_instance_segmentation")
    parser.add_argument("--output-dir", type=str,
        default="data/voxel_cache")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    args = parser.parse_args()

    train_files = sorted(glob.glob(os.path.join(args.multiscan_dir, "train", "*.pth")))
    val_files   = sorted(glob.glob(os.path.join(args.multiscan_dir, "val",   "*.pth")))
    test_files  = sorted(glob.glob(os.path.join(args.multiscan_dir, "test",  "*.pth")))

    print(f"Train scenes: {len(train_files)}")
    print(f"Val scenes:   {len(val_files)}")
    print(f"Test scenes:  {len(test_files)}")
    print(f"Voxel size:   {args.voxel_size}")

    precompute_split(train_files, os.path.join(args.output_dir, "train"), args.voxel_size)
    precompute_split(val_files,   os.path.join(args.output_dir, "val"),   args.voxel_size)
    if test_files:
        precompute_split(test_files, os.path.join(args.output_dir, "test"), args.voxel_size)

    print("\nDone!")