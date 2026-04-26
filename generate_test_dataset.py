"""
generate_test_dataset.py — 고정 test 데이터셋 생성 스크립트

MultiScan test split .pth 씬에 Nubzuki를 삽입하고
TA 명세와 동일한 augmentation을 적용하여 .npy 파일로 저장.

생성 규칙 (TA 명세 동일):
  - 씬당 1~5개 Nubzuki 삽입 (voxel 충돌 방지)
  - 각 인스턴스: anisotropic scale(0.5~1.5/axis) + 3축 랜덤 회전(-180°~+180°)
  - scene diagonal 대비 2.5%~20% base scale
  - color jitter (HSV hue 전체 회전 + brightness ±30%)
  - point drop 없음 (test는 고정 데이터)

저장 포맷 (TA test data 명세 동일):
  dict keys:
    xyz             : float32  (N, 3)
    rgb             : uint8    (N, 3)   0~255
    normal          : float32  (N, 3)
    instance_labels : int32    (N,)     0=bg, 1~=Nubzuki

출력 파일 수: --total-samples 개 (기본 100)
  - n_scenes × 2 를 base로, 나머지는 랜덤으로 선택한 씬에 +1 augmentation
  - 씬당 생성 횟수: 2회 혹은 3회

출력 파일명: {scene_name}_{suffix}.npy  (suffix = a, b, c, ...)
  예) scene_00005_00.pth  →  scene_00005_00_a.npy
                              scene_00005_00_b.npy
                             (scene_00005_00_c.npy)  ← 3회 대상 씬만

Usage (프로젝트 루트에서 실행):
    python generate_test_dataset.py \
        --multiscan-dir data/multiscan_instsegm/object_instance_segmentation \
        --nub-glb       data/asset/sample.glb \
        --voxel-cache-dir data/voxel_cache_010 \
        --output-dir    data/aug_test \
        --total-samples 100 \
        --seed          42
"""

import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

from src.utils.generate_scene import (
    generate_scene,
    load_nubzuki_mesh,
    load_scene,
    load_voxel_cache,
)
from src.utils.augmentation import color_jitter


# ─────────────────────────────────────────────
# 단일 씬 생성
# ─────────────────────────────────────────────

def make_one_scene(pth_path: str,
                   nub_mesh,
                   scene_data,
                   scene_voxels,
                   n_pts_lo: int,
                   n_pts_hi: int,
                   voxel_size: float,
                   max_retries: int = 20):
    """
    한 .pth 씬에서 Nubzuki가 포함된 test 샘플 하나를 생성한다.

    train의 NubzukiDataset.__getitem__ 중 test에 해당하는 부분과 동일:
      generate_scene (scale + rotation augmentation 포함)
      + color_jitter per instance
      (point_drop은 적용하지 않음)

    returns dict{xyz, rgb(uint8), normal, instance_labels} or None
    """
    for _ in range(max_retries):
        scene_data_out = generate_scene(
            scene_pth=pth_path,
            nub_mesh=nub_mesh,
            scene_data=scene_data,
            scene_voxels=scene_voxels,
            n_nubzuki_points_range=(n_pts_lo, n_pts_hi),
            voxel_size=voxel_size,
        )
        if scene_data_out is not None:
            break

    if scene_data_out is None:
        return None

    xyz    = scene_data_out["xyz"]                          # float32 (N,3)
    rgb    = scene_data_out["rgb"].astype(np.float32)       # float32 (N,3) 0~255
    normal = scene_data_out["normal"]                       # float32 (N,3)
    labels = scene_data_out["instance_labels"].astype(np.int32)

    # ── color jitter: fg 인스턴스마다 독립 적용 ──────────────────
    fg_mask = labels > 0
    for inst_id in np.unique(labels[fg_mask]):
        mask        = labels == inst_id
        rgb[mask]   = color_jitter(rgb[mask])               # float32 0~255 반환

    # ── rgb → uint8 저장 ─────────────────────────────────────────
    rgb_u8 = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    return {
        "xyz":             xyz,
        "rgb":             rgb_u8,
        "normal":          normal,
        "instance_labels": labels,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MultiScan test split → Nubzuki 삽입 + augmentation → .npy 저장"
    )
    parser.add_argument("--multiscan-dir",   type=str,
                        default="data/multiscan_instsegm/object_instance_segmentation")
    parser.add_argument("--nub-glb",         type=str,
                        default="data/asset/sample.glb")
    parser.add_argument("--voxel-cache-dir", type=str,
                        default="data/voxel_cache_010")
    parser.add_argument("--output-dir",      type=str,
                        default="data/aug_test")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--n-pts-lo",        type=int, default=8192)
    parser.add_argument("--n-pts-hi",        type=int, default=24576)
    parser.add_argument("--voxel-size",      type=float, default=0.01)
    parser.add_argument("--total-samples",   type=int, default=100,
                        help="생성할 총 파일 수 (기본 100)")
    args = parser.parse_args()

    test_pths = sorted(glob.glob(
        os.path.join(args.multiscan_dir, "test", "*.pth")
    ))
    if not test_pths:
        raise FileNotFoundError(
            f"test .pth 파일이 없습니다: {os.path.join(args.multiscan_dir, 'test')}"
        )

    n_scenes = len(test_pths)
    target   = args.total_samples

    if target < n_scenes:
        raise ValueError(
            f"--total-samples({target})가 씬 수({n_scenes})보다 작습니다. "
            "최소 씬 수 이상으로 설정해 주세요."
        )

    # ── 씬별 augmentation 횟수 결정 ─────────────────────────────
    # 모든 씬에 base_count(=2)회, 나머지 remainder개 씬에 +1회
    base_count = target // n_scenes          # 기본 횟수 (2)
    remainder  = target  % n_scenes          # 추가 1회가 필요한 씬 수 (18)

    rng       = np.random.default_rng(args.seed)
    extra_idx = set(rng.choice(n_scenes, size=remainder, replace=False).tolist())
    aug_counts = [base_count + (1 if i in extra_idx else 0) for i in range(n_scenes)]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Test scenes     : {n_scenes}")
    print(f"Total target    : {target}")
    print(f"Base aug/scene  : {base_count}")
    print(f"Scenes with +1  : {remainder}  ({base_count+1}회)")
    print(f"Output dir      : {args.output_dir}")
    print(f"Seed            : {args.seed}")
    print(f"n_pts range     : [{args.n_pts_lo}, {args.n_pts_hi}]")

    # ── Nubzuki mesh 한 번만 로드 ───────────────────────────────
    print("Nubzuki mesh 로드 중...")
    nub_mesh = load_nubzuki_mesh(args.nub_glb)

    saved, skipped = 0, 0
    suffixes = [chr(ord('a') + k) for k in range(26)]   # a, b, c, ...

    for i, pth in enumerate(tqdm(test_pths, desc="Generating test data")):
        scene_name = os.path.splitext(os.path.basename(pth))[0]
        n_aug      = aug_counts[i]

        # scene data / voxel cache는 씬당 한 번만 로드
        scene_data   = load_scene(pth)
        scene_voxels = load_voxel_cache(pth, args.voxel_cache_dir, args.voxel_size)

        for j in range(n_aug):
            suffix   = suffixes[j]                       # 'a', 'b', 'c'
            out_path = os.path.join(args.output_dir, f"{scene_name}_{suffix}.npy")

            if os.path.exists(out_path):
                saved += 1
                continue

            # (씬 인덱스, aug 인덱스) 조합으로 고정 시드 → 재현 가능
            np.random.seed(args.seed + i * 10 + j)

            sample = make_one_scene(
                pth_path=pth,
                nub_mesh=nub_mesh,
                scene_data=scene_data,
                scene_voxels=scene_voxels,
                n_pts_lo=args.n_pts_lo,
                n_pts_hi=args.n_pts_hi,
                voxel_size=args.voxel_size,
            )

            if sample is None:
                print(f"  [skip] Nubzuki 배치 실패: {scene_name}_{suffix}")
                skipped += 1
                continue

            np.save(out_path, sample)
            saved += 1

    print(f"\n완료: {saved}개 저장, {skipped}개 스킵")
    print(f"저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()
