# 1. 바로 사용하기 (Google Drive Download)
해당 [link](https://drive.google.com/drive/folders/1xqCtIY5Y7M-lJFSdU51EBkqNIVlxLfC0?usp=drive_link)에서 미리 생성된 test set 100개를 download 할 수 있음 **(seed=42로 생성함)**

다운받은 zip 파일을 data 디렉토리 내부에서 unzip

### 저장 위치

```
data/
├── aug_test/
│   └── scene_XXXXX_XX_a.npy
│   └── scene_XXXXX_XX_b.npy
│   └── ...
```


```bash
# 1. 평가 (TA 제공 코드 그대로 사용 가능)
python evaluate.py \
    --test-data-dir data/aug_test \
    --ckpt-path     runs/exp01/best.pth \
    --output-dir    runs/exp01/eval_aug_test
```

만약 직접 생성하고 싶다면 공유된 create_voxels.py와 generate_test_dataset.py를 통해 생성 가능

아래 가이드 참고 
<br>
<br>
# 2. Test Dataset 생성 가이드

고정된 test set을 만드는 과정에 필요한 두 스크립트의 역할 및 사용법을 정리함

참고: 약 15분 소요

---

## 전체 흐름

```
MultiScan test .pth scene
        │
        ▼
[1단계] create_voxels.py      ← test split voxel cache 생성하도록 수정 (코드 수정)
        │  data/voxel_cache_010/test/*.npy (추가로 생성)
        ▼
[2단계] generate_test_dataset.py  ← Nubzuki 삽입 + augmentation → .npy 저장
        │  data/aug_test/*.npy  (Default 100개)
        ▼
[평가]  evaluate.py
```

---

## 1. `src/utils/create_voxels.py`

### 역할

MultiScan 씬 파일(`.pth`)에서 **점유된 voxel 좌표를 미리 계산해 `.npy`로 캐싱**

`generate_test_dataset.py`가 Nubzuki 배치 시 충돌 검사에 이 캐시를 사용하므로,  
test 데이터 생성 전에 반드시 먼저 실행

### 기존 대비 변경 사항

| 항목 | 기존 | 변경 후 |
|------|------|---------|
| 처리 split | `train`, `val` | `train`, `val`, **`test` 추가** |
| `load_voxel_cache()` | `train` / `val` 감지만 | `test` 경로도 감지 (`elif "test" in pth_path`) |
| `__main__` | test 파일 수집·처리 없음 | `test_files` 수집 및 `precompute_split()` 호출 추가 |

> test 씬이 없을 경우 자동으로 skip (`if test_files:` 조건)

### 사용법

먼저 프로젝트 폴더 내에서 src/utils/create_voxels.py 파일과 교체

test split에 대한 voxel 캐싱 추가 이외 부분은 모두 동일하므로 train과 val data에 대한 영향은 없음

```bash
python src/utils/create_voxels.py \
    --multiscan-dir data/multiscan_instsegm/object_instance_segmentation \
    --output-dir    data/voxel_cache_010 \
    --voxel-size    0.01
```

### 출력 구조

```
data/voxel_cache_010/
├── train/
│   └── scene_XXXXX_XX.npy
├── val/
│   └── scene_XXXXX_XX.npy
└── test/                     ← 이번에 추가된 부분
    └── scene_XXXXX_XX.npy
```

---

## 2. `generate_test_dataset.py`

### 역할

MultiScan **test** split `.pth` 씬에 Nubzuki를 삽입하고,  
train dataloader와 동일한 augmentation을 적용한 뒤 고정된 `.npy` test 파일로 저장

동일한 seed로 실행하면 항상 동일한 데이터가 생성되므로, 팀 간 평가 일관성이 유지

Drive에 공유된 데이터는 **seed=42**

### Augmentation 규칙 (TA 명세 동일)

| 항목 | 내용 |
|------|------|
| Nubzuki 삽입 수 | 씬당 1~5개 (랜덤) |
| Voxel 충돌 방지 | voxel grid + 1-voxel dilate |
| Scale | scene diagonal 대비 2.5%~20% base scale |
| Anisotropic scaling | 각 축(x, y, z) 독립적으로 `[0.5, 1.5]` |
| Rotation | x, y, z축 각각 `[-180°, 180°]` |
| Color jitter | HSV hue 전체 회전 + brightness `[0.7, 1.3]` (인스턴스별 독립) |
| Point drop | **적용 안 함** (고정 test set) |

### 파일 수 구성 방식

- 총 생성 수: `--total-samples` (기본 **100개**)
- 씬당 기본 **2회** augmentation 생성
- `total_samples - n_scenes * 2` 개 씬은 추가로 **1회** 더 생성 (랜덤 선택, seed 고정)
- 예) test 씬 41개, 목표 100개 → 23개 씬 × 2회 + 18개 씬 × 3회 = 100개

### 파일명 규칙

```
{원본 scene 이름}_{suffix}.npy    (suffix: a, b, c, ...)
```

예시:
```
scene_00005_00_a.npy   ← 1번째 augmentation
scene_00005_00_b.npy   ← 2번째 augmentation
scene_00005_00_c.npy   ← 3번째 augmentation (해당 씬만)
```

### 저장 포맷

TA test data 명세와 동일한 numpy dict 형식:

| key | dtype | shape | 설명 |
|-----|-------|-------|------|
| `xyz` | float32 | (N, 3) | 포인트 좌표 (정규화 전) |
| `rgb` | uint8 | (N, 3) | RGB 색상 (0~255) |
| `normal` | float32 | (N, 3) | 법선 벡터 |
| `instance_labels` | int32 | (N,) | 0=배경, 1~=Nubzuki |

### 사용법

**[필수] create_voxels.py를 먼저 실행해 test voxel cache를 생성한 뒤 진행**

프로젝트 root directory에서 generate_test_dataset.py 실행

```bash
# 기본 실행 (100개 생성)
python generate_test_dataset.py \
    --multiscan-dir   data/multiscan_instsegm/object_instance_segmentation \
    --nub-glb         data/asset/sample.glb \
    --voxel-cache-dir data/voxel_cache_010 \
    --output-dir      data/aug_test \
    --seed            42
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--multiscan-dir` | `data/multiscan_instsegm/...` | MultiScan 데이터 루트 |
| `--nub-glb` | `data/asset/sample.glb` | Nubzuki mesh 파일 |
| `--voxel-cache-dir` | `data/voxel_cache_010` | voxel cache 디렉토리 |
| `--output-dir` | `data/aug_test` | 생성된 test .npy 저장 위치 |
| `--total-samples` | `100` | 생성할 총 파일 수 |
| `--seed` | `42` | 재현성을 위한 랜덤 시드 |
| `--n-pts-lo` | `8192` | Nubzuki 포인트 수 최솟값 |
| `--n-pts-hi` | `24576` | Nubzuki 포인트 수 최댓값 |
| `--voxel-size` | `0.01` | create_voxels.py와 동일하게 설정 |

> `--voxel-size`는 반드시 `create_voxels.py` 실행 시 사용한 값과 동일해야 함

---

## 빠른 실행 순서 요약

```bash
# 1. test voxel cache 생성 (기존에 train/val만 있었다면 재실행)
python src/utils/create_voxels.py \
    --multiscan-dir data/multiscan_instsegm/object_instance_segmentation \
    --output-dir    data/voxel_cache_010 \
    --voxel-size    0.01

# 2. test dataset 생성
python generate_test_dataset.py \
    --multiscan-dir   data/multiscan_instsegm/object_instance_segmentation \
    --nub-glb         data/asset/sample.glb \
    --voxel-cache-dir data/voxel_cache_010 \
    --output-dir      data/aug_test \
    --seed            42

# 3. 평가 (TA 제공 코드 그대로 사용 가능)
python evaluate.py \
    --test-data-dir data/aug_test \
    --ckpt-path     runs/exp01/best.pth \
    --output-dir    runs/exp01/eval_aug_test
```
