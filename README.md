# Project1 — Human Pose Detection using MMVR Dataset

Human pose estimation from radar heatmaps (MMVR dataset), with an RGB pose-detection demo via Streamlit.

## Project layout

```
.
├── notebook.ipynb                 # main training + analysis notebook (P1S1 & P1S2)
├── notebook improved fusion and baseline.ipynb
├── app.py                         # Streamlit app — RGB pose detection on uploaded images
├── requirements.txt               # unified dependencies for notebook + app
├── P1/                            # MMVR dataset (extracted from P1.zip)
├── full_best_baseline_p1s1.pth    # saved model checkpoints
├── full_best_fusion_p1s1.pth
├── full_best_baseline_p1s2.pth
├── full_best_fusion_p1s2.pth
└── IMG_1.jpeg ... IMG_5.jpeg      # demo RGB images
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Streamlit app

From the project root, with the venv activated:

```bash
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501).

### App features

- Upload one or more images (JPG / JPEG / PNG, including iPhone MPO)
- Adjustable detection confidence and MediaPipe model complexity
- Side-by-side original vs annotated view
- Downloadable annotated PNG
- Landmark coordinates table

This mirrors the RGB pose-detection logic from **Shell 1.2 / 2.2** of `notebook.ipynb`.

## Run the training notebook

```bash
jupyter notebook notebook.ipynb
```

## How `notebook.ipynb` works

The notebook predicts 17 human-body keypoints (COCO format) from **mmWave radar heatmaps**, and compares a single-view baseline against a two-view fusion model. It also includes an RGB sanity check using MediaPipe. The structure is organized into two parallel experiments (one per dataset split) plus an RGB demo.

### Notebook sections

| Section | Purpose |
|---|---|
| **1.0** Fusion + Baseline on **P1S1** | Trains and evaluates both models on the P1S1 split |
| **1.1** Radar inference + analysis | Reloads saved P1S1 checkpoints, compares GT vs Baseline vs Fusion, plots error histogram |
| **1.2** RGB pose detection (P1S1) | Runs MediaPipe Pose on demo JPEG images (`IMG_*.jpeg`) |
| **2.0** Fusion + Baseline on **P1S2** | Same pipeline on the harder P1S2 split |
| **2.1** Radar inference + analysis | Reloads P1S2 checkpoints, error analysis |
| **2.2** RGB pose detection (P1S2) | MediaPipe Pose again on the demo images |

### Pipeline walkthrough

**1. Configuration**
- `DATASET_PATH="P1"`, `SPLIT_NAME` is either `"P1S1"` or `"P1S2"`
- `EPOCHS=15`, `BATCH_SIZE=32`, `IMG_W=640, IMG_H=480`
- Auto-detects device: `mps` (Apple Silicon) → `cuda` (NVIDIA) → `cpu`
- Separate LR for each model: `LR_BASELINE=1e-3`, `LR_FUSION=1e-4`
- Reproducibility via fixed `SEED=42` for `random`, `numpy`, and `torch`

**2. Dataset: `RadarPoseDataset`**

Walks the dataset folder and collects tuples of `(radar, pose, bbox, meta)` `.npz` files per frame, filtered by segment names loaded from `P1/data_split.npz`.

For each sample it loads:
- `hm_hori`, `hm_vert` — 2D radar heatmaps (horizontal and vertical views), each z-score normalized
- `kp` — ground-truth keypoints (17 × 3: x, y, visibility), coordinates normalized to [0, 1] by image size
- `valid` — boolean mask per keypoint (visibility > 0 AND not at origin)
- `bbox_i`, `bbox_hori` — bounding boxes for OKS area and a simple depth target

Output tensor shapes:
- `radar`: `(1, H, W)` for baseline (horizontal only), `(2, H, W)` for fusion (both views)
- `keypoints`: `(17, 2)` normalized
- `valid`: `(17,)` bool
- `depth_target`: scalar (midpoint of `bbox_hori[0]` x-range)

**3. Models**

**`BaselinePoseDepthModel`** (single-branch CNN, ~3M params)
```
Input (1, H, W) → 4× [Conv3x3 → BN → ReLU → MaxPool2]
              → AdaptiveAvgPool(4×4) → Flatten → FC(1024) → FC(512)
              → pose_head (34 → reshape 17×2 → sigmoid)
              └→ depth_head (1 → clamp [0, 8])
```

**`FusionTwoBranchModel`** (two parallel branches, ~1.5M params)
```
horizontal view (1, H, W) → branch_h ──┐
                                        ├─ concat → FC(512) → FC(256)
vertical view   (1, H, W) → branch_v ──┘
                                        → pose_head (17×2, sigmoid)
                                        └→ optional depth_head
```

Each branch has 3 conv blocks + AdaptiveAvgPool(4×4). Fusion is feature-level concatenation.

**4. Losses and metrics**

- **Pose loss**: masked MSE — `((pred - gt)² * valid_mask).sum() / valid_mask.sum()`
- **Depth loss**: plain MSE with a small weight (0.01–0.1), to keep it from dominating training
- **PCK@0.1** — percentage of keypoints with normalized L2 error < 0.1
- **OKS** — Object Keypoint Similarity with COCO sigmas, computed per-instance then averaged
- **Depth MAE** — mean absolute error in pixel units of `bbox_hori` midpoint

**5. Training loop (`train_and_evaluate`)**

For each of `BASELINE` and `FUSION`:
1. Load split via `load_split(DATASET_PATH, SPLIT_NAME)` from `data_split.npz`
2. Build train/val/test `DataLoader`s
3. Adam optimizer, 15 epochs
4. Track train + val metrics per epoch; save best model by val PCK to `full_best_{model}_{split}.pth`
5. Reload best checkpoint, run on test set, dump metrics as JSON to `full_history_*.json`

**6. Visualization**

- Loss / PCK / OKS curves per model
- Side-by-side panels: horizontal radar, vertical radar, and GT-vs-predicted skeleton on a black canvas
- Fusion keypoint error histogram (pixel error distribution)

**7. Saved outputs**

After running, you'll see these files in the root:
- `full_best_baseline_{p1s1,p1s2}.pth`, `full_best_fusion_{p1s1,p1s2}.pth` — model weights
- `full_history_baseline_{p1s1,p1s2}.json`, `full_history_fusion_{p1s1,p1s2}.json` — per-epoch metrics

### Reported results (from notebook outputs)

**P1S1 split** (86,579 / 10,538 / 10,785 train/val/test samples)

| Model | Test PCK | Test OKS | Depth MAE |
|---|---|---|---|
| Baseline | **95.61%** | 0.389 | 22.06 |
| Fusion   | **~95.4%+** (trending up through training) | ~0.37 | — (depth disabled) |

**P1S2 split** (70,266 / 24,398 / 13,238 samples — different segments, harder generalization)

| Model | Test PCK | Test OKS | Depth MAE |
|---|---|---|---|
| Baseline | **86.56%** | 0.235 | 32.36 |
| Fusion   | **~93%+** (val PCK 93.33% after epoch 1) | — | — |

Key findings:
1. **Fusion > Baseline** on both splits — combining horizontal + vertical radar views helps.
2. **P1S2 is harder than P1S1** — the baseline drops ~9 PCK points due to distribution shift across scenes.
3. **Depth estimation is weak** (MAE 22–32 px) — that's why fusion disables depth by default.

### RGB sanity check (Shell 1.2 / 2.2)

Runs **MediaPipe Pose v0.10.21** on `IMG_*.jpeg` to visualize what "good" pose detection looks like given a real RGB camera — as a reference point for the radar-only models. iPhone MPO format is handled automatically via `PIL.ImageSequence`. This is the same logic that powers `app.py`.

## Notes

- `mediapipe==0.10.21` requires `numpy<2.0` — the `requirements.txt` enforces this.
- Developed and tested on Python 3.11, Apple Silicon (MPS backend for PyTorch).
- The full P1 dataset (`P1.zip`) is ~24 GB extracted; not included in the repo.
