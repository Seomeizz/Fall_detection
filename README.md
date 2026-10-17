# NTU RGB+D 60 — 8-Class Skeleton Action Recognition with Temporal Transformer

A research prototype that classifies one complete tracked person's motion clip into **eight mapped action groups** using 2D COCO-17 keypoints and a Temporal Transformer. The laptop/video demo uses YOLO Pose + BoT-SORT for pose extraction and tracking, then runs the trained Transformer and writes an annotated MP4 (no CSV).

> **Scope:** this is 8-class action recognition with a `falling` class, not a dedicated binary fall-vs-normal detector and not a certified safety system. The alert overlay also uses a heuristic temporal/geometric guard; its thresholds must be validated on labelled videos.

## Pipeline

```text
Laptop camera or MP4
  -> YOLO11n-Pose (COCO-17 keypoints)
  -> BoT-SORT track IDs
  -> rolling per-track clip
  -> uniform sampling to 30 frames
  -> confidence interpolation + clip-level normalization
  -> 51 features/frame (34 normalized XY + 17 confidence)
  -> 4-layer Temporal Transformer
  -> action class + probability
  -> optional fall-event heuristic
  -> annotated MP4
```

## Eight mapped classes

The notebook maps selected native NTU action IDs into the following groups:

| Class | NTU action IDs |
|---|---|
| `walking` | A059, A060 |
| `standing` | A001, A002, A003, A004, A010, A014, A015, A018, A019, A020, A021, A022, A023, A028, A031, A032, A033, A035, A036, A038, A039, A040, A041, A049, A056, A058 |
| `sitting` | A008, A011, A012, A029, A030 |
| `bending` | A005, A006, A013, A025, A037 |
| `falling` | A043 |
| `getting_up` | A009 |
| `stumbling` | A042 |
| `abnormal_movement` | A044–A048 |

These are **project-defined grouped labels**, not the original standard 60-class labels. In particular, the current `standing` group includes activities whose semantics are not strictly “standing”. Keep this caveat in reports, and revise the mapping if you want semantically purer groups.

## Transformer architecture

- Input per clip: `[30, 51]`
- Input projection: `51 -> 128`, LayerNorm, ReLU, Dropout(0.15)
- Sinusoidal positional encoding
- 4 Transformer Encoder layers, 4 attention heads, FFN dimension 256
- GELU activation, pre-norm encoder
- Temporal mean pooling
- Classification head: `128 -> 64 -> 8`

The deployment script loads only a checkpoint trained for the matching 8-class Transformer. It checks the class order in checkpoint metadata to avoid silently mixing incompatible checkpoints.

## Dataset and training

1. Open `notebooks/train_compare_8class.ipynb` in Kaggle.
2. Attach the permitted `ntu60_2d.pkl` skeleton annotation input, or enable the notebook's configured download route if allowed by your environment and dataset terms.
3. Run all cells. The notebook compares Transformer, BiLSTM, BiGRU, and TCN under the same clip-level protocol. It saves `best_transformer_8class.pth` along with metrics, confusion matrices, training curves, split manifest and config under `/kaggle/working/ntu60_8class_cliplevel_comparison/`.
4. Download `best_transformer_8class.pth` and place it in `artifacts/` locally. Do not upload restricted dataset files to GitHub.

**Model choice note:** Transformer is the project's selected deployment architecture. The included comparison notebook also trains other baselines; select a model by examining validation and held-out test metrics, not by assuming Transformer is always best.

## Local installation

Use Python 3.10 or 3.11 in a virtual environment. Install PyTorch for your OS/CPU/CUDA setup using the official PyTorch selector first, then install this project:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The first Ultralytics run may download `yolo11n-pose.pt`. Review the model weights and Ultralytics license terms before redistribution or commercial use.

## Run an MP4

Run from the repository root:

```bash
python scripts/run_demo.py --video "path/to/input.mp4" --checkpoint "artifacts/best_transformer_8class.pth" --pose-weights yolo11n-pose.pt --imgsz 640 --output "output/annotated.mp4"
```

## Run the laptop camera

```bash
python scripts/run_demo.py --camera 0 --checkpoint "artifacts/best_transformer_8class.pth" --pose-weights yolo11n-pose.pt --imgsz 640 --output "output/camera_result.mp4"
```

The camera image is mirrored before pose extraction by default, with readable on-screen text. Add `--no-mirror-camera` to disable the mirroring. Press `q` in the preview to stop. Output is MP4 only, no CSV.

Useful options:

```bash
python scripts/run_demo.py --video input.mp4 --checkpoint artifacts/best_transformer_8class.pth --clip-seconds 2.5 --stride-frames 15 --imgsz 640 --no-display
```

- `--clip-seconds`: duration kept in each person's rolling clip before resampling to 30 frames.
- `--stride-frames`: run Transformer inference every N input frames; YOLO tracking still runs on each frame.
- `--pose-weights`: start with `yolo11n-pose.pt` for speed; a larger pose model can be evaluated if needed.

## Evaluation

The notebook evaluates **one source clip = one sample = one prediction**; it does not report window-level metrics. It reports Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, Precision/Recall/F1 for `falling`, per-class classification reports and 8x8 confusion matrices, plus model latency measured in the notebook runtime.

A dataset test score does not directly establish webcam performance. The training keypoints come from a pre-extracted dataset pipeline, while live inference keypoints come from YOLO Pose, so domain shift is possible. For a credible fall-monitoring claim, separately collect and label camera videos and report fall-event recall, false alerts, event latency and FPS on the deployment hardware.

## Project status and limitations

- Research prototype, not an emergency or medical safety device.
- `falling` maps to NTU action A043; it does not cover all real-world fall types.
- One uniformly sampled sequence per complete training clip may undersample short events in long clips.
- The video demo uses a rolling recent clip rather than a labelled source clip; online predictions will differ from offline clip-level test results.
- The red alert includes heuristic pose geometry and temporal voting. These parameters are not learned and need independent validation.
- No dataset, model checkpoint, or YOLO weights are included by default.

## Repository structure

```text
configs/                       Model and preprocessing settings
notebooks/train_compare_8class.ipynb  Training + comparison experiment
src/fall_detection/            Shared labels, Transformer, preprocessing
scripts/run_demo.py            Laptop camera / MP4 inference
tests/                         Lightweight shape/smoke tests
docs/                          Dataset, evaluation, and license notes
artifacts/                      Local checkpoints (gitignored)
```

## License

This starter does not assign a repository-wide software license yet. See `LICENSE_NOTE.md`. Confirm the dataset terms and Ultralytics licensing constraints before choosing and publishing a license.
