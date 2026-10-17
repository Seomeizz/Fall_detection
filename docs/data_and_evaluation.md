# Dataset, label mapping, and evaluation protocol

## Dataset

The experiment notebook expects an MMAction2-style `ntu60_2d.pkl` containing annotations with `frame_dir`, `keypoint`, `keypoint_score`, and split metadata. The notebook does not contain the original RGB videos; it consumes pre-extracted 2D keypoints.

Do not include the data pickle or derivatives in this repository unless you have explicit permission to redistribute them. Ask users to obtain the dataset from its official source and follow the applicable terms.

## Label mapping

The project groups selected NTU IDs into eight author-defined classes. This is not the standard NTU RGB+D 60-class benchmark. Review `src/fall_detection/labels.py` and `configs/model_config.yaml` before training. In particular, the current `standing` group contains several action IDs whose semantics include object interaction or other activities; do not describe it as a pure static-standing class without revising the mapping.

## Split policy

The notebook tries to map official X-Sub train/validation or train/test IDs from the pickle. It then creates a validation split from the training pool. If the expected split fields cannot be mapped, it falls back to a subject-level split where possible, or an explicitly reported clip-stratified fallback. Inspect `split_manifest.json` and `experiment_config.json` from every run. For a paper, do not claim the official X-Sub protocol unless the generated manifest confirms it.

## Input pipeline

Each source clip is converted to one sequence of 30 uniformly sampled timestamps. The raw coordinates and confidences are sampled first, then missing/low-confidence coordinates are interpolated, then a single hip origin and median torso length are used to normalize the entire clip. The result contains 34 normalized XY coordinates plus 17 confidence values per frame: `[30, 51]`.

The inference code calls the same functions from `src/fall_detection/preprocessing.py`. Do not change one side without updating the other and re-running validation.

## Metrics

Primary multiclass metric recommended for model selection: validation Macro-F1 across all eight mapped classes. On a held-out test set, report Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, per-class Precision/Recall/F1, and especially `falling` Precision/Recall/F1. Review the confusion matrix and support counts; do not select or tune a threshold using test labels.

If claiming real-time fall-detection performance, additionally evaluate the complete event detector on independently labelled videos and report event-level recall, false alerts per hour, event latency and inference FPS. Classifying action A043 is not equivalent to robust detection of all real-world fall events.
