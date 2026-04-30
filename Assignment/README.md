# Vehicle Image Quality Classification

A YOLOv8-based system that classifies vehicle images as **good**, **blur**, or **low_light** to screen out poor-quality images before they reach downstream damage-detection models.

## Project Structure

```
Assignment/
├── inference.py            # Inference script (single dir → JSON/CSV output)
├── evaluate.py             # Evaluation script (runs on validation set)
├── PERFORMANCE_REPORT.md   # Full evaluation report with metrics & analysis
├── README.md               # This file
├── data/
│   ├── Blur/
│   │   ├── Train/          # Training data (blur_images, good_images)
│   │   └── val/            # Validation data (blur_images, good_images)
│   └── Low_Light/
│       ├── Train/          # Training data (low_light_images, good_images)
│       └── Val/            # Validation data (low_light_images, good_images)
└── eval_output/            # Generated evaluation artifacts
    ├── evaluation_report.json
    └── confusion_matrix.png

camcon_flag_model.pt        # Trained YOLOv8 model weights (root level)
```

## Environment Setup

### Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install ultralytics torch torchvision opencv-python numpy
pip install scikit-learn matplotlib seaborn
```

## Usage

### 1. Run Inference on a Directory of Images

```bash
python Assignment/inference.py \
    --input_dir <path_to_images> \
    --model camcon_flag_model.pt \
    --output results \
    --threshold 0.7
```

**Arguments:**
| Argument      | Default                | Description                          |
|---------------|------------------------|--------------------------------------|
| `--input_dir` | (required)             | Directory containing images          |
| `--model`     | `../camcon_flag_model.pt` | Path to trained `.pt` model file  |
| `--output`    | `results`              | Output file prefix (creates .json and .csv) |
| `--threshold` | `0.7`                  | Confidence threshold for Accept/Reject |

**Output format (JSON):**
```json
[
  {
    "filename": "image001.jpg",
    "predicted_class": "good_image",
    "confidence": 0.9234,
    "status": "Accept",
    "inference_time_ms": 25.4
  }
]
```

**Decision Logic:**
- `good_image` with confidence ≥ threshold → **Accept**
- `blur` or `low_light` with confidence ≥ threshold → **Reject**
- Any prediction below threshold → **Reject** (safety default)

### 2. Run Evaluation on Validation Set

```bash
python Assignment/evaluate.py --threshold 0.7 --output_dir Assignment/eval_output
```

This produces:
- Classification report (precision, recall, F1 per class)
- Confusion matrix (printed + saved as PNG)
- Inference latency statistics
- Failure analysis examples
- Full report as `evaluation_report.json`

## Model Details

- **Base:** YOLOv8 (fine-tuned)
- **Task:** Object detection (single-class per image)
- **Labels:** `0: good_image`, `1: blur`, `2: low_light`
- **Validation Accuracy:** 97.6% (573 images)
- **Avg Latency:** ~27 ms/image (CPU)

## Performance Summary

| Class      | Precision | Recall | F1    |
|------------|-----------|--------|-------|
| good_image | 0.977     | 0.980  | 0.979 |
| blur       | 1.000     | 0.938  | 0.968 |
| low_light  | 0.952     | 1.000  | 0.975 |

See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) for the full evaluation report including confusion matrix, failure analysis, and edge case discussion.
