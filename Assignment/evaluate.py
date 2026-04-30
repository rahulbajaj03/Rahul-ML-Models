"""
Vehicle Image Quality Classification — Evaluation Script

Runs the model on the validation dataset, computes:
  - Per-class Precision, Recall, F1-Score
  - Confusion Matrix
  - Average Inference Latency
  - Failure analysis examples

Usage:
    python evaluate.py --threshold 0.7
"""

import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from ultralytics import YOLO

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "camcon_flag_model.pt")

BLUR_VAL_BLUR = os.path.join(BASE_DIR, "data", "Blur", "val", "blur_images")
BLUR_VAL_GOOD = os.path.join(BASE_DIR, "data", "Blur", "val", "good_images")
LOW_LIGHT_VAL_LOW = os.path.join(BASE_DIR, "data", "Low_Light", "Val", "low_light_images")
LOW_LIGHT_VAL_GOOD = os.path.join(BASE_DIR, "data", "Low_Light", "Val", "good_images")

LABEL_MAP = {0: "good_image", 1: "blur", 2: "low_light"}
CLASS_NAMES = ["good_image", "blur", "low_light"]
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_image_files(directory: str) -> list:
    """Return sorted list of image file paths from a directory."""
    if not os.path.isdir(directory):
        print(f"Warning: directory not found: {directory}")
        return []
    return sorted(
        [os.path.join(directory, f) for f in os.listdir(directory)
         if Path(f).suffix.lower() in SUPPORTED_EXT]
    )


def predict_single(model: YOLO, image_path: str) -> tuple:
    """Predict on a single image. Returns (predicted_label, confidence, time_ms)."""
    start = time.time()
    results = model.predict(image_path, verbose=False)
    elapsed_ms = (time.time() - start) * 1000

    det = results[0]
    if len(det.boxes) == 0:
        return "unknown", 0.0, elapsed_ms

    boxes = det.boxes
    max_idx = torch.argmax(boxes.conf).item()
    cls_id = int(boxes.cls[max_idx])
    conf = float(boxes.conf[max_idx])
    label = LABEL_MAP.get(cls_id, "unknown")
    return label, conf, elapsed_ms


def build_val_dataset() -> list:
    """
    Build validation dataset as list of (image_path, ground_truth_label).
    
    Ground truth mapping:
      - Blur/val/blur_images/* → "blur"
      - Blur/val/good_images/* → "good_image"
      - Low_Light/Val/low_light_images/* → "low_light"
      - Low_Light/Val/good_images/* → "good_image"
    """
    dataset = []

    for img_path in get_image_files(BLUR_VAL_BLUR):
        dataset.append((img_path, "blur"))

    for img_path in get_image_files(BLUR_VAL_GOOD):
        dataset.append((img_path, "good_image"))

    for img_path in get_image_files(LOW_LIGHT_VAL_LOW):
        dataset.append((img_path, "low_light"))

    for img_path in get_image_files(LOW_LIGHT_VAL_GOOD):
        dataset.append((img_path, "good_image"))

    return dataset


def plot_confusion_matrix(cm, class_names, output_path):
    """Save confusion matrix as a heatmap image."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def run_evaluation(threshold: float, output_dir: str):
    """Run full evaluation on the validation set."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Threshold: {threshold}\n")

    dataset = build_val_dataset()
    print(f"Total validation images: {len(dataset)}")

    y_true = []
    y_pred = []
    latencies = []
    failures = []

    for img_path, gt_label in dataset:
        pred_label, conf, time_ms = predict_single(model, img_path)
        latencies.append(time_ms)

        y_true.append(gt_label)
        y_pred.append(pred_label)

        # Track failures for analysis
        if pred_label != gt_label:
            failures.append({
                "filename": os.path.basename(img_path),
                "ground_truth": gt_label,
                "predicted": pred_label,
                "confidence": round(conf, 4),
                "path": img_path,
            })

    # --- Metrics ---
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0)
    print(report)

    # Per-class precision, recall, f1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_NAMES, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(output_dir, "confusion_matrix.png"))

    # Latency stats
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"\nInference Latency:")
    print(f"  Average:  {avg_latency:.2f} ms")
    print(f"  Median:   {median_latency:.2f} ms")
    print(f"  P95:      {p95_latency:.2f} ms")

    # Accept/Reject analysis
    accept_count = sum(
        1 for gt, pred in zip(y_true, y_pred)
        if pred == "good_image"
    )
    reject_count = len(y_true) - accept_count
    print(f"\nAccept/Reject Summary:")
    print(f"  Accepted: {accept_count}")
    print(f"  Rejected: {reject_count}")

    # Failure analysis (top 5)
    print(f"\nTotal misclassifications: {len(failures)}")
    print("\nSample Failure Cases (up to 5):")
    for f in failures[:5]:
        print(f"  {f['filename']}: GT={f['ground_truth']}, Pred={f['predicted']}, Conf={f['confidence']}")

    # --- Save full report as JSON ---
    report_data = {
        "threshold": threshold,
        "total_images": len(dataset),
        "per_class_metrics": {},
        "confusion_matrix": cm.tolist(),
        "latency": {
            "average_ms": round(avg_latency, 2),
            "median_ms": round(median_latency, 2),
            "p95_ms": round(p95_latency, 2),
        },
        "accept_reject": {"accepted": accept_count, "rejected": reject_count},
        "total_misclassifications": len(failures),
        "failure_examples": failures[:5],
    }

    for i, cls_name in enumerate(CLASS_NAMES):
        report_data["per_class_metrics"][cls_name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    return report_data, failures


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vehicle Image Quality Classifier")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold (default: 0.7)")
    parser.add_argument("--output_dir", type=str, default="Assignment/eval_output", help="Output directory")
    args = parser.parse_args()

    run_evaluation(args.threshold, args.output_dir)


if __name__ == "__main__":
    main()
