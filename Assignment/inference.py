"""
Vehicle Image Quality Classification — Inference Script

Accepts a directory of images, runs YOLOv8-based quality classification,
and outputs results as JSON and CSV with Accept/Reject status.

Usage:
    python inference.py --input_dir <image_folder> --output results --threshold 0.7
"""

import argparse
import json
import csv
import os
import time
from pathlib import Path

from ultralytics import YOLO
import torch

# Default paths
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "camcon_flag_model.pt")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Class label mapping
LABEL_MAP = {0: "good_image", 1: "blur", 2: "low_light"}
REJECT_CLASSES = {"blur", "low_light"}


def load_model(model_path: str) -> YOLO:
    """Load the YOLOv8 model from the given path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return YOLO(model_path)


def classify_image(model: YOLO, image_path: str, threshold: float) -> dict:
    """
    Run inference on a single image.

    Returns a dict with:
        - filename
        - predicted_class
        - confidence
        - status (Accept / Reject)
    """
    results = model.predict(image_path, verbose=False)
    det = results[0]

    filename = os.path.basename(image_path)

    # No detections — treat as reject (uncertain)
    if len(det.boxes) == 0:
        return {
            "filename": filename,
            "predicted_class": "unknown",
            "confidence": 0.0,
            "status": "Reject",
        }

    # Pick the highest-confidence detection
    boxes = det.boxes
    max_idx = torch.argmax(boxes.conf).item()
    cls_id = int(boxes.cls[max_idx])
    conf = float(boxes.conf[max_idx])
    label = LABEL_MAP.get(cls_id, "unknown")

    # Decision logic
    if label == "good_image" and conf >= threshold:
        status = "Accept"
    elif label in REJECT_CLASSES and conf >= threshold:
        status = "Reject"
    else:
        # Low confidence — default reject for safety
        status = "Reject"

    return {
        "filename": filename,
        "predicted_class": label,
        "confidence": round(conf, 4),
        "status": status,
    }


def run_inference(input_dir: str, model_path: str, output_prefix: str, threshold: float):
    """Run inference on all images in input_dir and save results."""
    model = load_model(model_path)

    image_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if not image_files:
        print(f"No supported images found in {input_dir}")
        return []

    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Using threshold: {threshold}")

    results = []
    total_time = 0.0

    for img_path in image_files:
        start = time.time()
        result = classify_image(model, img_path, threshold)
        elapsed = time.time() - start
        total_time += elapsed
        result["inference_time_ms"] = round(elapsed * 1000, 2)
        results.append(result)

    avg_latency = (total_time / len(results)) * 1000 if results else 0
    print(f"\nProcessed {len(results)} images")
    print(f"Average inference latency: {avg_latency:.2f} ms/image")

    # Save JSON
    json_path = f"{output_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to {json_path}")

    # Save CSV
    csv_path = f"{output_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "predicted_class", "confidence", "status", "inference_time_ms"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV results saved to {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Vehicle Image Quality Classifier — Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to .pt model file")
    parser.add_argument("--output", type=str, default="results", help="Output file prefix (without extension)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold (default: 0.7)")
    args = parser.parse_args()

    run_inference(args.input_dir, args.model, args.output, args.threshold)


if __name__ == "__main__":
    main()
