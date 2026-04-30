from ultralytics import YOLO
import torch

image_quality_model = YOLO("camcon_flag_model.pt")

QUALITY_LABELS = {0: "good_image", 1: "blur", 2: "low_light"}
REJECT_CLASSES = {"blur", "low_light"}
DEFAULT_THRESHOLD = 0.7


def predict_image_quality(image, threshold=DEFAULT_THRESHOLD):
    """
    Accepts a CV2/numpy image.
    Returns: (label, confidence, status)
      - label: "good_image", "blur", "low_light", or "unknown"
      - confidence: float or None
      - status: "Accept" or "Reject"
    """
    try:
        results = image_quality_model.predict(image, verbose=False)
        det = results[0]

        if len(det.boxes) == 0:
            return "unknown", None, "Reject"

        boxes = det.boxes
        max_idx = torch.argmax(boxes.conf).item()
        cls_id = int(boxes.cls[max_idx])
        conf = float(boxes.conf[max_idx])
        label = QUALITY_LABELS.get(cls_id, "unknown")

        if label == "good_image" and conf >= threshold:
            status = "Accept"
        elif label in REJECT_CLASSES and conf >= threshold:
            status = "Reject"
        else:
            status = "Reject"

        return label, round(conf, 4), status

    except Exception:
        return "error", None, "Reject"
