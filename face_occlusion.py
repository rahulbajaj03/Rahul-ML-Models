from ultralytics import YOLO
import torch


face_occlusion_model = YOLO("modelFiles/model_occlusion_check.pt")
occlusion_type_model = YOLO("modelFiles/occlusion_items_model.pt")


OCCLUSION_TYPE_LABELS = {
    0: "image is blur",
    1: "Face not visible due to room lighting is dull",
    2: "remove mask/hand/cloth/helmet from face",
    3: "keep face straight"
}


def predict_face_occlusion(image):
    """
    Accepts PIL image
    Returns: (result_string, confidence_or_None)
    """

    try:
        # -------- Model 1 --------
        det1 = face_occlusion_model.predict(image, verbose=False)

        if len(det1[0].boxes) == 0:
            return "no face detected in image", None

        boxes1 = det1[0].boxes
        confs1 = boxes1.conf

        max_idx1 = torch.argmax(confs1).item()
        cls1 = int(boxes1.cls[max_idx1])

        # ✅ If face is correct → NO confidence
        if cls1 == 1:
            return "face correct", None

        # -------- Model 2 --------
        det2 = occlusion_type_model.predict(image, verbose=False)

        if len(det2[0].boxes) == 0:
            return "no face detected in image", None

        boxes2 = det2[0].boxes
        confs2 = boxes2.conf

        max_idx2 = torch.argmax(confs2).item()
        cls2 = int(boxes2.cls[max_idx2])
        conf2 = float(confs2[max_idx2])

        final_label = OCCLUSION_TYPE_LABELS.get(cls2, "Face occlusion detected, Please take picture again")

        return final_label, conf2

    except Exception:
        return "error", None
