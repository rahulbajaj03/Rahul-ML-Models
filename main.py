from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from face_occlusion import predict_face_occlusion
from image_quality import predict_image_quality
from helperfunction import read_image_from_bytes

app = FastAPI()

@app.post("/predict-face-occlusion/")
async def predict_face(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        image, _ = read_image_from_bytes(image_bytes)

        label, confidence = predict_face_occlusion(image)

        return {
            "label": label,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-image-quality/")
async def predict_quality(
    file: UploadFile = File(...),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Confidence threshold for Accept/Reject")
):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        image, _ = read_image_from_bytes(image_bytes)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        label, confidence, status = predict_image_quality(image, threshold)

        return {
            "label": label,
            "confidence": confidence,
            "status": status
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))