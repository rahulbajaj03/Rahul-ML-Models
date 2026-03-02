from fastapi import FastAPI, UploadFile, File, HTTPException
from face_occlusion import predict_face_occlusion
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