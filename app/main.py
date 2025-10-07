from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from PIL import Image
import io

from .model import ModelService, InferenceResult


app = FastAPI(title="Leaf Segmentation API", version="1.0.0")


# Allow all origins by default; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_service: ModelService | None = None


@app.on_event("startup")
def load_model() -> None:
    global model_service
    model_service = ModelService()


@app.get("/health")
def health() -> dict:
    if model_service is None:
        return {"status": "loading"}
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are supported")

    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    try:
        result: InferenceResult = model_service.run_inference(image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return JSONResponse(
        {
            "success": True,
            "results": {
                "disease_ratio": result.disease_ratio,
                "healthy_ratio": result.healthy_ratio,
                "segmentation_mask": result.segmentation_mask_base64,
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



