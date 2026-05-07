from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.predictor import CropPredictor


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

app = FastAPI(title="Crop Disease Classifier")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def get_predictor() -> CropPredictor:
    return CropPredictor(base_dir=BASE_DIR / "artifacts")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    predictor = get_predictor()
    crop_options = predictor.available_crops()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "crop_options": crop_options,
            "prediction": None,
            "selected_crop": None,
            "error": None,
            "has_models": bool(crop_options),
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, crop: str = Form(...), image: UploadFile = File(...)) -> HTMLResponse:
    predictor = get_predictor()
    crop_options = predictor.available_crops()
    try:
        result = await predictor.predict(crop=crop, uploaded_file=image)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "crop_options": crop_options,
                "prediction": result,
                "selected_crop": crop,
                "error": None,
                "has_models": bool(crop_options),
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "crop_options": crop_options,
                "prediction": None,
                "selected_crop": crop,
                "error": str(exc),
                "has_models": bool(crop_options),
            },
        )
