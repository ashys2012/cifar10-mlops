"""FastAPI inference server for the CIFAR-10 ResNet18 model.

Load the model once at startup via MLflow; expose /predict and /predict_with_confidence.

Model loading priority (first that works wins):
    1. MODEL_URI env var — explicit override, e.g. "runs:/abc123/model"
    2. MLflow Model Registry — loads the model aliased "champion" under
       MODEL_NAME (default: "cifar10-resnet18"). Falls back to the latest
       version if the alias doesn't exist yet.
    3. Best finished run — scans all runs in the experiment and picks the
       one with the highest val_accuracy (or accuracy). Use this path until
       you've registered a model in the registry.

Environment variables:
    MLFLOW_TRACKING_URI  — defaults to your ZenML SQLite store path
    MODEL_URI            — skip registry/run search, load this URI directly
    MODEL_NAME           — registry model name (default: cifar10-resnet18)
    MLFLOW_EXPERIMENT    — experiment name to search (default: "cifar_10")
    ACCURACY_METRIC      — run metric used to rank runs (default: val_accuracy)

Run with:
    python app.py
    # or:
    uvicorn app:app --host 0.0.0.0 --port 8000
    # GPU: single worker only (model lives on one GPU)
    # CPU: add --workers 4

Registering a model after training (do this once, then the registry path takes over):
    import mlflow
    mlflow.set_tracking_uri("sqlite:////path/to/mlflow.db")
    result = mlflow.register_model("runs:/<run_id>/model", "cifar10-resnet18")
    # Then promote it:
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias("cifar10-resnet18", "champion", result.version)
"""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config — override any of these with environment variables
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    "sqlite:////home/achazhoor/Documents/2026/Code/cifar_10/mlflow.db",
)

MODEL_NAME       = os.environ.get("MODEL_NAME", "cifar10-resnet18")
EXPERIMENT_NAME  = os.environ.get("MLFLOW_EXPERIMENT", "training_pipeline")
ACCURACY_METRIC  = os.environ.get("ACCURACY_METRIC", "val_accuracy")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model state — populated at startup, shared across requests
# ---------------------------------------------------------------------------
_model:     torch.nn.Module | None = None
_device:    torch.device    | None = None
_model_uri: str             | None = None 


# ---------------------------------------------------------------------------
# Model URI resolution
# ---------------------------------------------------------------------------
def _resolve_model_uri() -> str:
    """
    Return the best MODEL_URI to load, following the priority order in the
    module docstring. Raises RuntimeError if no model can be found.
    """
    client = mlflow.tracking.MlflowClient()

    # ── Priority 1: explicit env-var override ────────────────────────────────
    env_uri = os.environ.get("MODEL_URI")
    if env_uri:
        logger.info("[Priority 1] Using explicit MODEL_URI from environment: %s", env_uri)
        return env_uri

    # ── Priority 2: MLflow Model Registry ───────────────────────────────────
    try:
        # Try "champion" alias first (industry standard: promote only the best)
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        uri = f"models:/{MODEL_NAME}@champion"
        logger.info(
            "[Priority 2a] Registry champion found — model=%s version=%s uri=%s",
            MODEL_NAME, mv.version, uri,
        )
        return uri
    except mlflow.exceptions.MlflowException:
        pass  # alias doesn't exist yet — try latest version

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            uri = f"models:/{MODEL_NAME}/{latest.version}"
            logger.info(
                "[Priority 2b] Registry latest version — model=%s version=%s uri=%s",
                MODEL_NAME, latest.version, uri,
            )
            return uri
    except mlflow.exceptions.MlflowException:
        pass  # model not registered at all yet

    # ── Priority 3: best finished run in the experiment ──────────────────────
    logger.info(
        "[Priority 3] No registry entry found for '%s'. "
        "Searching experiment '%s' for best run by metric '%s'.",
        MODEL_NAME, EXPERIMENT_NAME, ACCURACY_METRIC,
    )
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' not found. "
            "Have you run the training pipeline at least once? "
            "Check MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT env vars."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{ACCURACY_METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        # Fallback: try plain "accuracy" in case the metric name differs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["metrics.accuracy DESC"],
            max_results=1,
        )

    if not runs:
        raise RuntimeError(
            f"No FINISHED runs found in experiment '{EXPERIMENT_NAME}'. "
            "Run the training pipeline first."
        )

    best_run = runs[0]
    metric_val = best_run.data.metrics.get(ACCURACY_METRIC) or best_run.data.metrics.get("accuracy")
    uri = f"runs:/{best_run.info.run_id}/model"
    logger.info(
        "[Priority 3] Best run — id=%s %s=%.4f uri=%s",
        best_run.info.run_id, ACCURACY_METRIC, metric_val or 0.0, uri,
    )
    return uri


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def _load_model() -> tuple[torch.nn.Module, torch.device, str]:
    """Resolve the best model URI, load it from MLflow, and return
    (model, device, resolved_uri)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uri = _resolve_model_uri()
    logger.info("Loading model from: %s  (device: %s)", uri, device)
    m = mlflow.pytorch.load_model(uri, map_location=device)
    m.to(device).eval()
    logger.info("Model ready.")
    return m, device, uri


# ---------------------------------------------------------------------------
# Lifespan — load model once on startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _model_uri
    try:
        _model, _device, _model_uri = _load_model()
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise RuntimeError(f"Model load failed: {exc}") from exc
    yield
    _model = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CIFAR-10 Classification API",
    description=(
        "Inference API for a ResNet18 model fine-tuned on CIFAR-10.\n\n"
        "Model loading priority:\n"
        "1. `MODEL_URI` env var\n"
        "2. MLflow Model Registry (`champion` alias → latest version)\n"
        "3. Best finished run ranked by `val_accuracy`"
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _read_image(file: UploadFile) -> Image.Image:
    """Validate, size-check, and decode an uploaded image."""
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PNG or JPEG.",
        )
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc


def _run_inference(image: Image.Image) -> torch.Tensor:
    """Preprocess image and return raw logits (1 × num_classes)."""
    tensor = PREPROCESS(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        return _model(tensor)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Status"])
def health_check():
    return {"status": "ok", "device": str(_device)}


@app.get("/metadata", tags=["Metadata"])
def get_metadata():
    return {
        "model_architecture": "ResNet18",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "device": str(_device),
        "model_uri": _model_uri,          # shows exactly what was loaded
        "tracking_uri": mlflow.get_tracking_uri(),
    }


@app.get("/class_names", tags=["Metadata"])
def get_class_names():
    return {"class_names": CLASS_NAMES}


@app.post("/predict", tags=["Inference"])
async def predict(file: Annotated[UploadFile, File()]):
    """Return the single most likely class label."""
    image = _read_image(file)
    logits = _run_inference(image)
    predicted_idx = int(torch.argmax(logits, dim=1).item())
    return {"predicted_class": CLASS_NAMES[predicted_idx]}


@app.post("/predict_with_confidence", tags=["Inference"])
async def predict_with_confidence(
    file: Annotated[UploadFile, File()],
    top_k: int = 3,
):
    """Return the top-k predictions with softmax confidence scores."""
    if not 1 <= top_k <= len(CLASS_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between 1 and {len(CLASS_NAMES)}.",
        )
    image = _read_image(file)
    logits = _run_inference(image)
    probs = torch.nn.functional.softmax(logits, dim=1)
    top_probs, top_idxs = torch.topk(probs, k=top_k)
    predictions = [
        {"class": CLASS_NAMES[idx], "confidence": round(float(prob), 4)}
        for idx, prob in zip(top_idxs[0].tolist(), top_probs[0].tolist())
    ]
    return {"predictions": predictions}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Single worker for GPU. Add workers=4 for CPU-only deployments.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)
