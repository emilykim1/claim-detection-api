from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import time
import logging

try:
    from api.predictor import ClaimPredictor
except ModuleNotFoundError:
    from predictor import ClaimPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = ClaimPredictor(model_path="assets/models/llama-1b-claim-ft/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    predictor.load()
    logger.info("Model ready")
    yield
    # Cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="Claim Detection API",
    description="Detects whether a sentence contains a factual claim.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sentence: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["The Empire State Building is the tallest building in New York City."],
    )

    @field_validator("sentence")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class PredictResponse(BaseModel):
    sentence: str
    is_claim: bool
    confidence: float
    claim_probability: float
    latency_ms: float


# ── Middleware: request timing ────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(elapsed, 2))
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.post("/reload-model")
def reload_model():
    predictor.load()
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    start = time.perf_counter()
    try:
        result = predictor.predict(body.sentence)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    latency = round((time.perf_counter() - start) * 1000, 2)

    return PredictResponse(
        sentence=body.sentence,
        latency_ms=latency,
        **result,
    )


@app.post("/predict/batch")
def predict_batch(sentences: list[str]):
    """Predict on multiple sentences at once (max 32)."""
    if len(sentences) > 32:
        raise HTTPException(status_code=400, detail="Max 32 sentences per batch")
    return [predictor.predict(s) for s in sentences]
