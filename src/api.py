"""API Service: FastAPI interface for experiments with pretrained models.

The text emotion model is the primary API of the report (section 9.3); the scene, detection,
frame-question and reaction routes expose the other models for experiments.
"""

import io
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from PIL import Image
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.schemas import (
    SingleSentimentResult,
    BatchSentimentResult,
    ImageClassificationResult,
    LLMReplyResult,
)
from src.nlp_sentiment import ChatSentimentAnalyzer
from src.scene_classify import SceneClassifier
from src.vision_backend import OllamaVisionClassifier
from src.video_live import LiveStreamVideoDetector
from src.llm_brain import LocalLLMBrain
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import (
    API_REQUESTS_TOTAL,
    API_REQUEST_DURATION,
    get_latest_metrics,
    CONTENT_TYPE_LATEST,
    record_model_latency,
    record_chat_stats,
    record_detected_objects,
    record_bot_reply,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Model references populated at startup.
nlp_model: Optional[ChatSentimentAnalyzer] = None
scene_model: Optional[SceneClassifier] = None
vision_model: Optional[OllamaVisionClassifier] = None
video_model: Optional[LiveStreamVideoDetector] = None
llm_model: Optional[LocalLLMBrain] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global nlp_model, scene_model, vision_model, video_model, llm_model
    print("[API] Initializing models on startup...")
    nlp_model = ChatSentimentAnalyzer()
    scene_model = SceneClassifier()
    vision_model = OllamaVisionClassifier()
    video_model = LiveStreamVideoDetector()
    llm_model = LocalLLMBrain()
    print("[API] Model instances initialized; Ollama inference is requested separately.")
    yield


app = FastAPI(
    title="llm-lab",
    description=(
        "API к готовым моделям: эмоции и тональность текста (основной маршрут), "
        "тип сцены кадра (CLIP), детекция объектов (YOLO), вопрос о кадре и реакция LLM. "
        "Метрики доступны в /metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if endpoint not in ("/", "/metrics"):
            method = (
                request.method
                if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}
                else "OTHER"
            )
            safe_telemetry_dispatch(
                lambda: API_REQUESTS_TOTAL.labels(
                    endpoint=endpoint, method=method, status_code=str(status_code)
                ).inc()
            )
            safe_telemetry_dispatch(
                lambda: API_REQUEST_DURATION.labels(endpoint=endpoint).observe(
                    time.perf_counter() - start
                )
            )


class ChatBatchRequest(BaseModel):
    messages: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=[["Невероятный клатч!", "Худшая игра за сегодня", "Вперед, тащим дальше!"]],
    )


class SingleChatRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, examples=["Это лучший стрим за последнее время, отличная игра!"]
    )


class ReactionRequest(BaseModel):
    streamer_speech: str = Field(..., examples=["Ребята, заходим на точку Б или держим позиции?"])
    chat_mood: str = Field(..., examples=["ХАЙП / ПОЗИТИВ"])
    visual_context: str = Field(..., examples=["Игрок целится на плент, осталось двое противников"])


def _read_image(file: UploadFile) -> Image.Image:
    """Decodes an uploaded image in memory; rejects oversized or non-image files with 4xx."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except Exception:  # PIL raises several types for truncated or non-image data.
        raise HTTPException(status_code=400, detail="File is not a readable image")


@app.get("/metrics")
def get_metrics():
    """Expose process metrics for Prometheus; Grafana visualizes the collected data."""
    return Response(content=get_latest_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Multimodal Stream Analysis API",
        "endpoints": [
            "/metrics",
            "/api/text/sentiment",
            "/api/text/batch-sentiment",
            "/api/image/scene",
            "/api/image/vlm",
            "/api/video/detect-frame",
            "/api/llm/generate-reaction",
        ],
    }


@app.post("/api/text/sentiment", response_model=SingleSentimentResult)
def analyze_single_chat(req: SingleChatRequest):
    """Analyze sentiment for a single message."""
    if not nlp_model:
        raise HTTPException(status_code=503, detail="NLP model not initialized")
    res = nlp_model.analyze_message(req.text)
    safe_telemetry_dispatch(record_model_latency, "nlp_sentiment", res.latency_ms / 1000.0)
    pos = 1.0 if "POS" in res.label.upper() else 0.0
    neg = 1.0 if "NEG" in res.label.upper() else 0.0
    safe_telemetry_dispatch(record_chat_stats, pos, neg)
    return res


@app.post("/api/text/batch-sentiment", response_model=BatchSentimentResult)
def analyze_chat_batch(req: ChatBatchRequest):
    """Analyze a batch of chat messages to calculate dominant room sentiment."""
    if not nlp_model:
        raise HTTPException(status_code=503, detail="NLP model not initialized")
    res = nlp_model.analyze_chat_batch(req.messages)
    safe_telemetry_dispatch(record_model_latency, "nlp_sentiment", res.latency_ms / 1000.0)
    safe_telemetry_dispatch(record_chat_stats, res.positive_ratio, res.negative_ratio)
    return res


@app.post("/api/image/scene", response_model=ImageClassificationResult)
def classify_scene(file: UploadFile = File(...)):
    """Classify the scene type of a frame with CLIP zero-shot classification."""
    if not scene_model:
        raise HTTPException(status_code=503, detail="Scene model not initialized")
    result = scene_model.classify_image(_read_image(file))
    safe_telemetry_dispatch(
        record_model_latency, "image_scene", result.inference_latency_ms / 1000.0
    )
    return result


@app.post("/api/image/vlm", response_model=ImageClassificationResult)
def vlm_analyze_image(file: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    """Ask the local LLM about a frame, including scene or interface text."""
    if not vision_model:
        raise HTTPException(status_code=503, detail="Vision model not initialized")
    image = _read_image(file)
    try:
        result = vision_model.classify_image(image, prompt=prompt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama vision request failed: {exc}")
    safe_telemetry_dispatch(record_model_latency, "image_vlm", result.inference_latency_ms / 1000.0)
    return result


@app.post("/api/video/detect-frame")
def detect_frame_objects(file: UploadFile = File(...)):
    """Detect objects in a single image using the configured YOLO model."""
    if not video_model:
        raise HTTPException(status_code=503, detail="Video model not initialized")
    frame = np.asarray(_read_image(file))[:, :, ::-1].copy()  # RGB -> BGR, as from OpenCV.
    result = video_model.process_frame(frame, annotate=False)
    safe_telemetry_dispatch(record_model_latency, "video_yolo", result.latency_ms / 1000.0)
    safe_telemetry_dispatch(record_detected_objects, result.detected_objects)
    return {
        "detected_objects": [
            {"class": o.class_name, "confidence": o.confidence, "box": o.box}
            for o in result.detected_objects
        ],
        "latency_ms": result.latency_ms,
    }


@app.post("/api/llm/generate-reaction", response_model=LLMReplyResult)
def generate_bot_reaction(req: ReactionRequest):
    """Generate a reaction from speech, message sentiment, and visual context using the local LLM."""
    if not llm_model:
        raise HTTPException(status_code=503, detail="LLM model not initialized")
    try:
        res = llm_model.generate_reply(
            streamer_speech=req.streamer_speech,
            chat_mood=req.chat_mood,
            visual_context=req.visual_context,
            bot_personality="",
            use_stream_prompt=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama generation failed: {exc}")
    safe_telemetry_dispatch(record_model_latency, "llm_brain", res.latency_ms / 1000.0)
    safe_telemetry_dispatch(record_bot_reply, res.engine)
    return res


if __name__ == "__main__":
    import uvicorn

    host = settings.network.api_host
    port = settings.network.api_port
    print(f"[API] Starting API server on http://{host}:{port} ...")
    uvicorn.run("src.api:app", host=host, port=port, reload=False)
