"""API Service: FastAPI Web Interface for Twitch Multimodal Models.

Provides REST endpoints for text sentiment, image classification,
object detection, and LLM reactions.
"""

from typing import List, Optional
import os
import time
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.schemas import (
    SingleSentimentResult,
    BatchSentimentResult,
    ImageClassificationResult,
    LLMReplyResult,
)
from src.nlp_sentiment import ChatSentimentAnalyzer
from src.image_classify import StreamImageClassifier
from src.vision_backend import create_vision_classifier
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

app = FastAPI(
    title="ai lab",
    description="REST API для инференса моделей мультимодального анализа трансляций",
    version="1.0.0",
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


# Lazy loading of model instances
nlp_model: Optional[ChatSentimentAnalyzer] = None
image_model: Optional[StreamImageClassifier] = None
vlm_model: Optional[StreamImageClassifier] = None
video_model: Optional[LiveStreamVideoDetector] = None
llm_model: Optional[LocalLLMBrain] = None


@app.on_event("startup")
def startup_event():
    global nlp_model, image_model, video_model, llm_model
    print("[API] Initializing models on startup...")
    nlp_model = ChatSentimentAnalyzer()
    image_model = create_vision_classifier()
    video_model = LiveStreamVideoDetector()
    llm_model = LocalLLMBrain()
    print("[API] All models loaded and ready.")


class ChatBatchRequest(BaseModel):
    messages: List[str] = Field(
        ..., example=["Невероятный клатч!", "Худшая игра за сегодня", "Вперед, тащим дальше!"]
    )


class SingleChatRequest(BaseModel):
    text: str = Field(..., example="Это лучший стрим за последнее время, отличная игра!")


class ReactionRequest(BaseModel):
    streamer_speech: str = Field(..., example="Ребята, заходим на точку Б или держим позиции?")
    chat_mood: str = Field(..., example="ХАЙП / ПОЗИТИВ")
    visual_context: str = Field(..., example="Игрок целится на плент, осталось двое противников")


@app.get("/metrics")
def get_metrics():
    """Prometheus exposition metrics endpoint for Grafana and Prometheus scraping."""
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
            "/api/image/classify",
            "/api/image/vlm",
            "/api/video/detect-frame",
            "/api/llm/generate-reaction",
        ],
    }


@app.post("/api/text/sentiment", response_model=SingleSentimentResult)
def analyze_single_chat(req: SingleChatRequest):
    """Analyze sentiment for a single Twitch chat message."""
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


@app.post("/api/image/classify", response_model=ImageClassificationResult)
async def classify_stream_image(file: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    """Classify an uploaded stream screenshot or camera frame using Vision backbone."""
    if not image_model:
        raise HTTPException(status_code=503, detail="Image model not initialized")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = image_model.classify_image(tmp_path, prompt=prompt)
        safe_telemetry_dispatch(
            record_model_latency, "image_classify", result.inference_latency_ms / 1000.0
        )
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/image/vlm", response_model=ImageClassificationResult)
async def vlm_analyze_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """Deep Vision-Language Model analysis of stream scene, game HUD, and OCR."""
    global vlm_model
    if vlm_model is None:
        try:
            vlm_model = create_vision_classifier()
        except Exception as e:
            print(f"[API] Error loading VLM: {e}, falling back to image_model")
            vlm_model = image_model

    active_vlm = vlm_model or image_model
    if not active_vlm:
        raise HTTPException(status_code=503, detail="Image/VLM model not initialized")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = active_vlm.classify_image(tmp_path, prompt=prompt)
        safe_telemetry_dispatch(
            record_model_latency, "image_vlm", result.inference_latency_ms / 1000.0
        )
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/video/detect-frame")
async def detect_frame_objects(file: UploadFile = File(...)):
    """Detect objects in a single video frame using YOLOv8."""
    if not video_model:
        raise HTTPException(status_code=503, detail="Video model not initialized")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import cv2

        frame = cv2.imread(tmp_path)
        result = video_model.process_frame(frame)
        safe_telemetry_dispatch(record_model_latency, "video_detector", result.latency_ms / 1000.0)
        safe_telemetry_dispatch(record_detected_objects, result.detected_objects)
        return {
            "detected_objects": [
                {"class": o.class_name, "confidence": o.confidence, "box": o.box}
                for o in result.detected_objects
            ],
            "latency_ms": result.latency_ms,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/llm/generate-reaction", response_model=LLMReplyResult)
def generate_bot_reaction(req: ReactionRequest):
    """Generate a contextual Twitch chat response using the local LLM brain."""
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
