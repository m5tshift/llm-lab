"""Vision adapter using the same local Ollama weights as the conversation model."""

import base64
import io
import time

from src.core.config import settings
from src.core.interfaces import BaseImageClassifier
from src.core.schemas import ImageClassificationResult
from src.image_classify import StreamImageClassifier
from src.llm_brain import _clean_llm_reply
from src.ollama_client import generate


class OllamaVisionClassifier(BaseImageClassifier):
    is_vlm = True
    is_caption_model = False

    def __init__(self):
        self.model_name = settings.models.ollama_model

    def classify_image(self, image_input, top_k=5, prompt=None, max_new_tokens=160):
        image = StreamImageClassifier._load_pil_image(image_input).copy()
        image.thumbnail((1280, 1280))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        started = time.perf_counter()
        prompt_text = (prompt or "").strip()
        encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
        # /api/generate treats an empty prompt as a model-load request.
        # An image-only chat message performs inference without adding instructions.
        visual_input = (
            {"prompt": prompt_text, "images": [encoded_image]}
            if prompt_text
            else {"messages": [{"role": "user", "content": "", "images": [encoded_image]}]}
        )
        data = generate(
            settings.models.ollama_base_url,
            {
                "model": self.model_name,
                **visual_input,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {
                    "num_ctx": settings.models.ollama_context,
                    "num_predict": max_new_tokens,
                    "temperature": 0.2,
                },
            },
            settings.models.ollama_timeout,
        )
        description = _clean_llm_reply(
            data.get("response", "") or data.get("message", {}).get("content", "")
        )
        if not description:
            raise RuntimeError("Ollama returned an empty visual description")
        return ImageClassificationResult(
            top_prediction=description,
            confidence=0.0,  # Generative captions do not have a calibrated class probability.
            predictions=[],
            inference_latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def create_vision_classifier():
    if settings.models.vision_backend == "ollama":
        return OllamaVisionClassifier()
    return StreamImageClassifier(model_name=settings.models.vlm_model)
