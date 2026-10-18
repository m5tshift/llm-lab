"""Reproducible local smoke check; this is not a live-stream quality benchmark."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.core.config import settings
from src.llm_brain import LocalLLMBrain
from src.vision_backend import OllamaVisionClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="samples/sample_stream.jpg")
    parser.add_argument("--output", default="output/checks/qwen35_smoke.json")
    args = parser.parse_args()
    base_url = settings.models.ollama_base_url.rstrip("/")
    brain = LocalLLMBrain()
    vision = OllamaVisionClassifier()
    cold = brain.generate_reply(
        streamer_speech="Привет, начинаем трансляцию!",
        chat_mood="нейтральное",
        visual_context="Кадр ещё не получен",
    )
    visual = vision.classify_image(
        args.image,
        prompt="Опиши видимые формы и цвета. Есть ли читаемый текст? Не придумывай игру или счёт.",
        max_new_tokens=120,
    )
    answer = brain.answer_stream_question(
        user_question="Что видно на изображении?",
        vlm_description=visual.top_prediction,
    )
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": settings.models.ollama_model,
        "num_ctx": settings.models.ollama_context,
        "image": args.image,
        "scope": "Single synthetic image and text requests; not a real Twitch stream benchmark",
        "initial_text_request": cold.model_dump(),
        "vision_request": visual.model_dump(),
        "question_using_vision": answer.model_dump(),
        "ollama_version": requests.get(f"{base_url}/api/version", timeout=5).json(),
        "resident_models": requests.get(f"{base_url}/api/ps", timeout=5).json(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
