"""Tab 4: Vision-Language Model (VLM & OCR) — настроенной VLM."""

import time
import os
import gradio as gr
from PIL import Image

from src.web.state import web_state
from src.vision_backend import create_vision_classifier
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import record_model_latency


SAMPLE_IMAGE_PATH = os.path.join("samples", "sample_stream.jpg")


def handle_vlm_analyze(image, prompt_text):
    """Processes image with настроенной VLM multimodal model."""
    if image is None:
        return (
            "Изображение не загружено. Пожалуйста, загрузите скриншот или используйте пример.",
            "0 ms",
            "Ожидание",
        )

    clean_prompt = (prompt_text or "").strip()

    try:
        model = web_state.vision_model or create_vision_classifier()
        web_state.vision_model = model
    except Exception as e:
        return f"Ошибка инициализации настроенной VLM: {e}", "0 ms", "Ошибка инициализации"

    start_t = time.perf_counter()
    try:
        res = model.classify_image(image, top_k=5, prompt=clean_prompt)
    except Exception as exc:
        return f"Ошибка анализа изображения: {exc}", "0 ms", model.model_name
    lat_ms = (time.perf_counter() - start_t) * 1000.0

    if hasattr(res, "inference_latency_ms") and res.inference_latency_ms > 0:
        lat_ms = res.inference_latency_ms

    safe_telemetry_dispatch(record_model_latency, "image_vlm", lat_ms / 1000.0)

    caption = (
        res.top_prediction
        if hasattr(res, "top_prediction")
        else (res.get("top_prediction", "") if isinstance(res, dict) else str(res))
    )
    latency_str = f"{lat_ms / 1000.0:.2f} с ({lat_ms:.0f} мс)"
    model_status = model.model_name

    return caption, latency_str, model_status


def load_sample_stream():
    """Loads sample stream screenshot for quick demonstration."""
    if os.path.exists(SAMPLE_IMAGE_PATH):
        try:
            return Image.open(SAMPLE_IMAGE_PATH)
        except Exception:
            return None
    return None


def create_vlm_tab():
    """Builds Tab 4: Vision-Language Model (VLM & OCR) demonstration."""
    with gr.TabItem("4. Зрение (VLM & OCR)"):
        gr.Markdown("### 4. Мультимодальный визуально-языковой анализ (VLM & OCR)")
        gr.Markdown(
            "**Архитектура:** общая мультимодальная модель для изображения и диалога.\n\n"
            "**Стек по умолчанию:** локальный Ollama, Qwen3.5 9B. Задержка измеряется при запросе.\n\n"
            "**Возможности:** семантический разбор кадра, чтение HUD (HP, патроны, радар, таймер), счет матча и русский OCR игрового интерфейса."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(
                    sources=["upload", "webcam"],
                    label="Изображение для VLM",
                    type="pil",
                )
                btn_sample = gr.Button(
                    "Загрузить тестовый скриншот стрима (sample_stream.jpg)",
                    variant="secondary",
                    size="sm",
                )

                prompt_in = gr.Textbox(
                    label="Промпт / Вопрос к кадру (русский язык)",
                    value="",
                    lines=2,
                )

                btn_run = gr.Button("Проанализировать кадр", variant="primary", size="lg")

            with gr.Column(scale=1):
                out_analysis = gr.Textbox(
                    label="Семантический разбор кадра / Распознанный текст (OCR)",
                    lines=8,
                    placeholder="Здесь появляется описание кадра, элементов интерфейса и распознанный текст",
                )
                with gr.Row():
                    out_latency = gr.Textbox(label="Время инференса (CPU Latency)", scale=1)
                    out_model = gr.Textbox(label="Модель и архитектура", scale=2)

                gr.Markdown(
                    "**Особенности архитектуры:**\n"
                    "- Поддержка произвольного соотношения сторон с динамическим разрешением;\n"
                    "- Прямая обработка визуальных токенов через Vision Transformer (ViT) и кросс-внимание;\n"
                    "- Высокая точность оптического распознавания (OCR) русскоязычного и англоязычного текста интерфейса."
                )

        # Load an example image
        btn_sample.click(load_sample_stream, inputs=[], outputs=[img_in])

        # Wire run button
        btn_run.click(
            handle_vlm_analyze,
            inputs=[img_in, prompt_in],
            outputs=[out_analysis, out_latency, out_model],
        )
