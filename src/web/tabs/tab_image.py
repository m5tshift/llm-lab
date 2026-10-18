"""Tab 3: Vision-Language Model (VLM) & Image-to-Text Scene Understanding."""

import gradio as gr
from src.web.state import web_state
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import record_model_latency


def handle_image_describe(image, model_choice, prompt_text):
    """Processes image via selected VLM or Captioning model with custom visual prompt."""
    if image is None:
        return "Изображение не загружено", "0 ms", "Нет данных"

    try:
        model = web_state.get_image_classifier(model_choice)
    except Exception as e:
        return f"Ошибка инициализации модели {model_choice}: {e}", "0 ms", "Ошибка"

    try:
        res = model.classify_image(image, top_k=5, prompt=prompt_text)
    except Exception as exc:
        return f"Ошибка описания изображения: {exc}", "0 ms", model.model_name
    lat_ms = (
        res.inference_latency_ms
        if hasattr(res, "inference_latency_ms")
        else res.get("inference_latency_ms", 50)
    )
    safe_telemetry_dispatch(record_model_latency, "image_classify", lat_ms / 1000.0)

    caption = (
        res.top_prediction if hasattr(res, "top_prediction") else res.get("top_prediction", "")
    )
    latency_str = f"{lat_ms:.1f} ms"
    arch_type = (
        "VLM (Vision-Language)" if getattr(model, "is_vlm", False) else "Image-to-Text Captioning"
    )
    model_info = f"{model.model_name} [{arch_type}]"

    return caption, latency_str, model_info


# Backward-compatible handler alias
def handle_image_classify(image):
    caption, _, _ = handle_image_describe(
        image,
        "Qwen/Qwen2-VL-2B-Instruct",
        "",
    )
    return {caption: 1.0}


def create_image_tab():
    with gr.TabItem("3. Сравнение: BLIP Large"):
        gr.Markdown("### 3. Описание изображения (Image-to-Text Captioning)")
        gr.Markdown(
            "**Архитектура:** `Salesforce/blip-image-captioning-large` (446M параметров, ViT-L/14 + Text Decoder)\n\n"
            "**Стек:** Hugging Face Transformers, PyTorch | **Среда:** CPU (Docker) | **Задержка:** ~2.40 с | **Метрика:** CIDEr 144.5\n\n"
            "Экспериментальная модель предыдущего этапа. Загружается только при ручном запросе; текущий живой режим использует общую модель Ollama."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(sources=["upload", "webcam"], label="Изображение", type="pil")
                prompt_in = gr.Textbox(
                    label="Направляющий промпт (опционально для условной генерации)",
                    value="",
                    lines=2,
                )
                btn_img = gr.Button(
                    "Сгенерировать описание сцены (BLIP Large)", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                out_caption = gr.Textbox(
                    label="Сгенерированное описание сцены (Scene Caption)", lines=6
                )
                with gr.Row():
                    out_latency = gr.Textbox(label="Время инференса (Latency)", scale=1)
                    out_model = gr.Textbox(label="Активная архитектура и модель", scale=2)

        def _handle_blip(image, prompt):
            return handle_image_describe(image, "Salesforce/blip-image-captioning-large", prompt)

        btn_img.click(
            _handle_blip, inputs=[img_in, prompt_in], outputs=[out_caption, out_latency, out_model]
        )
