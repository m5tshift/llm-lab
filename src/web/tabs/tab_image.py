"""Tab 3: Zero-shot stream scene classification with CLIP."""

import gradio as gr

from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import record_model_latency
from src.scene_classify import SCENE_LABELS_RU, SCENE_PROMPTS
from src.web.state import web_state


def handle_scene_classify(image):
    """Returns class probabilities for the Label component, latency and the model name."""
    if image is None:
        return {}, "0 ms", "Изображение не загружено"
    if not web_state.scene_model:
        return {}, "0 ms", "Модель CLIP не загружена"
    try:
        res = web_state.scene_model.classify_image(image)
    except Exception as exc:
        return {}, "0 ms", f"Ошибка классификации: {exc}"
    safe_telemetry_dispatch(record_model_latency, "image_scene", res.inference_latency_ms / 1000.0)
    probs = {SCENE_LABELS_RU.get(i.label, i.label): i.probability for i in res.predictions}
    return probs, f"{res.inference_latency_ms:.1f} ms", web_state.scene_model.model_name


def create_image_tab():
    classes = "\n".join(
        f"- **{SCENE_LABELS_RU[label]}** (`{label}`): " + "; ".join(f"_{p}_" for p in prompts)
        for label, prompts in SCENE_PROMPTS.items()
    )
    with gr.TabItem("3. Изображение (CLIP)"):
        gr.Markdown("### 3. Zero-shot классификация сцены кадра")
        gr.Markdown(
            "**Модель:** `openai/clip-vit-base-patch32`, Hugging Face Transformers. "
            "Классы заданы текстовыми описаниями без обучения; вероятности распределены "
            "только между этими классами.\n\n" + classes
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(sources=["upload", "webcam"], label="Кадр", type="pil")
                btn = gr.Button("Классифицировать сцену", variant="primary")
            with gr.Column(scale=1):
                out_label = gr.Label(label="Тип сцены", num_top_classes=len(SCENE_PROMPTS))
                with gr.Row():
                    out_latency = gr.Textbox(label="Время инференса", scale=1)
                    out_model = gr.Textbox(label="Модель", scale=2)

        btn.click(
            handle_scene_classify, inputs=[img_in], outputs=[out_label, out_latency, out_model]
        )
