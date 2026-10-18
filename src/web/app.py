"""Gradio Application Orchestrator."""

import gradio as gr

from src.core.config import settings
from src.metrics import start_metrics_server
from src.web.state import web_state
from src.web.styles import CUSTOM_CSS
from src.web.tabs.tab_text import create_text_tab
from src.web.tabs.tab_audio import create_audio_tab
from src.web.tabs.tab_image import create_image_tab
from src.web.tabs.tab_vlm import create_vlm_tab
from src.web.tabs.tab_video import create_video_tab
from src.web.tabs.tab_llm import create_llm_tab
from src.web.tabs.tab_twitch import create_twitch_tab


def create_web_app() -> gr.Blocks:
    start_metrics_server(port=settings.network.metrics_port)
    web_state.preload_models()

    with gr.Blocks(title="llm-lab", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div style="text-align: center; margin: 12px auto 20px auto; max-width: 960px;">
                <h1 style="font-size: 2.1rem; font-weight: 800; color: #f8fafc; margin-bottom: 8px; letter-spacing: -0.02em;">
                    llm-lab
                </h1>
            </div>
            """
        )
        gr.Markdown(
            "Изучение возможностей готовых моделей: эмоции текста, распознавание речи, "
            "описание изображений, вопросы о кадре, детекция объектов и диалог с LLM. "
            "Дополнительно доступна их совместная работа для реакций и ответов о происходящем."
        )
        with gr.Tabs():
            create_text_tab()
            create_audio_tab()
            create_image_tab()
            create_vlm_tab()
            create_video_tab()
            create_llm_tab()
            create_twitch_tab()

    return demo


demo = create_web_app()


def launch():
    host = settings.network.gradio_host
    port = settings.network.gradio_port
    print(f"[Gradio UI] Starting local web studio at http://{host}:{port} ...")
    try:
        demo.launch(server_name=host, server_port=port, css=CUSTOM_CSS, share=False)
    except TypeError:
        demo.launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    launch()
