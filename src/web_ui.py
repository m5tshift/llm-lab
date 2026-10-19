"""Gradio Web UI Facade & Entrypoint.

Preserves backward compatibility for container commands (python -m src.web_ui)
and existing imports while delegating implementation to modular src.web package.
"""

from src.web.app import demo, create_web_app, launch
from src.web.state import web_state
from src.web.styles import make_chat_status_html
from src.web.tabs.tab_text import handle_sentiment_single, handle_sentiment_batch
from src.web.tabs.tab_audio import handle_audio_transcribe
from src.web.tabs.tab_image import handle_scene_classify
from src.web.tabs.tab_video import handle_frame_detect
from src.web.tabs.tab_llm import handle_llm_reaction
from src.web.tabs.tab_twitch import handle_stream_question, stream_live_twitch

__all__ = [
    "demo",
    "create_web_app",
    "launch",
    "web_state",
    "make_chat_status_html",
    "handle_sentiment_single",
    "handle_sentiment_batch",
    "handle_audio_transcribe",
    "handle_scene_classify",
    "handle_frame_detect",
    "handle_llm_reaction",
    "handle_stream_question",
    "stream_live_twitch",
]


if __name__ == "__main__":
    launch()
