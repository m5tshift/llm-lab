"""Standalone tabs must stay blank; Twitch must disclose its live instructions."""

import gradio as gr

from src.live_vision import LIVE_VLM_PROMPT
from src.prompts import LIVE_BOT_PERSONALITY, LIVE_REACTION_PROMPT, LIVE_QUESTION_PROMPT
from src.web.tabs.tab_image import create_image_tab
from src.web.tabs.tab_llm import create_llm_tab
from src.web.tabs.tab_twitch import create_twitch_tab
from src.web.tabs.tab_vlm import create_vlm_tab


def test_standalone_tabs_have_no_prefilled_text():
    with gr.Blocks() as app:
        with gr.Tabs():
            create_image_tab()
            create_vlm_tab()
            create_llm_tab()
    for component in app.blocks.values():
        if isinstance(component, gr.Textbox):
            assert not component.value


def test_twitch_displays_actual_live_templates():
    with gr.Blocks() as app:
        with gr.Tabs():
            create_twitch_tab()
    values = [
        component.value
        for component in app.blocks.values()
        if isinstance(component, gr.Textbox) and component.interactive is False
    ]
    assert LIVE_VLM_PROMPT in values
    assert LIVE_BOT_PERSONALITY in values
    assert LIVE_REACTION_PROMPT in values
    assert any(value.startswith(LIVE_QUESTION_PROMPT) for value in values if value)
