"""Manual VLM selection must not replace the live stream's BLIP model."""

from unittest.mock import patch

from src.web.state import WebState


def test_vlm_cache_does_not_change_live_scene_model():
    state = WebState()
    blip = object()
    state.image_model = blip
    with patch("src.web.state.StreamImageClassifier", side_effect=lambda model_name: model_name):
        vlm = state.get_image_classifier("Qwen/Qwen2-VL-2B-Instruct")
        assert vlm == "Qwen/Qwen2-VL-2B-Instruct"
        assert state.get_image_classifier("Qwen/Qwen2-VL-2B-Instruct") == vlm
    assert state.image_model is blip
