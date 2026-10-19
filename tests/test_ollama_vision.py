"""Verify local vision transport and compatibility with cached live analysis."""

import base64
import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.core.config import settings
from src.vision_backend import OllamaVisionClassifier


@patch("requests.post")
def test_vision_sends_rgb_image_to_same_model_as_text(post):
    post.return_value.status_code = 200
    post.return_value.json.return_value = {"response": "Красный прямоугольник"}
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # OpenCV BGR red.
    result = OllamaVisionClassifier().classify_image(frame, prompt="Что видно?")
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == settings.models.ollama_model
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == settings.models.ollama_context
    image = Image.open(io.BytesIO(base64.b64decode(payload["images"][0])))
    red, green, blue = image.getpixel((0, 0))
    assert red > 240 and green < 10 and blue < 10
    assert result.top_prediction == "Красный прямоугольник"
    assert result.confidence == 0  # No invented confidence score.


@patch("requests.post")
def test_vision_propagates_ollama_error(post):
    post.return_value.status_code = 500
    post.return_value.text = "model not loaded"
    with pytest.raises(RuntimeError, match="model not loaded"):
        OllamaVisionClassifier().classify_image(Image.new("RGB", (8, 8)))


@pytest.mark.parametrize("prompt", [None, "", "   "])
@patch("requests.post")
def test_empty_vision_prompt_has_no_hidden_instructions(post, prompt):
    post.return_value.status_code = 200
    post.return_value.json.return_value = {"message": {"content": "Красный квадрат"}}
    result = OllamaVisionClassifier().classify_image(Image.new("RGB", (8, 8)), prompt=prompt)
    assert post.call_args.args[0].endswith("/api/chat")
    payload = post.call_args.kwargs["json"]
    assert "prompt" not in payload
    assert payload["messages"][0]["content"] == ""
    assert payload["messages"][0]["images"]
    assert result.top_prediction == "Красный квадрат"
