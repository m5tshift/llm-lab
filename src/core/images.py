"""Image input normalization shared by vision models."""

import os
from typing import Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Image.Image, np.ndarray]


def load_pil_image(image_input: ImageInput) -> Image.Image:
    """Converts a file path, an OpenCV BGR array, or a PIL image to an RGB PIL image."""
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        if image_input.shape[2] == 3:
            return Image.fromarray(image_input[:, :, ::-1])  # OpenCV frames are BGR.
        return Image.fromarray(image_input).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image_input)}")
