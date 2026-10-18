"""Module 3: Image Processing - Stream Scene & Screenshot Image-to-Text Captioning.

Framework: Hugging Face (transformers) / PyTorch
Task: Image-to-Text (Image Captioning & Scene Description)
Model: Salesforce/blip-image-captioning-large (with optional BLIP Base / ResNet-50 fallback)
Architectural decision: Progressed from ResNet-50 (rigid ImageNet-1K classes) to BLIP Base,
and upgraded to BLIP Large (ViT-L/14, 446M params) for richer semantic scene details.
"""

import time
import os
from typing import Optional, Union
from PIL import Image
import numpy as np
import torch

from src.core.config import settings
from src.core.schemas import ImageClassificationResult, ClassificationItem
from src.core.interfaces import BaseImageClassifier


class StreamImageClassifier(BaseImageClassifier):
    """
    Image-to-Text scene describer for stream screenshots.
    Generates natural language captions using Vision-Language Transformers (BLIP).
    Supports backward-compatible ImageNet classification if CNN model is explicitly specified.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = (model_name or settings.models.image_model).strip()
        print(
            f"[Image] Loading Image-to-Text vision model: '{self.model_name}' on {self.device}..."
        )
        start = time.perf_counter()

        self.is_vlm = any(
            k in self.model_name.lower() for k in ["qwen2-vl", "qwen-vl", "vlm", "florence"]
        )
        self.is_caption_model = not self.is_vlm and (
            any(
                k in self.model_name.lower()
                for k in ["blip", "caption", "git-", "vit-gpt2", "image-to-text"]
            )
            or (
                "resnet" not in self.model_name.lower()
                and "mobilenet" not in self.model_name.lower()
                and "vit_b" not in self.model_name.lower()
            )
        )

        if self.is_vlm:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            self.process_vision_info = process_vision_info
            min_pixels = 256 * 28 * 28
            max_pixels = 512 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, min_pixels=min_pixels, max_pixels=max_pixels
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device.type == "cuda" else "cpu",
                low_cpu_mem_usage=True,
            )
            if self.device.type != "cuda":
                self.model.to(self.device)
            self.model.eval()
        elif self.is_caption_model:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(
                self.device
            )
            self.model.eval()
        else:
            # Legacy CNN classification fallback (ResNet / MobileNet)
            import torchvision.models as models

            if "mobilenet" in self.model_name.lower():
                self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
                self.model = models.mobilenet_v3_small(weights=self.weights).to(self.device)
            elif "vit" in self.model_name.lower():
                self.weights = models.ViT_B_16_Weights.DEFAULT
                self.model = models.vit_b_16(weights=self.weights).to(self.device)
            else:
                self.weights = models.ResNet50_Weights.DEFAULT
                self.model = models.resnet50(weights=self.weights).to(self.device)

            self.model.eval()
            self.transform = self.weights.transforms()
            self.categories = self.weights.meta["categories"]

        self.load_time = time.perf_counter() - start
        mode_str = "VLM" if self.is_vlm else ("Caption" if self.is_caption_model else "CNN")
        print(
            f"[Image] Model loaded successfully in {self.load_time:.2f}s (Architecture mode: {mode_str})"
        )

    def describe_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        max_new_tokens: int = 40,
    ) -> str:
        """Generates a natural language sentence describing the scene in the image."""
        img = self._load_pil_image(image_input)
        prompt = (prompt or "").strip()

        if self.is_vlm:
            prompt_text = (prompt or "").strip()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        *([{"type": "text", "text": prompt_text}] if prompt_text else []),
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
            return caption
        elif self.is_caption_model:
            if prompt:
                inputs = self.processor(img, prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return caption
        else:
            # Fallback using top ImageNet category
            res = self.classify_image(img, top_k=1)
            return res.top_prediction

    def classify_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 5,
        prompt: Optional[str] = None,
        max_new_tokens: int = 40,
    ) -> ImageClassificationResult:
        """
        Processes image and returns structured ImageClassificationResult.
        In VLM & Image-to-Text mode: returns the natural language scene analysis as top_prediction.
        """
        img = self._load_pil_image(image_input)
        start = time.perf_counter()

        if self.is_vlm or self.is_caption_model:
            caption = self.describe_image(img, prompt=prompt, max_new_tokens=max_new_tokens)
            latency_ms = (time.perf_counter() - start) * 1000
            # Legacy placeholder scores, not probabilities of caption correctness.
            conf = 0.98 if self.is_vlm else 0.96

            prediction_item = ClassificationItem(
                rank=1, label=caption, probability=conf, percentage=f"{conf * 100:.1f}%"
            )
            return ImageClassificationResult(
                top_prediction=caption,
                confidence=conf,
                predictions=[prediction_item],
                inference_latency_ms=round(latency_ms, 2),
            )
        else:
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            top_prob, top_indices = torch.topk(probabilities, top_k)
            latency_ms = (time.perf_counter() - start) * 1000

            items = [
                ClassificationItem(
                    rank=i + 1,
                    label=self.categories[idx],
                    probability=round(float(prob), 4),
                    percentage=f"{float(prob) * 100:.1f}%",
                )
                for i, (prob, idx) in enumerate(zip(top_prob, top_indices))
            ]

            return ImageClassificationResult(
                top_prediction=items[0].label,
                confidence=items[0].probability,
                predictions=items,
                inference_latency_ms=round(latency_ms, 2),
            )

    @staticmethod
    def _load_pil_image(image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Converts filepath, numpy array, or existing PIL Image to RGB PIL Image."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return Image.fromarray(image_input).convert("RGB")
            # If BGR from OpenCV, convert to RGB
            if image_input.shape[2] == 3:
                return Image.fromarray(image_input[:, :, ::-1])
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")


if __name__ == "__main__":
    classifier = StreamImageClassifier()
    test_img = "samples/sample_stream.jpg"
    if os.path.exists(test_img):
        res = classifier.classify_image(test_img, top_k=3)
        print(f"[Test] Top prediction / Caption: {res.top_prediction}")
        print(f"[Test] Latency: {res.inference_latency_ms} ms")
