"""Task 3: Image Processing - zero-shot stream scene classification.

Framework: Hugging Face (transformers) / PyTorch
Task: Zero-shot image classification into scene types defined by text prompts
Default model: openai/clip-vit-base-patch32
"""

import time
from typing import Dict, List, Optional

import torch

from src.core.config import settings
from src.core.images import ImageInput, load_pil_image
from src.core.interfaces import BaseImageClassifier
from src.core.schemas import ClassificationItem, ImageClassificationResult

# CLIP was trained on English captions, so the prompts are English. Several prompts per class
# are averaged in embedding space (prompt ensembling); the class set is the evaluation label set.
SCENE_PROMPTS: Dict[str, List[str]] = {
    "gameplay": [
        "a screenshot of video game gameplay",
        "a first-person view in a video game",
        "a third-person view of a character in a video game",
    ],
    "menu": [
        "a video game menu screen",
        "a game settings menu with buttons and options",
        "a game lobby or inventory screen",
    ],
    "loading": [
        "a video game loading screen",
        "a loading screen with a progress bar",
    ],
    "streamer_camera": [
        "a webcam video of a person talking to the camera",
        "a streamer sitting in front of a microphone",
    ],
    "intermission": [
        "a stream starting soon screen",
        "a be right back screen with text",
        "a static stream intermission screen",
    ],
}

SCENE_LABELS_RU: Dict[str, str] = {
    "gameplay": "Игровой процесс",
    "menu": "Меню игры",
    "loading": "Загрузка",
    "streamer_camera": "Камера стримера",
    "intermission": "Заставка / перерыв",
}


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _features(output) -> torch.Tensor:
    # transformers 5 returns a model output with projected pooler_output; 4.x returns a tensor.
    return output.pooler_output if hasattr(output, "pooler_output") else output


class SceneClassifier(BaseImageClassifier):
    """Classifies a frame into the scene types of SCENE_PROMPTS with CLIP, without training."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        prompts: Optional[Dict[str, List[str]]] = None,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = (model_name or settings.models.image_model).strip()
        self.device = resolve_device(device)
        self.prompts = prompts or SCENE_PROMPTS
        self.labels = list(self.prompts)

        print(f"[Scene] Loading CLIP model '{self.model_name}' on {self.device}...")
        start = time.perf_counter()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self.class_embeddings = self._encode_prompts()
        self.load_time = time.perf_counter() - start
        print(f"[Scene] Model loaded in {self.load_time:.2f}s ({len(self.labels)} classes)")

    def _encode_prompts(self) -> torch.Tensor:
        """Returns one L2-normalized embedding per class: the mean of its prompt embeddings."""
        rows = []
        with torch.inference_mode():
            for label in self.labels:
                inputs = self.processor(
                    text=self.prompts[label], return_tensors="pt", padding=True
                ).to(self.device)
                emb = _features(self.model.get_text_features(**inputs))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                rows.append(mean / mean.norm())
        return torch.stack(rows)

    def predict_proba(self, image_input: ImageInput) -> Dict[str, float]:
        """Softmax over the defined classes; the probabilities sum to 1 across this class set."""
        image = load_pil_image(image_input)
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            img = _features(self.model.get_image_features(**inputs))
            img = img / img.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * img @ self.class_embeddings.T
            probs = logits.softmax(dim=-1)[0].tolist()
        return dict(zip(self.labels, probs))

    def classify_image(
        self, image_input: ImageInput, top_k: int = 5, prompt: Optional[str] = None
    ) -> ImageClassificationResult:
        start = time.perf_counter()
        probs = self.predict_proba(image_input)
        latency_ms = (time.perf_counter() - start) * 1000
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        items = [
            ClassificationItem(
                rank=i + 1,
                label=label,
                probability=round(p, 4),
                percentage=f"{p * 100:.1f}%",
            )
            for i, (label, p) in enumerate(ranked)
        ]
        return ImageClassificationResult(
            top_prediction=items[0].label,
            confidence=items[0].probability,
            predictions=items,
            inference_latency_ms=round(latency_ms, 2),
        )


if __name__ == "__main__":
    import os

    classifier = SceneClassifier()
    sample = "samples/sample_stream.jpg"
    if os.path.exists(sample):
        res = classifier.classify_image(sample)
        print(f"[Test] {SCENE_LABELS_RU[res.top_prediction]} ({res.confidence:.2f})")
        print(f"[Test] Latency: {res.inference_latency_ms} ms")
