"""CLIP scene classification and track counting, with stand-in models instead of real weights."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.scene_classify import SceneClassifier
from src.video_live import LiveStreamVideoDetector

PROMPTS = {"gameplay": ["video game", "game view"], "menu": ["game menu"]}


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    def __call__(self, text=None, images=None, **_):
        return FakeInputs(texts=text) if text is not None else FakeInputs(image=images)


class FakeClip:
    logit_scale = torch.tensor(math.log(100.0))

    def to(self, _device):
        return self

    def eval(self):
        return self

    def get_text_features(self, texts):
        # "menu" prompts point along axis 1, all other prompts along axis 0.
        return torch.tensor([[0.0, 1.0] if "menu" in t else [1.0, 0.0] for t in texts])

    def get_image_features(self, image):
        return torch.tensor([[0.9, 0.1]])


def make_classifier():
    with (
        patch("transformers.CLIPProcessor.from_pretrained", return_value=FakeProcessor()),
        patch("transformers.CLIPModel.from_pretrained", return_value=FakeClip()),
    ):
        return SceneClassifier("fake-clip", device="cpu", prompts=PROMPTS)


def test_scene_probabilities_cover_only_defined_classes():
    clf = make_classifier()
    probs = clf.predict_proba(np.zeros((4, 4, 3), dtype=np.uint8))
    assert set(probs) == {"gameplay", "menu"}
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-6)
    res = clf.classify_image(np.zeros((4, 4, 3), dtype=np.uint8))
    assert res.top_prediction == "gameplay"
    assert res.predictions[0].probability > res.predictions[1].probability


def _results(classes, ids):
    boxes = [
        SimpleNamespace(
            cls=torch.tensor([c]), conf=torch.tensor([0.9]), xyxy=torch.tensor([[0.0, 0, 10, 10]])
        )
        for c in classes
    ]

    class Boxes(list):
        id = torch.tensor(ids) if ids is not None else None

    return SimpleNamespace(boxes=Boxes(boxes), plot=lambda: None)


def test_unique_objects_count_distinct_track_ids_until_reset():
    detector = LiveStreamVideoDetector.__new__(LiveStreamVideoDetector)
    detector.model = SimpleNamespace(names={0: "person", 2: "car"})
    detector._track_model = None
    from collections import defaultdict

    detector._seen_tracks = defaultdict(set)
    detector._to_result(_results([0, 0], [1, 2]), 5.0, annotate=False)
    res = detector._to_result(_results([0, 2], [2, 7]), 5.0, annotate=False)
    assert res.unique_objects == {"car": 1, "person": 2}
    assert [o.track_id for o in res.detected_objects] == [2, 7]

    detector.reset_tracking()
    assert detector.unique_objects() == {}
    plain = detector._to_result(_results([0], None), 5.0, annotate=False)
    assert plain.detected_objects[0].track_id is None and plain.unique_objects == {}
