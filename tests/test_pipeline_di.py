"""Unit tests for multimodal pipeline with Dependency Injection and Mock Models."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.bot_pipeline import StreamMultimodalPipeline
from src.core.schemas import (
    BatchSentimentResult,
    AudioTranscriptionResult,
    ImageClassificationResult,
    ClassificationItem,
    VideoDetectionResult,
    DetectedObject,
    LLMReplyResult,
)


class TestPipelineDependencyInjection(unittest.TestCase):
    def setUp(self):
        # 1. Mock Text Analyzer
        self.mock_text = MagicMock()
        self.mock_text.analyze_chat_batch.return_value = BatchSentimentResult(
            dominant_mood="ХАЙП / ПОЗИТИВ",
            positive_ratio=0.8,
            negative_ratio=0.1,
            total_messages=10,
            latency_ms=5.0,
        )

        # 2. Mock Audio Transcriber
        self.mock_audio = MagicMock()
        self.mock_audio.transcribe_audio_file.return_value = AudioTranscriptionResult(
            transcription="Заходим на точку Б!", language="russian", inference_time_s=0.05
        )

        # 3. Mock Image Classifier
        self.mock_image = MagicMock()
        self.mock_image.classify_image.return_value = ImageClassificationResult(
            top_prediction="video game screen",
            confidence=0.92,
            predictions=[
                ClassificationItem(
                    rank=1, label="video game screen", probability=0.92, percentage="92.0%"
                )
            ],
            inference_latency_ms=10.0,
        )

        # 4. Mock Video Detector
        self.mock_video = MagicMock()
        self.mock_video.process_frame.return_value = VideoDetectionResult(
            detected_objects=[
                DetectedObject(class_name="person", confidence=0.88, box=[10, 10, 100, 100])
            ],
            annotated_frame=np.zeros((100, 100, 3), dtype=np.uint8),
            latency_ms=8.0,
        )

        # 5. Mock LLM Brain
        self.mock_llm = MagicMock()
        self.mock_llm.generate_reply.return_value = LLMReplyResult(
            reply="Отличный план, заходим аккуратно!", engine="Mock LLM", latency_ms=15.0
        )

        self.pipeline = StreamMultimodalPipeline(
            text_analyzer=self.mock_text,
            audio_transcriber=self.mock_audio,
            image_classifier=self.mock_image,
            video_detector=self.mock_video,
            llm_brain=self.mock_llm,
        )

    def test_end_to_end_cycle(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        chat = ["Вперед команда!", "Отличная игра!"]

        result = self.pipeline.process_stream_cycle(
            audio_file_path="dummy.wav", image_or_frame=frame, chat_messages=chat
        )

        self.assertEqual(result.streamer_speech, "Заходим на точку Б!")
        self.assertEqual(result.bot_chat_reply, "Отличный план, заходим аккуратно!")
        self.assertEqual(result.llm_engine, "Mock LLM")
        self.assertIn("person", result.visual_objects)
        self.assertTrue(result.cycle_total_latency_ms > 0)

    def test_cached_vlm_description_reaches_llm(self):
        result = self.pipeline.process_stream_cycle(
            audio_file_path="dummy.wav",
            image_or_frame=np.zeros((2, 2, 3), dtype=np.uint8),
            chat_messages=["Гол!"],
            vlm_description="На табло 2:1",
        )
        self.assertEqual(
            self.mock_llm.generate_reply.call_args.kwargs["vlm_description"],
            "На табло 2:1",
        )
        self.assertEqual(result.detailed_results["vlm_description"], "На табло 2:1")

    def test_live_cycle_reuses_vision_without_running_another_image_request(self):
        result = self.pipeline.process_stream_cycle(
            audio_file_path="dummy.wav",
            image_or_frame=np.zeros((2, 2, 3), dtype=np.uint8),
            chat_messages=[],
            vlm_description="Синий прямоугольник",
            use_cached_vision=True,
        )
        self.mock_image.classify_image.assert_not_called()
        self.assertEqual(result.scene_classification, "Синий прямоугольник")


if __name__ == "__main__":
    unittest.main()
