"""Main Pipeline: Multimodal Stream Observer & Analyzer.

Integrates the five models of the report:
- Text: RuBERT Tiny2 CEDR emotions -> chat mood (Hugging Face Transformers)
- Audio: Whisper Small speech recognition (Hugging Face Transformers)
- Image: CLIP zero-shot scene classification (Hugging Face Transformers)
- Video: YOLO26s object detection (Ultralytics)
- LLM: local Ollama model; its frame description arrives as vlm_description
"""

import time
import os
from typing import Any, List, Optional

from src.core.schemas import LLMReplyResult, MultimodalCycleResult
from src.nlp_sentiment import ChatSentimentAnalyzer
from src.audio_transcribe import StreamAudioTranscriber
from src.scene_classify import SCENE_LABELS_RU, SceneClassifier
from src.video_live import LiveStreamVideoDetector
from src.llm_brain import LocalLLMBrain


class StreamMultimodalPipeline:
    """End-to-End Multimodal Orchestrator for Stream Monitoring with Dependency Injection."""

    def __init__(
        self,
        text_analyzer: Optional[Any] = None,
        audio_transcriber: Optional[Any] = None,
        image_classifier: Optional[Any] = None,
        video_detector: Optional[Any] = None,
        llm_brain: Optional[Any] = None,
    ):
        print("=" * 60)
        print("Initializing Multimodal Stream Pipeline...")
        print("=" * 60)
        self.text_analyzer = text_analyzer or ChatSentimentAnalyzer()
        self.audio_transcriber = audio_transcriber or StreamAudioTranscriber()
        self.image_classifier = image_classifier or SceneClassifier()
        self.video_detector = video_detector or LiveStreamVideoDetector()
        self.llm_brain = llm_brain or LocalLLMBrain()
        print("\n[Bot] All 5 AI/ML systems initialized successfully!")

    def process_stream_cycle(
        self,
        audio_file_path: str,
        image_or_frame,
        chat_messages: List[str],
        vlm_description: str = "",
    ) -> MultimodalCycleResult:
        """Executes one observation cycle across audio, text, video, image and the LLM."""
        cycle_start = time.perf_counter()
        print("\n--- [CYCLE START] Gathering Multimodal Stream Data ---")

        print("[1/5] Transcribing Streamer Audio...")
        try:
            streamer_speech = self.audio_transcriber.transcribe_audio_file(
                audio_file_path
            ).transcription
        except Exception as e:
            streamer_speech = f"(Audio Error: {e})"
        print(f' -> Streamer Said: "{streamer_speech}"')

        print("[2/5] Analyzing Twitch Chat Sentiment...")
        chat_res = self.text_analyzer.analyze_chat_batch(chat_messages)
        chat_mood = (
            f"{chat_res.dominant_mood} (Pos: {chat_res.positive_ratio * 100:.0f}%, "
            f"Neg: {chat_res.negative_ratio * 100:.0f}%)"
        )
        print(f" -> Chat Mood: {chat_mood}")

        print("[3/5] Running Object Detection on Frame...")
        vid_res = self.video_detector.process_frame(image_or_frame)
        detected_labels = [o.class_name for o in vid_res.detected_objects]
        obj_summary = ", ".join(sorted(set(detected_labels))) or "no distinct objects"
        print(f" -> Detected in stream: {obj_summary} ({vid_res.latency_ms}ms)")

        print("[4/5] Classifying Stream Scene...")
        img_res = self.image_classifier.classify_image(image_or_frame)
        scene_type = (
            f"{SCENE_LABELS_RU.get(img_res.top_prediction, img_res.top_prediction)} "
            f"({img_res.confidence * 100:.0f}%)"
        )
        print(f" -> Scene: {scene_type}")

        print("[5/5] Generating Twitch Bot Response via Local LLM...")
        try:
            llm_res = self.llm_brain.generate_reply(
                streamer_speech=streamer_speech,
                chat_mood=chat_mood,
                visual_context=f"Objects visible: {obj_summary}. Scene: {scene_type}",
                recent_chat_messages=chat_messages,
                scene_description=scene_type,
                detected_objects=obj_summary,
                chat_stats={
                    "dominant_mood": chat_res.dominant_mood,
                    "positive_ratio": chat_res.positive_ratio,
                    "negative_ratio": chat_res.negative_ratio,
                    "total_messages": chat_res.total_messages,
                },
                vlm_description=vlm_description,
            )
        except Exception as e:
            # Same convention as the audio step: the cycle completes and the error is shown.
            llm_res = LLMReplyResult(
                reply=f"(LLM Error: {e.__class__.__name__})", engine="unavailable", latency_ms=0.0
            )
        print(f' -> Bot Chat Reply: "{llm_res.reply}" (via {llm_res.engine})')

        total_latency_ms = (time.perf_counter() - cycle_start) * 1000
        self._record_telemetry(
            total_latency_ms=total_latency_ms,
            pos_ratio=chat_res.positive_ratio,
            neg_ratio=chat_res.negative_ratio,
            engine=llm_res.engine,
            detected_labels=detected_labels,
            vid_latency=vid_res.latency_ms,
            img_latency=img_res.inference_latency_ms,
            chat_latency=chat_res.latency_ms,
            llm_latency=llm_res.latency_ms,
        )

        return MultimodalCycleResult(
            streamer_speech=streamer_speech,
            chat_mood=chat_mood,
            visual_objects=detected_labels,
            scene_classification=scene_type,
            detailed_results={"vlm_description": vlm_description} if vlm_description else None,
            bot_chat_reply=llm_res.reply,
            llm_engine=llm_res.engine,
            cycle_total_latency_ms=round(total_latency_ms, 2),
        )

    def _record_telemetry(
        self,
        total_latency_ms,
        pos_ratio,
        neg_ratio,
        engine,
        detected_labels,
        vid_latency,
        img_latency,
        chat_latency,
        llm_latency,
    ):
        from src.metrics import (
            record_ai_update,
            record_chat_stats,
            record_bot_reply,
            record_detected_objects,
            record_model_latency,
        )
        from src.monitoring.decorators import safe_telemetry_dispatch as emit

        emit(record_ai_update, total_latency_ms / 1000, mode="pipeline")
        emit(record_chat_stats, pos_ratio, neg_ratio)
        emit(record_bot_reply, engine)
        emit(record_detected_objects, detected_labels)
        for model, latency in (
            ("video_yolo", vid_latency),
            ("image_scene", img_latency),
            ("nlp_sentiment", chat_latency),
            ("llm_brain", llm_latency),
        ):
            if latency > 0:  # Cached results are not new inference operations.
                emit(record_model_latency, model, latency / 1000)


def run_demo():
    """Run an automated end-to-end demo verifying all modules."""
    import numpy as np
    import wave

    demo_wav = "demo_speech.wav"
    sample_rate = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    data = (np.sin(2 * np.pi * 300 * t) * 16384).astype(np.int16)
    with wave.open(demo_wav, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    chat_sample = [
        "THIS GUY IS CRACKED PogChamp",
        "clip it clip it right now!!",
        "W streamer, so good!",
        "LMAO what was that play",
    ]

    try:
        pipeline = StreamMultimodalPipeline()
        result = pipeline.process_stream_cycle(
            audio_file_path=demo_wav, image_or_frame=frame, chat_messages=chat_sample
        )
        print("\n" + "=" * 60)
        print("DEMO EXECUTION SUMMARY:")
        print(f"Total Cycle Latency: {result.cycle_total_latency_ms} ms")
        print(f"Final Bot Output: {result.bot_chat_reply}")
        print("=" * 60)
    finally:
        if os.path.exists(demo_wav):
            os.remove(demo_wav)


if __name__ == "__main__":
    run_demo()
