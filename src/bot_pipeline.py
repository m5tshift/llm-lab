"""Main Pipeline: Multimodal Stream Observer & Analyzer.

Integrates 5 AI/ML systems:
- Audio: OpenAI Whisper Small ASR
- Video: Ultralytics YOLO26s Real-time Detection
- Image: Salesforce BLIP Large Captioning & Qwen2-VL VLM
- Text: Hugging Face RuBERT Tiny2 CEDR Sentiment
- LLM: Local Ollama Qwen 3 8B Brain
"""

import time
import os
from typing import Any, List, Optional

from src.core.schemas import MultimodalCycleResult
from src.nlp_sentiment import ChatSentimentAnalyzer
from src.audio_transcribe import StreamAudioTranscriber
from src.image_classify import StreamImageClassifier
from src.video_live import LiveStreamVideoDetector
from src.llm_brain import LocalLLMBrain
from src.core.config import settings
from src.vision_backend import create_vision_classifier


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

        # 1. NLP Text Model
        self.text_analyzer = text_analyzer or ChatSentimentAnalyzer()

        # 2. Audio Whisper Model
        self.audio_transcriber = audio_transcriber or StreamAudioTranscriber()

        # 3. Image Classifier Model
        self.image_classifier = image_classifier or (
            create_vision_classifier()
            if settings.models.vision_backend == "ollama"
            else StreamImageClassifier()
        )

        # 4. Video Live Detector
        self.video_detector = video_detector or LiveStreamVideoDetector()

        # 5. Local LLM Brain
        self.llm_brain = llm_brain or LocalLLMBrain()

        print("\n[Bot] All 5 AI/ML systems initialized successfully!")

    def process_stream_cycle(
        self,
        audio_file_path: str,
        image_or_frame,
        chat_messages: List[str],
        vlm_description: str = "",
        use_cached_vision: bool = False,
    ) -> MultimodalCycleResult:
        """Executes one full multimodal observation cycle across Audio, Text, Video, Image, and LLM."""
        cycle_start = time.perf_counter()
        print("\n--- [CYCLE START] Gathering Multimodal Stream Data ---")

        # 1. Audio Processing
        print("[1/5] Transcribing Streamer Audio...")
        try:
            audio_res = self.audio_transcriber.transcribe_audio_file(audio_file_path)
            streamer_speech = (
                audio_res["transcription"]
                if isinstance(audio_res, dict)
                else audio_res.transcription
            )
        except Exception as e:
            streamer_speech = f"(Audio Error: {e})"
        print(f' -> Streamer Said: "{streamer_speech}"')

        # 2. Text (Chat) Processing
        print("[2/5] Analyzing Twitch Chat Sentiment...")
        chat_res = self.text_analyzer.analyze_chat_batch(chat_messages)
        dominant_mood = (
            chat_res["dominant_mood"] if isinstance(chat_res, dict) else chat_res.dominant_mood
        )
        pos_ratio = (
            chat_res["positive_ratio"] if isinstance(chat_res, dict) else chat_res.positive_ratio
        )
        neg_ratio = (
            chat_res["negative_ratio"] if isinstance(chat_res, dict) else chat_res.negative_ratio
        )
        chat_mood = f"{dominant_mood} (Pos: {pos_ratio * 100:.0f}%, Neg: {neg_ratio * 100:.0f}%)"
        print(f" -> Chat Mood: {chat_mood}")

        # 3. Video Processing (Object Detection)
        print("[3/5] Running Real-Time Object Detection on Frame...")
        vid_res = self.video_detector.process_frame(image_or_frame)
        raw_objects = (
            vid_res["detected_objects"] if isinstance(vid_res, dict) else vid_res.detected_objects
        )
        vid_latency = vid_res["latency_ms"] if isinstance(vid_res, dict) else vid_res.latency_ms

        detected_labels = []
        for o in raw_objects:
            lbl = (
                o.get("class") or o.get("class_name")
                if isinstance(o, dict)
                else getattr(o, "class_name", "")
            )
            if lbl:
                detected_labels.append(lbl)

        obj_summary = ", ".join(set(detected_labels)) if detected_labels else "no distinct objects"
        print(f" -> Detected in stream: {obj_summary} ({vid_latency}ms)")

        # 4. Image Processing (Scene Classification)
        print("[4/5] Classifying Stream Scene / Screenshot...")
        if use_cached_vision:
            scene_type = vlm_description or "Свежий анализ кадра пока не готов"
            img_latency = 0
        else:
            img_res = self.image_classifier.classify_image(image_or_frame)
            scene_type = (
                img_res["top_prediction"] if isinstance(img_res, dict) else img_res.top_prediction
            )
            img_latency = (
                img_res.get("inference_latency_ms", 0)
                if isinstance(img_res, dict)
                else getattr(img_res, "inference_latency_ms", 0)
            )
        print(f" -> Scene Classification: {scene_type}")

        # 5. Local LLM Synthesis
        print("[5/5] Generating Twitch Bot Response via Local LLM...")
        visual_context = f"Objects visible: {obj_summary}. Scene: {scene_type}"
        chat_stats = {
            "dominant_mood": dominant_mood,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "total_messages": chat_res.get("total_messages", len(chat_messages))
            if isinstance(chat_res, dict)
            else getattr(chat_res, "total_messages", len(chat_messages)),
        }
        llm_res = self.llm_brain.generate_reply(
            streamer_speech=streamer_speech,
            chat_mood=chat_mood,
            visual_context=visual_context,
            recent_chat_messages=chat_messages,
            scene_description=scene_type,
            detected_objects=obj_summary,
            chat_stats=chat_stats,
            vlm_description=vlm_description,
        )
        reply = llm_res["reply"] if isinstance(llm_res, dict) else llm_res.reply
        engine = llm_res["engine"] if isinstance(llm_res, dict) else llm_res.engine
        llm_latency = (
            llm_res.get("latency_ms", 0)
            if isinstance(llm_res, dict)
            else getattr(llm_res, "latency_ms", 0)
        )
        print(f' -> Bot Chat Reply: "{reply}" (via {engine})')

        total_latency_ms = (time.perf_counter() - cycle_start) * 1000

        # Prometheus metrics recording
        self._record_telemetry(
            total_latency_ms=total_latency_ms,
            pos_ratio=pos_ratio,
            neg_ratio=neg_ratio,
            engine=engine,
            detected_labels=detected_labels,
            vid_latency=vid_latency,
            img_latency=img_latency,
            chat_latency=chat_res.get("latency_ms", 0)
            if isinstance(chat_res, dict)
            else getattr(chat_res, "latency_ms", 0),
            llm_latency=llm_latency,
        )

        return MultimodalCycleResult(
            streamer_speech=streamer_speech,
            chat_mood=chat_mood,
            visual_objects=detected_labels,
            scene_classification=scene_type,
            detailed_results={"vlm_description": vlm_description} if vlm_description else None,
            bot_chat_reply=reply,
            llm_engine=engine,
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
            ("image_classify", img_latency),
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


TwitchMultimodalCompanion = StreamMultimodalPipeline


if __name__ == "__main__":
    run_demo()
