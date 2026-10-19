"""
Twitch Live Stream High-Level Observer.
Orchestrates chat listener, stream video source, and multimodal pipeline.
"""

import os
import time
from typing import Optional
import cv2

from src.connectors.twitch_chat import TwitchChatListener
from src.connectors.twitch_video import TwitchStreamSource
from src.bot_pipeline import StreamMultimodalPipeline
from src.live_vision import BackgroundVLMAnalyzer


class LiveTwitchObserver:
    """High-level orchestrator connecting Live Twitch Video & Chat to the Multimodal Pipeline."""

    def __init__(
        self,
        channel: str,
        quality: str = "720p",
        pipeline: Optional[StreamMultimodalPipeline] = None,
    ):
        self.channel = channel.lstrip("#").lower()
        self.quality = quality
        self.chat_listener = TwitchChatListener(channel)
        self.stream_source = TwitchStreamSource(channel, quality)
        self.pipeline = pipeline or StreamMultimodalPipeline()

    def run_live(
        self,
        interval_seconds: int = 12,
        display: bool = True,
        max_cycles: Optional[int] = None,
        vlm_interval_seconds: int = 20,
    ):
        """Runs the live monitoring loop."""
        print("\n" + "=" * 65)
        print(f"Starting Live Twitch AI Observer on channel: #{self.channel}")
        print("=" * 65)

        # 1. Start Chat Listener
        self.chat_listener.start()

        # 2. Connect to Video Stream
        if not self.stream_source.connect():
            print(f"[Error] Could not connect to Twitch video for '{self.channel}'.")
            print("Make sure the streamer is LIVE right now and channel name is correct.")
            self.chat_listener.stop()
            return

        print(
            f"\n[Observer] Listening to #{self.channel}. Inference interval: {interval_seconds}s."
        )
        if display:
            print("[Observer] OpenCV HUD window active. Press 'q' in the video window to stop.")

        cycle_count = 0
        audio_tmp = f"live_{self.channel}_audio.wav"
        vlm_analyzer = BackgroundVLMAnalyzer(
            interval_seconds=vlm_interval_seconds,
            max_age_seconds=max(60, vlm_interval_seconds * 2),
        )

        try:
            while True:
                cycle_count += 1
                cycle_start = time.time()
                print(
                    f"\n>>> [Live Cycle #{cycle_count}] Fetching stream data from #{self.channel}..."
                )

                # 1. Grab Frame
                frame = self.stream_source.get_frame()
                if frame is None:
                    print("[Observer] Failed to read frame from stream. Re-syncing...")
                    time.sleep(2)
                    continue

                vlm_analyzer.submit(frame)

                # 2. Grab Chat
                chat_messages = self.chat_listener.get_recent_messages(limit=15)
                if not chat_messages:
                    chat_messages = ["(Chat is calm or just started observing)"]
                print(f"[Observer] Collected {len(chat_messages)} recent chat messages.")

                # 3. Audio chunk
                audio_file = self.stream_source.capture_audio_chunk(
                    duration_s=4, output_wav=audio_tmp
                )

                # 4. Multimodal Process Cycle
                result = self.pipeline.process_stream_cycle(
                    audio_file_path=audio_file if audio_file else "none",
                    image_or_frame=frame,
                    chat_messages=chat_messages,
                    vlm_description=vlm_analyzer.get_latest(),
                )

                # 5. Display HUD on Video Frame
                if display:
                    annotated_frame = frame.copy()
                    vid_info = self.pipeline.video_detector.process_frame(annotated_frame)
                    annotated = vid_info.get("annotated_frame", annotated_frame)
                    if annotated is None:
                        annotated = annotated_frame

                    # Draw Twitch Bot HUD Card
                    h, w = annotated.shape[:2]
                    cv2.rectangle(annotated, (10, h - 110), (w - 10, h - 10), (20, 20, 20), -1)
                    cv2.rectangle(annotated, (10, h - 110), (w - 10, h - 10), (147, 51, 234), 2)

                    reply_text = f"TwitchBot: {result['bot_chat_reply']}"
                    mood_text = f"Room Mood: {result['chat_mood']}"
                    scene_text = f"Scene: {result['scene_classification']}"

                    cv2.putText(
                        annotated,
                        mood_text,
                        (25, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 100),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        scene_text,
                        (25, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (100, 255, 200),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        reply_text[:75],
                        (25, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow(f"Twitch AI Observer - #{self.channel}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Observer] Exit signal received.")
                        break

                if max_cycles and cycle_count >= max_cycles:
                    print(f"[Observer] Completed {max_cycles} cycles.")
                    break

                # Sleep remaining time until next observation cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0.5, interval_seconds - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Observer] Stopping observer (Ctrl+C)...")
        finally:
            vlm_analyzer.close()
            self.chat_listener.stop()
            self.stream_source.release()
            if display:
                cv2.destroyAllWindows()
            if os.path.exists(audio_tmp):
                try:
                    os.remove(audio_tmp)
                except Exception:
                    pass
            print(f"[Observer] #{self.channel} monitoring stopped.")
