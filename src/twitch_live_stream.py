"""Twitch Live Stream Facade & Compatibility Layer.

Re-exports Twitch connectors (Chat, Audio, Video, Observer) for backward compatibility
while providing an executable CLI entrypoint.
"""

import os
import argparse

from src.connectors.twitch_chat import TWITCH_DEFAULT_COLORS, get_twitch_color, TwitchChatListener
from src.connectors.twitch_audio import TwitchAudioListener
from src.connectors.twitch_video import TwitchStreamSource
from src.connectors.twitch_observer import LiveTwitchObserver

__all__ = [
    "TWITCH_DEFAULT_COLORS",
    "get_twitch_color",
    "TwitchChatListener",
    "TwitchAudioListener",
    "TwitchStreamSource",
    "LiveTwitchObserver",
    "main",
]


def main():
    """CLI runner for Twitch Live Stream AI Observer."""
    default_ch = os.getenv("DEFAULT_TWITCH_CHANNEL", "")
    parser = argparse.ArgumentParser(description="Twitch Live Stream AI Observer")
    parser.add_argument(
        "--channel",
        type=str,
        default=default_ch,
        required=not bool(default_ch),
        help="Twitch channel name (e.g. streamer_name)",
    )
    parser.add_argument(
        "--quality", type=str, default="720p", help="Stream quality: 360p, 480p, 720p, worst, best"
    )
    parser.add_argument(
        "--interval", type=int, default=10, help="Inference interval in seconds between cycles"
    )
    parser.add_argument(
        "--vlm-interval",
        type=int,
        default=20,
        help="Interval between background Qwen2-VL analyses in seconds",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable GUI video display window (headless mode)"
    )
    parser.add_argument("--cycles", type=int, default=None, help="Maximum number of cycles to run")

    args = parser.parse_args()

    observer = LiveTwitchObserver(channel=args.channel, quality=args.quality)
    observer.run_live(
        interval_seconds=args.interval,
        display=not args.no_display,
        max_cycles=args.cycles,
        vlm_interval_seconds=args.vlm_interval,
    )


if __name__ == "__main__":
    main()
