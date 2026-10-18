"""
Centralized application configuration using standard Pydantic.
Structured into domain sub-models (models, network, twitch) with environment variable overrides.
"""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    vision_backend: Literal["ollama", "transformers"] = Field(
        default_factory=lambda: os.getenv("VISION_BACKEND", "ollama")
    )
    ollama_context: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_CONTEXT", "4096")), ge=512
    )
    nlp_model: str = Field(
        default_factory=lambda: os.getenv(
            "NLP_MODEL", "cointegrated/rubert-tiny2-cedr-emotion-detection"
        ),
        description="HuggingFace model for chat sentiment and emotion analysis",
    )
    whisper_model: str = Field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", "openai/whisper-small"),
        description="Whisper model checkpoint (openai/whisper-small or openai/whisper-large-v3-turbo)",
    )
    whisper_language: str = Field(
        default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "ru"),
        description="Target transcription language",
    )
    image_model: str = Field(
        default_factory=lambda: os.getenv("IMAGE_MODEL", "Salesforce/blip-image-captioning-large"),
        description="Vision image-to-text / captioning model backbone",
    )
    vlm_model: str = Field(
        default_factory=lambda: os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct"),
        description="Vision-Language Model (VLM) for deep scene reasoning, UI, and Russian OCR",
    )
    yolo_model: str = Field(
        default_factory=lambda: os.getenv("YOLO_MODEL", "yolo26s.pt"),
        description="YOLO detection model checkpoint",
    )
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434")
        ),
        description="Local Ollama server endpoint",
    )
    ollama_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
        description="Target LLM model tag",
    )
    ollama_timeout: float = Field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "120.0")),
        description="LLM request timeout in seconds",
    )


class NetworkSettings(BaseModel):
    api_host: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"), description="FastAPI bind address"
    )
    api_port: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000")), description="FastAPI port"
    )
    gradio_host: str = Field(
        default_factory=lambda: os.getenv("WEB_HOST", os.getenv("GRADIO_HOST", "0.0.0.0")),
        description="Gradio Web UI bind address",
    )
    gradio_port: int = Field(
        default_factory=lambda: int(os.getenv("WEB_PORT", os.getenv("GRADIO_PORT", "7860"))),
        description="Gradio Web UI port",
    )
    metrics_port: int = Field(
        default_factory=lambda: int(
            os.getenv(
                "WEB_METRICS_PORT", os.getenv("METRICS_PORT", os.getenv("PROMETHEUS_PORT", "8001"))
            )
        ),
        description="Prometheus scrape port",
    )


class TwitchSettings(BaseModel):
    default_channel: str = Field(
        default_factory=lambda: os.getenv(
            "DEFAULT_TWITCH_CHANNEL", os.getenv("TWITCH_CHANNEL", "")
        ),
        description="Target Twitch channel name",
    )
    irc_server: str = Field(
        default_factory=lambda: os.getenv("TWITCH_IRC_SERVER", "irc.chat.twitch.tv"),
        description="Twitch IRC server",
    )
    irc_port: int = Field(
        default_factory=lambda: int(os.getenv("TWITCH_IRC_PORT", "6697")),
        description="Twitch TLS IRC port",
    )
    nickname: str = Field(
        default_factory=lambda: os.getenv("TWITCH_NICKNAME", "justinfan12345"),
        description="Twitch anonymous nickname",
    )
    token: Optional[str] = Field(
        default_factory=lambda: os.getenv("TWITCH_TOKEN", None), description="Twitch OAuth token"
    )


class AppConfig(BaseModel):
    """Root application configuration."""

    model_config = {"extra": "ignore"}

    models: ModelSettings = Field(default_factory=ModelSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    twitch: TwitchSettings = Field(default_factory=TwitchSettings)

    # Root backward-compatibility properties
    @property
    def API_HOST(self) -> str:
        return self.network.api_host

    @property
    def API_PORT(self) -> int:
        return self.network.api_port

    @property
    def WEB_HOST(self) -> str:
        return self.network.gradio_host

    @property
    def WEB_PORT(self) -> int:
        return self.network.gradio_port

    @property
    def TWITCH_CHANNEL(self) -> str:
        return self.twitch.default_channel

    @property
    def OLLAMA_URL(self) -> str:
        return self.models.ollama_base_url

    @property
    def OLLAMA_MODEL(self) -> str:
        return self.models.ollama_model

    @property
    def WHISPER_MODEL(self) -> str:
        return self.models.whisper_model

    @property
    def YOLO_MODEL(self) -> str:
        return self.models.yolo_model


# Singleton instance
settings = AppConfig()
