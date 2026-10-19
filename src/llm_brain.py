"""Module 5: LLM Decision Engine / Bot Brain.

Connects to local Ollama (Qwen3.5 9B by default) for contextual chat responses.
Provides stream Q&A and multimodal synthesis via direct Ollama REST API.
"""

import time
import re
import requests
from typing import Dict, Any, Optional, List, Union

from src.core.config import settings
from src.core.schemas import LLMReplyResult
from src.core.interfaces import BaseLLMBrain
from src.ollama_client import generate
from src.prompts import (
    LIVE_BOT_PERSONALITY,
    LIVE_REACTION_PROMPT,
    LIVE_QUESTION_PROMPT,
    LIVE_QUESTION_SUFFIX,
)


def _clean_llm_reply(text: str) -> str:
    """Strips <think>...</think> reasoning blocks from hybrid thinking models like Qwen 3."""
    cleaned = re.sub(r"(?i)<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip().strip('"').strip("'")
    return cleaned if cleaned else text.strip()


class LocalLLMBrain(BaseLLMBrain):
    """Synthesizes multimodal context to generate in-character Twitch chat comments."""

    def __init__(self, ollama_base_url: Optional[str] = None, model_name: Optional[str] = None):
        self.ollama_base_url = (ollama_base_url or settings.models.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.models.ollama_model
        self.timeout = settings.models.ollama_timeout
        # Startup probe for the log only: requests are always attempted, so an Ollama server
        # started after the application is picked up without a restart.
        self.is_ollama_available = self._check_health()

        if self.is_ollama_available:
            print(
                f"[LLM] Connected to local Ollama at {self.ollama_base_url}. Model: '{self.model_name}'"
            )
        else:
            print(f"[LLM] Warning: Ollama not reachable at {self.ollama_base_url}.")

    def _options(self, **options) -> Dict[str, Any]:
        """Generation options shared by all requests; a fixed seed makes evaluation repeatable."""
        options["num_ctx"] = settings.models.ollama_context
        if settings.models.ollama_seed is not None:
            options["seed"] = settings.models.ollama_seed
        return options

    def _to_result(self, reply: str, data: Dict[str, Any], start: float) -> LLMReplyResult:
        """Builds the reply with token counts and the timings Ollama reports (nanoseconds)."""
        prompt_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        eval_ns = data.get("eval_duration", 0)
        return LLMReplyResult(
            reply=reply,
            engine=f"Ollama ({self.model_name})",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            speed_tps=round(output_tokens / (eval_ns / 1e9), 1) if eval_ns > 0 else 0.0,
            load_ms=round(data.get("load_duration", 0) / 1e6, 2),
            prompt_eval_ms=round(data.get("prompt_eval_duration", 0) / 1e6, 2),
        )

    def _check_health(self) -> bool:
        """Pings Ollama endpoint."""
        try:
            res = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return res.status_code == 200
        except Exception:
            return False

    def chat(self, messages: List[Dict[str, str]]) -> LLMReplyResult:
        """Send a standalone conversation without adding a persona or stream context."""
        start = time.perf_counter()
        data = generate(
            self.ollama_base_url,
            {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": self._options(temperature=0.8, num_predict=512),
            },
            self.timeout,
        )
        reply = _clean_llm_reply(data.get("message", {}).get("content", ""))
        if not reply:
            raise RuntimeError("Ollama вернула пустой ответ")
        return self._to_result(reply, data, start)

    def generate_reply(
        self,
        streamer_speech: str,
        chat_mood: str,
        visual_context: str,
        bot_personality: str = LIVE_BOT_PERSONALITY,
        recent_chat_messages: Optional[List[str]] = None,
        scene_description: str = "",
        detected_objects: Optional[Union[str, List[Any]]] = None,
        speech_history: Optional[List[str]] = None,
        chat_stats: Optional[Dict[str, Any]] = None,
        vlm_description: str = "",
        use_stream_prompt: bool = True,
    ) -> LLMReplyResult:
        """Generates an in-character Twitch chat reaction based on live multimodal context."""
        start = time.perf_counter()

        # Build comprehensive visual context
        if scene_description and detected_objects:
            visual_full = f"{scene_description}. Объекты: {detected_objects}"
        elif scene_description:
            visual_full = scene_description
        elif detected_objects:
            visual_full = f"Объекты: {detected_objects}"
        else:
            visual_full = visual_context or "Нет данных об изображении"
        vlm_line = (
            f"- Детали кадра и текст интерфейса (визуальный анализ): {vlm_description}\n"
            if vlm_description
            else ""
        )

        # Build speech context with history
        clean_speech = (
            streamer_speech
            if (streamer_speech and not streamer_speech.startswith("("))
            else "Стример молчит или играет музыка"
        )
        speech_hist_str = ""
        if speech_history:
            recent_speech = [f'"{p}"' for p in speech_history[-4:] if p and not p.startswith("(")]
            if recent_speech:
                speech_hist_str = f" (Недавно говорил: {', '.join(recent_speech)})"

        # Build chat snippet
        chat_snippet = ""
        if recent_chat_messages:
            lines = [f"  * {m}" for m in recent_chat_messages[-6:]]
            chat_snippet = "\n- Последние сообщения в чате Twitch:\n" + "\n".join(lines)

        # Build sentiment statistics snippet
        stats_snippet = ""
        if chat_stats and (
            chat_stats.get("positive_ratio") is not None
            or chat_stats.get("negative_ratio") is not None
        ):
            pos_pct = round(chat_stats.get("positive_ratio", 0) * 100)
            neg_pct = round(chat_stats.get("negative_ratio", 0) * 100)
            stats_snippet = f" (Позитив: {pos_pct}%, Негатив: {neg_pct}%)"

        prompt = LIVE_REACTION_PROMPT.format(
            bot_personality=bot_personality,
            visual_full=visual_full,
            vlm_line=vlm_line,
            clean_speech=clean_speech,
            speech_hist_str=speech_hist_str,
            chat_mood=chat_mood,
            stats_snippet=stats_snippet,
            chat_snippet=chat_snippet,
        )

        if not use_stream_prompt:
            # Standalone requests contain only text explicitly supplied by the user.
            prompt = "\n".join(
                value
                for value in (bot_personality, streamer_speech, chat_mood, visual_context)
                if value and value.strip()
            )
            if not prompt.strip():
                raise ValueError("Введите запрос к модели")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": self._options(temperature=0.8, top_p=0.9, num_predict=60),
        }
        data = generate(self.ollama_base_url, payload, self.timeout)
        return self._to_result(_clean_llm_reply(data.get("response", "")), data, start)

    def answer_stream_question(
        self,
        user_question: str,
        streamer_speech: str = "",
        chat_mood: str = "",
        visual_context: str = "",
        channel_name: str = "",
        chat_history: Optional[list] = None,
        recent_chat_messages: Optional[List[str]] = None,
        scene_description: str = "",
        detected_objects: Optional[Union[str, List[Any]]] = None,
        speech_history: Optional[List[str]] = None,
        speech_language: str = "russian",
        chat_stats: Optional[Dict[str, Any]] = None,
        session_duration: int = 0,
        stream_fps: float = 0.0,
        vlm_description: str = "",
    ) -> LLMReplyResult:
        """Answers user questions about the live stream using all multimodal signals."""
        history_str = ""
        if chat_history:
            recent = chat_history[-6:]
            for item in recent:
                if isinstance(item, dict):
                    role_label = "Зритель" if item.get("role") == "user" else "Ассистент"
                    history_str += f"{role_label}: {item.get('content', '')}\n"
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    history_str += f"Зритель: {item[0]}\nАссистент: {item[1]}\n"

        # 1. Computer Vision signals
        clean_scene = (
            scene_description.strip()
            if scene_description
            else (visual_context or "Нет отдельного описания сцены")
        )
        if isinstance(detected_objects, list):
            obj_strs = []
            for obj in detected_objects:
                cls_name = getattr(obj, "class_name", None) or (
                    obj.get("class_name") if isinstance(obj, dict) else str(obj)
                )
                conf = getattr(obj, "confidence", None) or (
                    obj.get("confidence") if isinstance(obj, dict) else None
                )
                if conf is not None:
                    obj_strs.append(f"{cls_name} ({conf * 100:.0f}%)")
                else:
                    obj_strs.append(str(cls_name))
            clean_objects = ", ".join(obj_strs) if obj_strs else "Объекты не выделены"
        elif detected_objects:
            clean_objects = str(detected_objects)
        else:
            clean_objects = visual_context or "Объекты не выделены"

        # 2. Audio & Speech signals
        clean_speech = (
            streamer_speech
            if (streamer_speech and not streamer_speech.startswith("("))
            else "Стример молчит или играет музыка"
        )
        if speech_history:
            valid_hist = [p for p in speech_history if p and not p.startswith("(")]
            if valid_hist:
                speech_history_str = "\n".join(
                    [f'  - [{i + 1}] "{phrase}"' for i, phrase in enumerate(valid_hist[-6:])]
                )
            else:
                speech_history_str = "  (Ранее зафиксированных реплик нет)"
        else:
            speech_history_str = "  (Ранее зафиксированных реплик нет)"

        # 3. Chat & Sentiment signals
        clean_mood = chat_mood or "Спокойное"
        if chat_stats and (
            chat_stats.get("positive_ratio") is not None
            or chat_stats.get("negative_ratio") is not None
        ):
            pos_pct = round(chat_stats.get("positive_ratio", 0) * 100)
            neg_pct = round(chat_stats.get("negative_ratio", 0) * 100)
            neu_pct = round(chat_stats.get("neutral_ratio", 0) * 100)
            total_m = chat_stats.get("total_messages", len(recent_chat_messages or []))
            chat_stats_str = f"Позитив: {pos_pct}%, Негатив: {neg_pct}%, Нейтрально: {neu_pct}% (сообщений в окне: {total_m})"
        else:
            chat_stats_str = clean_mood

        if recent_chat_messages:
            lines = [f"  * {m}" for m in recent_chat_messages[-15:]]
            chat_context_str = "\n".join(lines)
        else:
            chat_context_str = "  (Новых текстовых сообщений от зрителей пока нет)"

        # 4. Stream Telemetry
        clean_channel = channel_name.strip().lstrip("#") if channel_name else ""
        if clean_channel:
            stream_context_label = f"канала #{clean_channel}"
            channel_telemetry = f"#{clean_channel}"
        else:
            stream_context_label = "трансляции"
            channel_telemetry = "не указан / текущий эфир"

        session_duration_str = f"{session_duration} сек" if session_duration > 0 else "активна"
        fps_str = f"{stream_fps:.1f} FPS" if stream_fps > 0 else "N/A"
        vlm_line = (
            f"- Детали кадра и текст интерфейса (визуальный анализ): {vlm_description}\n"
            if vlm_description
            else ""
        )

        prompt = LIVE_QUESTION_PROMPT.format(
            stream_context_label=stream_context_label,
            clean_scene=clean_scene,
            clean_objects=clean_objects,
            vlm_line=vlm_line,
            speech_language=speech_language,
            clean_speech=clean_speech,
            speech_history_str=speech_history_str,
            clean_mood=clean_mood,
            chat_stats_str=chat_stats_str,
            chat_context_str=chat_context_str,
            channel_telemetry=channel_telemetry,
            session_duration_str=session_duration_str,
            fps_str=fps_str,
        )
        if history_str:
            prompt += f"\nПРЕДЫДУЩИЙ ДИАЛОГ:\n{history_str}\n"

        prompt += LIVE_QUESTION_SUFFIX.format(
            user_question=user_question,
        )

        start = time.perf_counter()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": self._options(temperature=0.7, num_predict=150),
        }
        data = generate(self.ollama_base_url, payload, self.timeout)
        return self._to_result(_clean_llm_reply(data.get("response", "")), data, start)
