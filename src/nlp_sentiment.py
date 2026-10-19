"""Module 1: Text Processing (NLP) - Twitch Chat Sentiment & Emotion Analysis.

Framework: Hugging Face (transformers)
Task: Natural Language Processing / Emotion Detection & Sentiment Analysis
Model: cointegrated/rubert-tiny2-cedr-emotion-detection (five emotions + no_emotion)
"""

import time
from typing import List, Dict, Any, Optional
from transformers import pipeline

from src.core.config import settings
from src.core.schemas import SingleSentimentResult, BatchSentimentResult, SentimentDetail
from src.core.interfaces import BaseTextAnalyzer


EMOTION_TO_SENTIMENT = {
    "joy": "POSITIVE",
    "surprise": "POSITIVE",
    "anger": "NEGATIVE",
    "sadness": "NEGATIVE",
    "fear": "NEGATIVE",
    "no_emotion": "NEUTRAL",
}

EMOTION_RU_NAMES = {
    "joy": "Радость",
    "surprise": "Удивление",
    "anger": "Гнев",
    "sadness": "Грусть",
    "fear": "Страх",
    "no_emotion": "Нейтрально",
}


class ChatSentimentAnalyzer(BaseTextAnalyzer):
    """Analyzes the emotional tone and discrete emotions of chat messages using Hugging Face."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.models.nlp_model
        self.device = device or "cpu"
        print(f"[NLP] Loading Sentiment / Emotion model: '{self.model_name}'...")
        start = time.perf_counter()
        self.pipe = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            device=self.device,
        )
        self.load_time = time.perf_counter() - start
        print(f"[NLP] Model loaded successfully in {self.load_time:.2f}s")

    def _parse_scores(self, raw_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Maps raw classification scores to both sentiment (POSITIVE/NEGATIVE/NEUTRAL) and emotions."""
        has_emotions = any(s.get("label", "").lower() in EMOTION_TO_SENTIMENT for s in raw_scores)
        if has_emotions:
            top_item = max(raw_scores, key=lambda x: x["score"])
            top_label = top_item["label"].lower()
            mapped_sentiment = EMOTION_TO_SENTIMENT.get(top_label, "NEUTRAL")
            emotions_dict = {
                EMOTION_RU_NAMES.get(s["label"].lower(), s["label"]): round(float(s["score"]), 4)
                for s in raw_scores
            }
            return {
                "sentiment_label": mapped_sentiment,
                "sentiment_score": round(float(top_item["score"]), 4),
                "dominant_emotion": EMOTION_RU_NAMES.get(top_label, top_label),
                "emotions": emotions_dict,
            }
        else:
            top_item = (
                max(raw_scores, key=lambda x: x["score"])
                if raw_scores
                else {"label": "NEUTRAL", "score": 1.0}
            )
            raw_label = top_item["label"].upper()
            if "POS" in raw_label:
                clean_label = "POSITIVE"
            elif "NEG" in raw_label:
                clean_label = "NEGATIVE"
            else:
                clean_label = "NEUTRAL"
            return {
                "sentiment_label": clean_label,
                "sentiment_score": round(float(top_item["score"]), 4),
                "dominant_emotion": clean_label,
                "emotions": {s["label"]: round(float(s["score"]), 4) for s in raw_scores},
            }

    def predict_scores(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Raw per-label scores (model label names) for each text, for evaluation."""
        outputs = self.pipe(texts, batch_size=batch_size)
        return [{s["label"].lower(): float(s["score"]) for s in scores} for scores in outputs]

    def analyze_message(self, text: str) -> SingleSentimentResult:
        """Analyze sentiment and emotions for a single message."""
        start = time.perf_counter()
        raw_res = self.pipe(text)
        latency_ms = (time.perf_counter() - start) * 1000
        scores_list = raw_res[0] if raw_res and isinstance(raw_res[0], list) else raw_res
        parsed = self._parse_scores(scores_list)

        return SingleSentimentResult(
            text=text,
            label=parsed["sentiment_label"],
            score=parsed["sentiment_score"],
            latency_ms=round(latency_ms, 2),
            dominant_emotion=parsed["dominant_emotion"],
            emotions=parsed["emotions"],
        )

    def analyze_chat_batch(self, messages: List[str]) -> BatchSentimentResult:
        """Analyze a batch of message texts (without author prefixes) and aggregate the mood."""
        if not messages:
            return BatchSentimentResult(
                dominant_mood="NEUTRAL",
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=0.0,
                total_messages=0,
            )

        start = time.perf_counter()
        results = self.pipe(messages)
        total_latency_ms = (time.perf_counter() - start) * 1000

        detailed = []
        pos_count = 0
        neg_count = 0
        neutral_count = 0
        emotion_counts: Dict[str, int] = {}

        for msg, r in zip(messages, results):
            scores_list = r if isinstance(r, list) else [r]
            parsed = self._parse_scores(scores_list)
            sent = parsed["sentiment_label"]
            if sent == "POSITIVE":
                pos_count += 1
            elif sent == "NEGATIVE":
                neg_count += 1
            else:
                neutral_count += 1

            dom_em = parsed["dominant_emotion"]
            emotion_counts[dom_em] = emotion_counts.get(dom_em, 0) + 1

            detailed.append(
                SentimentDetail(
                    message=msg,
                    label=sent,
                    score=parsed["sentiment_score"],
                    dominant_emotion=dom_em,
                    emotions=parsed["emotions"],
                )
            )

        total = len(messages)
        pos_ratio = pos_count / total
        neg_ratio = neg_count / total
        neutral_ratio = neutral_count / total

        if pos_count > neg_count and pos_count >= neutral_count:
            dominant = "ХАЙП / ПОЗИТИВ"
        elif neg_count > pos_count and neg_count >= neutral_count:
            dominant = "ТОКСИЧНО / НЕГАТИВ"
        else:
            dominant = "СБАЛАНСИРОВАННОЕ"

        top_em = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "Нейтрально"

        return BatchSentimentResult(
            total_messages=total,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neutral_count,
            positive_ratio=round(pos_ratio, 2),
            negative_ratio=round(neg_ratio, 2),
            neutral_ratio=round(neutral_ratio, 2),
            dominant_mood=dominant,
            dominant_emotion=top_em,
            latency_ms=round(total_latency_ms, 2),
            avg_latency_per_message_ms=round(total_latency_ms / total, 2),
            detailed=detailed,
        )


if __name__ == "__main__":
    analyzer = ChatSentimentAnalyzer()
    sample_chat = [
        "Отличный стрим сегодня, супер игра!",
        "Стример опять ошибся на ровном месте...",
        "Какой невероятный клатч, просто лучший!",
        "Скучный раунд, надо собраться",
        "Прекрасная атмосфера в чате, продолжаем тащить",
    ]
    print("\n--- Testing Single Message Analysis ---")
    single_res = analyzer.analyze_message("Невероятный момент в игре!")
    print(single_res)
    print("\n--- Testing Batch Chat Analysis ---")
    batch_res = analyzer.analyze_chat_batch(sample_chat)
    print(f"Dominant Mood: {batch_res.dominant_mood} (Эмоция: {batch_res.dominant_emotion})")
    print(
        f"Positive: {batch_res.positive_ratio * 100}% | Negative: {batch_res.negative_ratio * 100}%"
    )
