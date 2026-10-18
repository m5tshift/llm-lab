"""Tab 1: Text Sentiment Analysis (RuBERT)."""

import gradio as gr
from src.web.state import web_state
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import record_model_latency, record_chat_stats


def handle_sentiment_single(text: str):
    if not text:
        return "Введите текст", 0.0, "0 ms"
    if not web_state.nlp_model:
        return "NLP модель не инициализирована", 0.0, "0 ms"
    res = web_state.nlp_model.analyze_message(text)
    safe_telemetry_dispatch(record_model_latency, "nlp_sentiment", res.latency_ms / 1000.0)
    label_display = (
        f"{res.label} [Эмоция: {res.dominant_emotion}]" if res.dominant_emotion else res.label
    )
    return label_display, res.score, f"{res.latency_ms} ms"


def handle_sentiment_batch(chat_text: str):
    if not chat_text:
        return "Чат пуст", "", "0 ms"
    if not web_state.nlp_model:
        return "NLP модель не инициализирована", "", "0 ms"
    lines = [line.strip() for line in chat_text.strip().split("\n") if line.strip()]
    res = web_state.nlp_model.analyze_chat_batch(lines)
    safe_telemetry_dispatch(record_model_latency, "nlp_sentiment", res.latency_ms / 1000.0)
    safe_telemetry_dispatch(record_chat_stats, res.positive_ratio, res.negative_ratio)
    em_info = (
        f" | Доминирующая эмоция: {res.dominant_emotion}"
        if hasattr(res, "dominant_emotion") and res.dominant_emotion
        else ""
    )
    summary = (
        f"Общее настроение: {res.dominant_mood}{em_info}\n"
        f"Позитивных: {res.positive_count} ({res.positive_ratio * 100:.0f}%)\n"
        f"Негативных: {res.negative_count} ({res.negative_ratio * 100:.0f}%)\n"
        f"Всего сообщений: {res.total_messages}"
    )
    details = "\n".join(
        [
            f"• [{d.label} | {d.dominant_emotion or 'нейтрально'}] {d.message} ({d.score * 100:.1f}%)"
            for d in res.detailed
        ]
    )
    return summary, details, f"{res.avg_latency_per_message_ms} ms/сообщение"


def create_text_tab():
    with gr.TabItem("1. Текст (RuBERT Emotion)"):
        gr.Markdown(
            "### Анализ настроения и дискретных эмоций чата (cointegrated/rubert-tiny2-cedr-emotion-detection)"
        )
        gr.Markdown(
            "Мультиэмоциональная трансформерная модель (29.3M параметров). Распознает 6 категорий эмоций: Радость, Удивление, Гнев, Грусть, Страх и Нейтрально со сведением к тональности зала."
        )
        with gr.Row():
            with gr.Column():
                single_msg_in = gr.Textbox(
                    label="Одиночное сообщение", value="Это был самый невероятный клатч в игре!"
                )
                btn_single_nlp = gr.Button("Оценить тональность")
            with gr.Column():
                single_msg_label = gr.Textbox(label="Результат классификации")
                single_msg_score = gr.Slider(
                    minimum=0.0, maximum=1.0, label="Уверенность (Confidence)"
                )
                single_msg_latency = gr.Textbox(label="Задержка (Latency)")

        btn_single_nlp.click(
            handle_sentiment_single,
            inputs=[single_msg_in],
            outputs=[single_msg_label, single_msg_score, single_msg_latency],
        )

        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                batch_msg_in = gr.Textbox(
                    label="Пакет сообщений из чата (по строкам)",
                    lines=5,
                    value="Отличный стрим сегодня, супер игра!\nСтример опять ошибся на ровном месте\nОчень крутой момент, продолжай в том же духе!\nСкучный раунд, надо собраться",
                )
                btn_batch_nlp = gr.Button("Анализировать тональность чата")
            with gr.Column():
                batch_summary = gr.Textbox(label="Сводка тональности чата", lines=4)
                batch_details = gr.Textbox(label="Разбор каждого сообщения", lines=5)
                batch_latency = gr.Textbox(label="Среднее время обработки")

        btn_batch_nlp.click(
            handle_sentiment_batch,
            inputs=[batch_msg_in],
            outputs=[batch_summary, batch_details, batch_latency],
        )
