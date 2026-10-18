"""Tab 2: Audio Speech-to-Text (Whisper Small)."""

import gradio as gr
from src.web.state import web_state
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import record_model_latency


def handle_audio_transcribe(audio_path, language_choice="Русский"):
    if not audio_path:
        return "Аудиофайл не загружен", "0 s"
    if not web_state.audio_model:
        return "Audio модель не инициализирована", "0 s"
    lang_map = {"Русский": "russian", "English": "english", "Автоопределение": "auto"}
    target_lang = lang_map.get(language_choice, "russian")
    res = web_state.audio_model.transcribe_audio_file(audio_path, language=target_lang)
    latency_s = (
        res.inference_time_s
        if hasattr(res, "inference_time_s")
        else res.get("inference_time_s", 0.1)
    )
    text = res.transcription if hasattr(res, "transcription") else res.get("transcription", "")
    safe_telemetry_dispatch(record_model_latency, "audio_whisper", latency_s)
    return text, f"{latency_s} сек"


def create_audio_tab():
    with gr.TabItem("2. Аудио (Whisper)"):
        gr.Markdown("### Распознавание русской речи стримера (Семейство OpenAI Whisper)")
        gr.Markdown(
            "По умолчанию **Whisper Small** распознаёт речь на CPU. Для сравнения можно задать другую модель через WHISPER_MODEL перед запуском. Сверяйте текст с записью: музыка и паузы могут приводить к ошибкам. Время текущего запроса показано рядом с результатом."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Запись голоса или аудиофайл",
                )
                audio_lang = gr.Dropdown(
                    choices=["Русский", "English", "Автоопределение"],
                    value="Русский",
                    label="Язык распознавания",
                )
                btn_audio = gr.Button("Распознать речь в текст", variant="primary")
            with gr.Column():
                audio_out_text = gr.Textbox(label="Транскрибированный текст речи", lines=4)
                audio_out_latency = gr.Textbox(label="Время инференса Whisper")

        btn_audio.click(
            handle_audio_transcribe,
            inputs=[audio_in, audio_lang],
            outputs=[audio_out_text, audio_out_latency],
        )
