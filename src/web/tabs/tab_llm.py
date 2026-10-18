"""Tab 6: Standalone LLM requests with optional user instructions and context."""

import gradio as gr
from src.web.state import web_state
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import record_model_latency, record_bot_reply


def handle_llm_reaction(instructions, context, prompt):
    reply, status, _, _ = handle_llm_session(instructions, context, prompt, False, [])
    return reply, status


def handle_llm_session(instructions, context, prompt, session_enabled, history):
    history = [dict(message) for message in (history or [])] if session_enabled else []
    prompt = (prompt or "").strip()
    if not prompt:
        return "Введите промпт", "0 ms", history, history
    if not web_state.llm_model:
        return "LLM модель не подключена", "0 ms", history, history
    try:
        user_message = {
            "role": "user",
            "content": "\n".join(
                part.strip() for part in (context, prompt) if part and part.strip()
            ),
        }
        messages = []
        if instructions and instructions.strip():
            messages.append({"role": "system", "content": instructions.strip()})
        messages.extend(history)
        messages.append(user_message)
        res = web_state.llm_model.chat(messages)
        reply = res.reply if hasattr(res, "reply") else res.get("reply", "")
        engine = res.engine if hasattr(res, "engine") else res.get("engine", "local")
        lat_ms = res.latency_ms if hasattr(res, "latency_ms") else res.get("latency_ms", 0.0)

        safe_telemetry_dispatch(record_model_latency, "llm_brain", lat_ms / 1000.0)
        safe_telemetry_dispatch(record_bot_reply, engine)

        status_str = f"Движок: {engine} | Latency: {lat_ms} ms"
        tps = getattr(res, "speed_tps", 0.0) or (
            res.get("speed_tps", 0.0) if isinstance(res, dict) else 0.0
        )
        if tps > 0:
            status_str += f" | Скорость: {tps} tok/s"
        if session_enabled:
            history.extend([user_message, {"role": "assistant", "content": reply}])
        return reply, status_str, history, history
    except Exception as exc:
        return f"Ошибка обращения к Ollama: {exc}", "0 ms", history, history


def reset_llm_session(enabled):
    return [], gr.update(value=[], visible=enabled), gr.update(visible=enabled), "", ""


def create_llm_tab():
    with gr.TabItem("6. LLM (локальная модель)"):
        gr.Markdown("### 6. Общение с LLM")
        gr.Markdown(
            "Введите промпт. При необходимости добавьте инструкции и дополнительный контекст."
        )
        session_history = gr.State([])
        session_enabled = gr.Checkbox(label="Сессия с историей переписки", value=False)
        gr.Markdown(
            "В режиме сессии модель учитывает предыдущие сообщения. "
            "История хранится до очистки, переключения режима или обновления страницы."
        )
        with gr.Row():
            with gr.Column():
                llm_instructions = gr.Textbox(
                    label="Инструкции (опционально)",
                    value="",
                    lines=2,
                )
                llm_context = gr.Textbox(
                    label="Дополнительный контекст (опционально)",
                    value="",
                    lines=3,
                )
                llm_prompt = gr.Textbox(label="Промпт", value="", lines=3)
                btn_llm = gr.Button("Отправить", variant="primary")
            with gr.Column():
                llm_out = gr.Textbox(label="Ответ LLM", lines=8, interactive=False)
                llm_latency = gr.Textbox(label="Метрики инференса и движок")
        conversation = gr.Chatbot(label="Переписка текущей сессии", visible=False, height=400)
        btn_reset = gr.Button("Начать новую сессию", visible=False)

        btn_llm.click(
            handle_llm_session,
            inputs=[llm_instructions, llm_context, llm_prompt, session_enabled, session_history],
            outputs=[llm_out, llm_latency, session_history, conversation],
            concurrency_id="standalone_llm_session",
            concurrency_limit=1,
        )
        reset_outputs = [session_history, conversation, btn_reset, llm_out, llm_latency]
        session_enabled.change(
            reset_llm_session,
            inputs=[session_enabled],
            outputs=reset_outputs,
            concurrency_id="standalone_llm_session",
            concurrency_limit=1,
        )
        btn_reset.click(
            reset_llm_session,
            inputs=[session_enabled],
            outputs=reset_outputs,
            concurrency_id="standalone_llm_session",
            concurrency_limit=1,
        )
