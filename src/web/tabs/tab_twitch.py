"""Tab 6: Live Twitch Stream Observer, Real-time Dashboard & Multimodal AI Assistant."""

import time
import threading
import cv2
import gradio as gr

from src.core.config import settings
from src.live_vision import BackgroundVLMAnalyzer, LIVE_VLM_PROMPT
from src.scene_classify import SCENE_LABELS_RU
from src.prompts import (
    LIVE_BOT_PERSONALITY,
    LIVE_REACTION_PROMPT,
    LIVE_QUESTION_PROMPT,
    LIVE_QUESTION_SUFFIX,
)
from src.vision_backend import OllamaVisionClassifier
from src.web.state import web_state
from src.web.styles import make_chat_status_html
from src.connectors.twitch_chat import TwitchChatListener
from src.connectors.twitch_video import TwitchStreamSource
from src.connectors.twitch_audio import TwitchAudioListener
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import (
    record_model_latency,
    record_chat_stats,
    record_detected_objects,
    record_bot_reply,
    record_frames_processed,
    FrameRateTracker,
    record_ai_update,
    record_processing_error,
    set_stream_active,
    update_system_resource_metrics,
)


def handle_stream_question(user_question: str, history: list, channel_name: str):
    """Answers user questions regarding the stream context using all multimodal signals."""
    if not user_question or not user_question.strip():
        return history, ""

    history = history or []
    q = user_question.strip()
    st = web_state.stream_state
    ch = channel_name.strip() if channel_name else st.get("channel", "")
    sp = st.get("speech", "")
    sp_hist = st.get("speech_history", [])
    vis = st.get("visual", "")
    scene = st.get("scene", "")
    vlm = st.get("vlm", "")
    mood = st.get("mood", "")
    chat_stats = st.get("chat_stats", {})
    fps = st.get("fps", 0.0)
    session_time = st.get("session_time", 0)

    # 1. Retrieve real-time audio and speech history if active
    if getattr(web_state, "active_audio_listener", None):
        try:
            active_hist = web_state.active_audio_listener.get_speech_history()
            if active_hist:
                sp_hist = active_hist
            cur_sp = web_state.active_audio_listener.get_latest_speech()
            if cur_sp:
                sp = cur_sp
        except Exception:
            pass

    # 2. Retrieve real-time Twitch IRC messages with timestamps & authors
    chat_msgs = []
    if getattr(web_state, "active_chat_listener", None):
        try:
            chat_msgs = web_state.active_chat_listener.get_recent_formatted_chat(
                limit=25, include_timestamp=True
            )
        except Exception:
            pass
    if not chat_msgs:
        chat_msgs = st.get("chat_messages", [])

    if web_state.llm_model:
        try:
            res = web_state.llm_model.answer_stream_question(
                user_question=q,
                streamer_speech=sp,
                chat_mood=mood,
                visual_context=vis,
                channel_name=ch,
                chat_history=history,
                recent_chat_messages=chat_msgs,
                scene_description=scene,
                detected_objects=vis,
                speech_history=sp_hist,
                chat_stats=chat_stats,
                session_duration=session_time,
                stream_fps=fps,
                vlm_description=vlm,
            )
            answer = (
                res.reply
                if hasattr(res, "reply")
                else res.get("reply", "Не удалось сгенерировать ответ.")
            )
            tokens = getattr(res, "total_tokens", 0) or (
                res.get("total_tokens", 0) if isinstance(res, dict) else 0
            )
            lat = getattr(res, "latency_ms", 0.0) or (
                res.get("latency_ms", 0.0) if isinstance(res, dict) else 0.0
            )
            speed = getattr(res, "speed_tps", 0.0) or (
                res.get("speed_tps", 0.0) if isinstance(res, dict) else 0.0
            )
            if tokens:
                prompt_t = getattr(res, "prompt_tokens", 0) or (
                    res.get("prompt_tokens", 0) if isinstance(res, dict) else 0
                )
                comp_t = getattr(res, "completion_tokens", 0) or (
                    res.get("completion_tokens", 0) if isinstance(res, dict) else 0
                )
                answer += f"\n\n*(Токены: {tokens} [Промпт: {prompt_t}, Ответ: {comp_t}], Скорость: {speed} tok/s, Latency: {lat}ms)*"
            safe_telemetry_dispatch(record_model_latency, "llm_brain", lat / 1000.0)
            safe_telemetry_dispatch(record_bot_reply, getattr(res, "engine", "local"))
        except Exception as exc:
            answer = f"Ошибка обращения к локальной модели Ollama: {exc}"
    else:
        answer = "Локальная модель LLM сейчас не загружена."

    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def stream_live_twitch(
    channel_name: str,
    quality: str,
    ai_interval: int = 6,
    max_frames: int = 60,
    audio_lang_choice: str = "Русский",
    vlm_interval: int = 20,
):
    """Runs continuous frame-by-frame live streaming and periodic multimodal AI updates."""
    if not channel_name or not channel_name.strip():
        yield (
            None,
            make_chat_status_html(
                "Укажите имя канала Twitch (например, 'dota2ti', 'rostislav_999')", "#fbbf24"
            ),
            "",
            None,
            "",
            "",
            "",
            "",
            "0 ms",
            "",
        )
        return

    clean_channel = channel_name.strip().lstrip("#").lower()
    yield (
        None,
        make_chat_status_html(
            f"Подключение к #{clean_channel} (HLS видео + IRC чат)...", "#60a5fa"
        ),
        "Инициализация аудио...",
        None,
        "Инициализация...",
        "Инициализация...",
        "Ожидание запуска...",
        "0 ms",
        "Ожидание анализа VLM...",
    )

    listener = None
    source = None
    audio_listener = None
    vlm_analyzer = None
    try:
        source = TwitchStreamSource(clean_channel, quality=quality)
        if not source.connect():
            yield (
                None,
                make_chat_status_html(
                    f"Канал '{clean_channel}' сейчас не в эфире (OFFLINE) или недоступен.",
                    "#ef4444",
                ),
                "OFFLINE",
                None,
                "OFFLINE",
                "OFFLINE",
                "Стрим недоступен",
                "0 ms",
                "",
            )
            return

        listener = TwitchChatListener(clean_channel)
        listener.start()
        web_state.active_chat_listener = listener

        lang_map = {"Русский": "russian", "English": "english", "Автоопределение": "auto"}
        selected_lang = lang_map.get(audio_lang_choice, "russian")
        if web_state.audio_model and source.stream_url:
            audio_listener = TwitchAudioListener(
                lambda: source.stream_url,
                audio_transcriber=web_state.audio_model,
                chunk_duration_s=3,
                interval_s=5,
                language=selected_lang,
            )
            audio_listener.start()
            web_state.active_audio_listener = audio_listener

        yield (
            None,
            make_chat_status_html(
                f"Подключено к #{clean_channel}. Старт анализа трансляции...", "#4ade80"
            ),
            "Подключение звука...",
            None,
            "Поток активен",
            "Чат подключен",
            "Запуск моделей...",
            "0 ms",
            "Ожидание анализа VLM...",
        )
        vlm_analyzer = BackgroundVLMAnalyzer(
            model_factory=lambda: web_state.vision_model or OllamaVisionClassifier(),
            interval_seconds=vlm_interval,
            max_age_seconds=max(60, vlm_interval * 2),
        )
        safe_telemetry_dispatch(set_stream_active, True)
        if web_state.video_model:
            web_state.video_model.reset_tracking()

        start_time = time.time()
        last_ai_time = start_time
        ai_busy = False
        ai_lock = threading.Lock()
        last_audio_snippet = None
        cached_obj_text = "Детекция объектов..."
        cached_scene_text = "Классификация..."
        cached_mood_text = "Сбор чата..."
        cached_bot_reply = "Ожидание контекста..."
        cached_speech = "Ожидание речи стримера..."
        cached_chat_stats = {}
        frame_count = 0
        last_seq = -1
        fps_tracker = FrameRateTracker()
        current_fps = 0.0
        elapsed = 0

        def _async_ai_worker(snap_frame, msgs, speech_txt):
            nonlocal \
                cached_scene_text, \
                cached_mood_text, \
                cached_bot_reply, \
                cached_chat_stats, \
                ai_busy
            update_started = time.perf_counter()
            try:
                # 1. Chat sentiment (RuBERT)
                if msgs and web_state.nlp_model:
                    chat_res = web_state.nlp_model.analyze_chat_batch(msgs)
                    with ai_lock:
                        cached_mood_text = f"{chat_res.dominant_mood} (Позитив: {chat_res.positive_ratio * 100:.0f}%, Негатив: {chat_res.negative_ratio * 100:.0f}%)"
                        cached_chat_stats = {
                            "dominant_mood": chat_res.dominant_mood,
                            "positive_ratio": chat_res.positive_ratio,
                            "negative_ratio": chat_res.negative_ratio,
                            "neutral_ratio": chat_res.neutral_ratio,
                            "total_messages": chat_res.total_messages,
                        }
                    safe_telemetry_dispatch(
                        record_model_latency, "nlp_sentiment", chat_res.latency_ms / 1000.0
                    )
                    safe_telemetry_dispatch(
                        record_chat_stats, chat_res.positive_ratio, chat_res.negative_ratio
                    )

                # 2. Scene type (CLIP zero-shot)
                if snap_frame is not None and web_state.scene_model:
                    scene_res = web_state.scene_model.classify_image(snap_frame)
                    scene_name = SCENE_LABELS_RU.get(
                        scene_res.top_prediction, scene_res.top_prediction
                    )
                    with ai_lock:
                        cached_scene_text = f"{scene_name} ({scene_res.confidence * 100:.0f}%)"
                    safe_telemetry_dispatch(
                        record_model_latency,
                        "image_scene",
                        scene_res.inference_latency_ms / 1000.0,
                    )

                # 3. LLM Reaction (with the background frame description from the LLM)
                if web_state.llm_model:
                    with ai_lock:
                        cur_obj = cached_obj_text
                        cur_scene = cached_scene_text
                        cur_mood = cached_mood_text
                        cur_stats = dict(cached_chat_stats)
                    visual_ctx = f"Объекты: {cur_obj}. Сцена: {cur_scene}"
                    prompt_speech = (
                        speech_txt
                        if speech_txt and not speech_txt.startswith("(")
                        else f"Стрим #{clean_channel}"
                    )
                    chat_formatted_list = (
                        listener.get_recent_formatted_chat(limit=15, include_timestamp=True)
                        if listener
                        else msgs
                    )
                    speech_hist = audio_listener.get_speech_history() if audio_listener else []
                    llm_res = web_state.llm_model.generate_reply(
                        streamer_speech=prompt_speech,
                        chat_mood=cur_mood,
                        visual_context=visual_ctx,
                        recent_chat_messages=chat_formatted_list,
                        scene_description=cur_scene,
                        detected_objects=cur_obj,
                        speech_history=speech_hist,
                        chat_stats=cur_stats,
                        vlm_description=vlm_analyzer.get_latest(),
                    )
                    with ai_lock:
                        cached_bot_reply = f"{llm_res.reply} ({llm_res.engine})"
                    safe_telemetry_dispatch(
                        record_model_latency, "llm_brain", llm_res.latency_ms / 1000.0
                    )
                    safe_telemetry_dispatch(record_bot_reply, llm_res.engine)
            except Exception as e:
                safe_telemetry_dispatch(record_processing_error, "live_ai")
                print(f"[Async AI error] {e}")
            finally:
                safe_telemetry_dispatch(record_ai_update, time.perf_counter() - update_started)
                ai_busy = False

        while frame_count < max_frames:
            t_frame_start = time.perf_counter()
            seq, frame = source.get_frame_with_seq()
            if frame is None or seq == last_seq:
                time.sleep(0.01)  # Track only newly decoded frames, never the same frame twice.
                continue
            last_seq = seq

            frame_count += 1
            vlm_analyzer.submit(frame)

            if web_state.video_model:
                det_res = web_state.video_model.track_frame(frame)
                annotated_rgb = cv2.cvtColor(det_res.annotated_frame, cv2.COLOR_BGR2RGB)
                safe_telemetry_dispatch(record_detected_objects, det_res.detected_objects)
                safe_telemetry_dispatch(
                    record_model_latency, "video_yolo", det_res.latency_ms / 1000.0
                )

                in_frame = {}
                for obj in det_res.detected_objects:
                    in_frame[obj.class_name] = in_frame.get(obj.class_name, 0) + 1
                with ai_lock:
                    cached_obj_text = (
                        ", ".join(
                            f"{name}: {n} в кадре, {det_res.unique_objects.get(name, n)} за сессию"
                            for name, n in in_frame.items()
                        )
                        or "Объекты не найдены"
                    )
            else:
                annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_duration = time.perf_counter() - t_frame_start
            frame_latency = frame_duration * 1000
            safe_telemetry_dispatch(record_frames_processed, 1)
            current_fps = fps_tracker.tick()
            elapsed = int(time.time() - start_time)

            with ai_lock:
                web_state.stream_state["channel"] = clean_channel
                web_state.stream_state["speech"] = cached_speech
                web_state.stream_state["visual"] = cached_obj_text
                web_state.stream_state["scene"] = cached_scene_text
                web_state.stream_state["vlm"] = vlm_analyzer.get_latest()
                web_state.stream_state["mood"] = cached_mood_text
                web_state.stream_state["chat_stats"] = dict(cached_chat_stats)
                if audio_listener:
                    web_state.stream_state["speech_history"] = audio_listener.get_speech_history()
                if listener:
                    web_state.stream_state["chat_messages"] = listener.get_recent_formatted_chat(
                        limit=25, include_timestamp=True
                    )
                web_state.stream_state["fps"] = current_fps
                web_state.stream_state["session_time"] = elapsed
                web_state.stream_state["is_live"] = True

            now = time.time()
            if (now - last_ai_time >= ai_interval) and not ai_busy:
                last_ai_time = now
                ai_busy = True
                recent_msgs = listener.get_recent_messages(limit=12) if listener else []
                cur_speech = audio_listener.get_latest_speech() if audio_listener else ""
                threading.Thread(
                    target=_async_ai_worker,
                    args=(frame.copy(), recent_msgs, cur_speech),
                    daemon=True,
                ).start()

            sys_stats = update_system_resource_metrics() if frame_count % 10 == 0 else {}
            proc_mem = sys_stats.get("process_memory_mb")
            mem_str = f" | RAM: {proc_mem:.0f}MB" if proc_mem else ""
            stats = f"LIVE (#{frame_count}) | Время: {elapsed}с | FPS: {current_fps:.1f}{mem_str} | Latency: {frame_latency:.1f}ms"

            chat_html = listener.get_recent_chat_html(limit=40, max_height=860) if listener else ""
            if audio_listener:
                cached_speech = audio_listener.get_latest_speech()

            new_audio = audio_listener.get_latest_audio_path() if audio_listener else None
            if new_audio and new_audio != last_audio_snippet:
                audio_yield = new_audio
                last_audio_snippet = new_audio
            else:
                audio_yield = last_audio_snippet

            with ai_lock:
                yield_vis = f"{cached_obj_text} | Сцена: {cached_scene_text}"
                yield_mood = cached_mood_text
                yield_reply = cached_bot_reply

            yield (
                annotated_rgb,
                chat_html,
                cached_speech,
                audio_yield,
                yield_vis,
                yield_mood,
                yield_reply,
                stats,
                vlm_analyzer.get_status(),
            )
            time.sleep(0.02)

    except Exception as e:
        yield (
            None,
            make_chat_status_html(f"Ошибка во время трансляции: {e}", "#ef4444"),
            "",
            None,
            "",
            "",
            "",
            "0 ms",
            "",
        )
    finally:
        if vlm_analyzer:
            vlm_analyzer.close()
        web_state.stream_state["vlm"] = ""
        safe_telemetry_dispatch(set_stream_active, False)
        web_state.stream_state["is_live"] = False
        web_state.active_chat_listener = None
        web_state.active_audio_listener = None
        if audio_listener:
            audio_listener.stop()
        if listener:
            listener.stop()
        if source:
            source.release()


def create_twitch_tab():
    """Builds Tab 7: Live Twitch Stream Observer, Real-time Dashboard and Multimodal AI Assistant."""
    with gr.TabItem("7. Живой стрим Twitch (Live)"):
        gr.Markdown(
            "### 7. Прямая трансляция Twitch в реальном времени (HLS Video + Anonymous IRC Chat)"
        )
        gr.Markdown(
            f"**RuBERT** — чат; **Whisper Small** — речь; **CLIP** — тип сцены; **YOLO26s + ByteTrack** — объекты; **{settings.models.ollama_model}** — описание кадра, реакции и ответы."
        )

        with gr.Accordion("Препромпты, используемые на этой странице", open=False):
            gr.Markdown(
                "Поля в фигурных скобках заполняются текущими данными трансляции. "
                "В ответы на вопросы также добавляются последние сообщения диалога."
            )
            gr.Textbox(
                label="Анализ кадра (VLM)", value=LIVE_VLM_PROMPT, lines=4, interactive=False
            )
            gr.Textbox(
                label="Характер бота", value=LIVE_BOT_PERSONALITY, lines=2, interactive=False
            )
            gr.Textbox(
                label="Шаблон реакции бота", value=LIVE_REACTION_PROMPT, lines=10, interactive=False
            )
            gr.Textbox(
                label="Шаблон ответа на вопрос",
                value=LIVE_QUESTION_PROMPT
                + "\nПРЕДЫДУЩИЙ ДИАЛОГ:\n{history_str}\n"
                + LIVE_QUESTION_SUFFIX,
                lines=18,
                interactive=False,
            )

        default_channel = settings.twitch.default_channel
        with gr.Group():
            gr.Markdown("#### Параметры подключения и управление эфиром")
            with gr.Row(equal_height=True):
                live_channel = gr.Textbox(label="Канал Twitch", value=default_channel, scale=3)
                live_quality = gr.Dropdown(
                    choices=["worst", "360p", "480p", "720p", "best"],
                    value="720p",
                    label="Качество видео",
                    scale=1,
                )
                live_audio_lang = gr.Dropdown(
                    choices=["Русский", "English", "Автоопределение"],
                    value="Русский",
                    label="Язык речи (Whisper)",
                    scale=1,
                )

            with gr.Row(equal_height=True):
                ai_interval = gr.Slider(
                    minimum=2, maximum=20, value=5, step=1, label="Интервал ИИ (сек)", scale=1
                )
                max_frames = gr.Slider(
                    minimum=10, maximum=3600, value=1200, step=10, label="Лимит кадров", scale=1
                )
                vlm_interval = gr.Slider(
                    minimum=10,
                    maximum=120,
                    value=20,
                    step=5,
                    label="Интервал VLM (сек)",
                    scale=1,
                )
                live_lat = gr.Textbox(
                    label="Телеметрия потока и FPS",
                    value="Ожидание запуска...",
                    lines=1,
                    interactive=False,
                    scale=2,
                )

            with gr.Row(equal_height=True):
                btn_start = gr.Button(
                    "Запустить прямой эфир", variant="primary", scale=3, size="lg"
                )
                btn_stop = gr.Button("Остановить", variant="stop", scale=1, size="lg")

        with gr.Row():
            # Левая колонка: Эфир, сенсоры и диалог с ИИ
            with gr.Column(scale=2, min_width=520):
                with gr.Group():
                    live_frame = gr.Image(label="Видеопоток с детекцией (YOLO26s)", height=380)
                    live_visual = gr.Textbox(label="Объекты (YOLO26s) и описание сцены", lines=1)
                    live_vlm = gr.Textbox(label="Детали кадра и HUD (VLM)", lines=2)

                    with gr.Row(equal_height=True):
                        live_audio = gr.Audio(
                            label="Звук стрима", type="filepath", interactive=False, scale=2
                        )
                        live_speech = gr.Textbox(
                            label="Речь стримера (Whisper Small)",
                            value="Ожидание речи стримера...",
                            lines=2,
                            scale=3,
                        )

                    live_bot = gr.Textbox(label="Реакция ИИ-бота в чат (модель Ollama)", lines=1)

                with gr.Group():
                    gr.Markdown("#### Диалог со стрим-ассистентом (модель Ollama)")
                    stream_chatbot = gr.Chatbot(
                        label="Диалог со стрим-ассистентом (модель Ollama)",
                        show_label=False,
                        height=450,
                        placeholder="Здесь можно спросить о том что происходит на трансляции",
                    )

                    with gr.Row(equal_height=True):
                        user_question_in = gr.Textbox(
                            placeholder="Задайте вопрос о трансляции",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        btn_ask = gr.Button("Спросить", variant="primary", scale=1)

                    with gr.Row(equal_height=True):
                        quick_q1 = gr.Button("О чем стрим?", size="sm", scale=1)
                        quick_q2 = gr.Button("Что сказал стример?", size="sm", scale=1)
                        quick_q3 = gr.Button("Что на экране?", size="sm", scale=1)
                        quick_q4 = gr.Button("Настроение в чате?", size="sm", scale=1)

            # Правая колонка: Живой чат
            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    gr.Markdown("#### Живой чат Twitch (IRC)")
                    live_chat = gr.HTML(
                        value=make_chat_status_html(
                            "Чат пока не запущен. Нажмите 'Запустить прямой эфир'.",
                            "#71717a",
                            max_height=860,
                        ),
                        label="Живой чат Twitch (IRC)",
                    )
                    live_mood = gr.Textbox(label="Тональность чата (RuBERT)", lines=1)

        # =====================================================================
        # ОБРАБОТЧИКИ СОБЫТИЙ И СВЯЗКИ КОМПОНЕНТОВ
        # =====================================================================
        stream_job = btn_start.click(
            stream_live_twitch,
            inputs=[
                live_channel,
                live_quality,
                ai_interval,
                max_frames,
                live_audio_lang,
                vlm_interval,
            ],
            outputs=[
                live_frame,
                live_chat,
                live_speech,
                live_audio,
                live_visual,
                live_mood,
                live_bot,
                live_lat,
                live_vlm,
            ],
        )

        btn_stop.click(fn=None, cancels=[stream_job])

        btn_ask.click(
            handle_stream_question,
            inputs=[user_question_in, stream_chatbot, live_channel],
            outputs=[stream_chatbot, user_question_in],
        )
        user_question_in.submit(
            handle_stream_question,
            inputs=[user_question_in, stream_chatbot, live_channel],
            outputs=[stream_chatbot, user_question_in],
        )

        quick_q1.click(
            lambda h, c: handle_stream_question("О чем этот стрим и какая игра идет?", h, c),
            inputs=[stream_chatbot, live_channel],
            outputs=[stream_chatbot, user_question_in],
        )
        quick_q2.click(
            lambda h, c: handle_stream_question("Что только что говорил стример в микрофон?", h, c),
            inputs=[stream_chatbot, live_channel],
            outputs=[stream_chatbot, user_question_in],
        )
        quick_q3.click(
            lambda h, c: handle_stream_question(
                "Что сейчас происходит и какие объекты видны на экране?", h, c
            ),
            inputs=[stream_chatbot, live_channel],
            outputs=[stream_chatbot, user_question_in],
        )
        quick_q4.click(
            lambda h, c: handle_stream_question(
                "Какое настроение у чата и как зрители реагируют на стрим?", h, c
            ),
            inputs=[stream_chatbot, live_channel],
            outputs=[stream_chatbot, user_question_in],
        )
