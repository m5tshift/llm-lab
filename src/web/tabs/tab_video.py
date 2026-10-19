"""Tab 4: Real-time Video Object Detection (YOLO26s)."""

import gradio as gr
from src.web.state import web_state
from src.monitoring.decorators import safe_telemetry_dispatch
from src.metrics import record_model_latency, record_detected_objects


def handle_frame_detect(frame):
    if frame is None:
        return None, "Кадр не передан"
    if not web_state.video_model:
        return frame, "YOLO модель не инициализирована"
    res = web_state.video_model.process_frame(frame)
    lat_ms = res.latency_ms if hasattr(res, "latency_ms") else res.get("latency_ms", 0.0)
    annotated = (
        res.annotated_frame if hasattr(res, "annotated_frame") else res.get("annotated_frame")
    )
    detected = (
        res.detected_objects
        if hasattr(res, "detected_objects")
        else res.get("detected_objects", [])
    )

    safe_telemetry_dispatch(record_model_latency, "video_detector", lat_ms / 1000.0)
    safe_telemetry_dispatch(record_detected_objects, detected)

    obj_labels = []
    for o in detected:
        cls_name = o.class_name if hasattr(o, "class_name") else o.get("class", "")
        conf = o.confidence if hasattr(o, "confidence") else o.get("confidence", 0)
        obj_labels.append(f"{cls_name} ({conf * 100:.0f}%)")

    stats = f"Найдено объектов: {len(detected)} [{', '.join(obj_labels)}] | Задержка: {lat_ms} ms"
    return annotated, stats


def create_video_tab():
    with gr.TabItem("4. Видео (YOLO26s)"):
        gr.Markdown(
            "### 4. Детекция объектов на кадре стрима / веб-камеры (Ultralytics YOLO26 — yolo26s.pt)"
        )
        gr.Markdown(
            "YOLO26s возвращает рамки, классы объектов и оценки уверенности. Сравните разметку с исходным кадром: игровые объекты могут не соответствовать классам модели. Задержка измеряется при обработке кадра и зависит от оборудования и входных данных."
        )
        with gr.Row():
            with gr.Column():
                frame_in = gr.Image(sources=["upload", "webcam"], label="Кадр для детекции")
                btn_detect = gr.Button("Детектировать объекты", variant="primary")
            with gr.Column():
                frame_out = gr.Image(label="Размеченный кадр (Bounding Boxes)")
                frame_stats = gr.Textbox(label="Статистика детекции")

        btn_detect.click(handle_frame_detect, inputs=[frame_in], outputs=[frame_out, frame_stats])
