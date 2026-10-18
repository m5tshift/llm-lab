"""Generate Grafana dashboards for application and model metrics."""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "Prometheus"}
WEB = 'job="twitch_bot_web"'


def panel(title, expr, unit="short", legend="{{job}}", description="", stat=False):
    return {
        "title": title,
        "type": "stat" if stat else "timeseries",
        "datasource": DS,
        "description": description,
        "fieldConfig": {"defaults": {"unit": unit, "noValue": "Нет данных"}, "overrides": []},
        "options": (
            {
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "colorMode": "value",
                "graphMode": "none",
            }
            if stat
            else {"legend": {"displayMode": "list", "placement": "bottom"}}
        ),
        "targets": [
            {
                "refId": "A",
                "expr": expr,
                "legendFormat": legend,
                "datasource": DS,
                "range": not stat,
                "instant": stat,
            }
        ],
    }


def dashboards():
    stream = [
        panel("Эфир активен (1/0)", f"twitch_stream_active{{{WEB}}}", stat=True),
        panel(
            "Длительность текущей сессии",
            f"twitch_session_duration_seconds{{{WEB}}}",
            "s",
            stat=True,
        ),
        panel(
            "Обработанные кадры/с",
            f"rate(twitch_frames_processed_total{{{WEB}}}[1m])",
            "fps",
            description="Реальная пропускная способность, включая ожидание UI/сети. Не FPS источника Twitch.",
        ),
        panel(
            "Полученные сообщения/мин",
            f"rate(twitch_chat_received_messages_total{{{WEB}}}[1m]) * 60",
            description="Считаются на входе IRC; повторный анализ окна не увеличивает счётчик.",
        ),
        panel(
            "Реакции и ответы/мин",
            f"sum by (job) (rate(twitch_bot_replies_total{{{WEB}}}[1m])) * 60",
            description="Сгенерированные ответы в интерфейсе, не отправленные в Twitch сообщения.",
        ),
        panel(
            "Позитивная доля последнего окна", f"twitch_chat_positive_ratio{{{WEB}}}", "percentunit"
        ),
        panel(
            "Негативная доля последнего окна", f"twitch_chat_negative_ratio{{{WEB}}}", "percentunit"
        ),
        panel(
            "Возраст кадра последнего VLM-результата",
            f"time() - twitch_vlm_frame_timestamp_seconds{{{WEB}}}",
            "s",
            description="От момента передачи кадра на анализ; включает время инференса. Нет данных до первого результата.",
        ),
        panel(
            "Объектов в последнем кадре",
            f"twitch_video_latest_objects_count{{{WEB}}}",
            description="Детекции YOLO, не число уникальных объектов за эфир.",
        ),
        panel(
            "Фоновое обновление AI: среднее время",
            "sum by (mode) (rate(twitch_ai_update_duration_seconds_sum[5m])) / sum by (mode) (rate(twitch_ai_update_duration_seconds_count[5m]))",
            "s",
            "{{mode}}",
            "live: NLP/описание/реакция. Параллельные YOLO, Whisper и VLM сюда не входят; pipeline: последовательный CLI-цикл.",
        ),
    ]
    models = [
        panel(
            "Последняя задержка модели",
            "twitch_model_latest_latency_ms",
            "ms",
            "{{job}} / {{model}}",
        ),
        panel(
            "Средняя задержка модели",
            "sum by (job, model) (rate(twitch_model_latency_seconds_sum[5m])) / sum by (job, model) (rate(twitch_model_latency_seconds_count[5m]))",
            "s",
            "{{job}} / {{model}}",
        ),
        panel(
            "Успешные измеренные вызовы/мин",
            "sum by (job, model) (rate(twitch_model_latency_seconds_count[5m])) * 60",
            legend="{{job}} / {{model}}",
        ),
        panel(
            "Ошибки фоновой обработки/мин",
            "sum by (component) (rate(twitch_processing_errors_total[5m])) * 60",
            legend="{{component}}",
        ),
        panel(
            "Запросы Ollama/мин",
            "sum by (task, status) (rate(twitch_ollama_requests_total[5m])) * 60",
            legend="{{task}} / {{status}}",
        ),
        panel(
            "Скорость генерации Ollama",
            "twitch_llm_generation_speed_tokens_per_second",
            "short",
            "{{job}} / {{model}} / {{task}}",
            "Выходные токены / время декодирования. Без ожидания очереди и обработки входа.",
        ),
        panel(
            "Токены Ollama/мин",
            'sum by (task, direction) (rate(twitch_llm_tokens_total{task=~"text|vision"}[5m])) * 60',
            legend="{{task}} / {{direction}}",
            description="Текстовые и визуальные запросы. Входные токены — значение, сообщённое Ollama.",
        ),
        panel(
            "Запросы API/с",
            "sum by (endpoint, status_code) (rate(twitch_api_requests_total[5m]))",
            legend="{{endpoint}} / {{status_code}}",
        ),
        panel(
            "Среднее время API",
            "sum by (endpoint) (rate(twitch_api_request_duration_seconds_sum[5m])) / sum by (endpoint) (rate(twitch_api_request_duration_seconds_count[5m]))",
            "s",
            "{{endpoint}}",
        ),
    ]
    infra = [
        panel(
            "Доступность экспортёров (1/0)",
            "up",
            legend="{{job}}",
            stat=True,
            description="Доступность сбора метрик API и веб-приложения.",
        ),
        panel("Доступность Ollama (1/0)", f"twitch_ollama_up{{{WEB}}}", stat=True),
        panel(
            "Модели в памяти Ollama",
            f"twitch_ollama_model_loaded{{{WEB}}}",
            legend="{{model}}",
            stat=True,
            description="Наличие в памяти Ollama не означает полное размещение на GPU.",
        ),
        panel(
            "Видеопамять Ollama",
            f"twitch_ollama_vram_usage_bytes{{{WEB}}}",
            "bytes",
            "{{model}}",
            "Выделение памяти модели по /api/ps; это не загрузка GPU и не вся занятая VRAM.",
        ),
        panel(
            "RAM процессов (RSS)",
            'process_resident_memory_bytes{job=~"twitch_bot_api|twitch_bot_web"}',
            "bytes",
        ),
        panel(
            "CPU процессов (100% = одно ядро)",
            'rate(process_cpu_seconds_total{job=~"twitch_bot_api|twitch_bot_web"}[1m]) * 100',
            "percent",
        ),
        panel("Потоки процессов", "twitch_process_threads_count"),
    ]
    return [
        ("01_business_metrics.json", "twitch-business", "1. Прямой эфир", stream),
        ("02_model_metrics.json", "twitch-models", "2. Модели и API", models),
        (
            "03_infrastructure_metrics.json",
            "twitch-infrastructure",
            "3. Ресурсы и доступность",
            infra,
        ),
    ]


def build():
    out = Path(__file__).resolve().parents[1] / "monitoring/grafana/dashboards"
    out.mkdir(parents=True, exist_ok=True)
    for filename, uid, title, panels in dashboards():
        for i, item in enumerate(panels):
            item.update(id=i + 1, gridPos={"x": (i % 2) * 12, "y": (i // 2) * 7, "w": 12, "h": 7})
        document = {
            "uid": uid,
            "title": title,
            "schemaVersion": 39,
            "version": 2,
            "refresh": "5s",
            "timezone": "browser",
            "time": {"from": "now-15m", "to": "now"},
            "tags": ["twitch", "operational"],
            "panels": panels,
            "editable": True,
        }
        (out / filename).write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(filename)


if __name__ == "__main__":
    build()
