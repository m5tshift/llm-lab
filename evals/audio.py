"""Task 2 evaluation: Russian speech recognition (WER, CER, real-time factor, hallucinations).

Usage:
  uv run --locked --group eval python -m evals.audio --limit 200
  uv run --locked --group eval python -m evals.audio --manifest evals/local/audio/stream.csv
  ... --noise-dir evals/local/audio/noise     # clips without speech: music, game sounds, silence

Default data: google/fleurs, config ru_ru, test split (775 utterances, CC-BY-4.0). A manifest CSV
with columns path,text evaluates own recordings instead.
"""

import io
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from evals.common import (
    PeakMemory,
    base_parser,
    environment,
    latency_stats,
    read_manifest,
    save_result,
    subsample,
    timed,
    warmup,
)

SAMPLE_RATE = 16000
DEFAULT_MODELS = ["openai/whisper-small", "openai/whisper-large-v3-turbo"]


def normalize(text: str) -> str:
    """Lowercase, ё→е, punctuation removed, spaces collapsed: WER counts words, not formatting."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _decode(audio: Dict) -> np.ndarray:
    import librosa
    import soundfile as sf

    if audio.get("bytes"):
        array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    else:
        array, sr = sf.read(audio["path"], dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return (
        librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE) if sr != SAMPLE_RATE else array
    )


def load_hf(dataset: str, config: str, split: str, text_col: str) -> List[Dict]:
    from datasets import Audio, load_dataset

    # decode=False: audio is decoded here with soundfile, without the torchcodec backend.
    ds = load_dataset(dataset, config, split=split).cast_column("audio", Audio(decode=False))
    return [
        {"id": str(r.get("id", i)), "audio": r["audio"], "text": r[text_col]}
        for i, r in enumerate(ds)
    ]


def load_files(rows: List[Dict], text_key: str = "text") -> List[Dict]:
    return [
        {"id": Path(r["path"]).name, "audio": {"path": r["path"]}, "text": r.get(text_key, "")}
        for r in rows
    ]


def evaluate_model(
    model_name: str,
    items: List[Dict],
    noise: List[Dict],
    device: str,
    language: str,
    max_new_tokens: int,
) -> Dict:
    import jiwer

    from src.audio_transcribe import StreamAudioTranscriber

    with PeakMemory() as memory:
        transcriber, load_ms = timed(
            StreamAudioTranscriber, model_name, language, device, max_new_tokens
        )
        arrays = [_decode(it["audio"]) for it in items]
        warmup(lambda a: transcriber.transcribe({"array": a, "sampling_rate": SAMPLE_RATE}), arrays)
        outputs, latencies = [], []
        for array in arrays:
            res, ms = timed(transcriber.transcribe, {"array": array, "sampling_rate": SAMPLE_RATE})
            outputs.append(res)
            latencies.append(ms)
        noise_out = [
            transcriber.transcribe({"array": _decode(n["audio"]), "sampling_rate": SAMPLE_RATE})
            for n in noise
        ]

    refs = [normalize(it["text"]) for it in items]
    raw = [normalize(o.raw_transcription) for o in outputs]
    filtered = [normalize(o.transcription) for o in outputs]
    audio_s = sum(len(a) for a in arrays) / SAMPLE_RATE
    examples = [
        {
            "id": it["id"],
            "reference": it["text"],
            "raw": o.raw_transcription,
            "filtered": o.transcription,
        }
        for it, o in list(zip(items, outputs))[:20]
    ]
    metrics = {
        "wer_raw": round(jiwer.wer(refs, raw), 4),
        "cer_raw": round(jiwer.cer(refs, raw), 4),
        "wer_filtered": round(jiwer.wer(refs, filtered), 4),
        "empty_after_filter_rate": round(sum(not f for f in filtered) / len(filtered), 4),
        "rtf": round(sum(latencies) / 1000 / audio_s, 4),
        "audio_minutes": round(audio_s / 60, 1),
        "latency_per_utterance": latency_stats(latencies),
        "model_load_ms": round(load_ms, 1),
        "peak_rss_mb": round(memory.peak_mb, 1),
        "examples": examples,
    }
    if noise:
        metrics["noise_clips"] = len(noise)
        metrics["hallucination_rate_raw"] = round(
            sum(bool(o.raw_transcription.strip()) for o in noise_out) / len(noise), 4
        )
        metrics["hallucination_rate_filtered"] = round(
            sum(bool(o.transcription.strip()) for o in noise_out) / len(noise), 4
        )
        metrics["noise_examples"] = [
            o.raw_transcription for o in noise_out if o.raw_transcription.strip()
        ][:10]
    return metrics


def main(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", default="google/fleurs")
    parser.add_argument("--config", default="ru_ru")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="transcription")
    parser.add_argument("--manifest", type=Path, help="CSV path,text instead of the HF dataset")
    parser.add_argument(
        "--noise-dir", type=Path, help="Folder of .wav/.mp3/.flac clips without speech"
    )
    parser.add_argument("--language", default="russian")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Live app uses 64")
    args = parser.parse_args(argv)

    if args.manifest:
        items = load_files(read_manifest(args.manifest, ["path", "text"]))
        source = str(args.manifest)
    else:
        items = load_hf(args.dataset, args.config, args.split, args.text_column)
        source = f"{args.dataset}/{args.config}:{args.split}"
    items = subsample(items, args.limit, args.seed)
    noise = []
    if args.noise_dir:
        noise = load_files(
            [
                {"path": str(p)}
                for p in sorted(args.noise_dir.iterdir())
                if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
            ]
        )
    print(f"[audio] {len(items)} utterances from {source}; {len(noise)} noise clips")

    per_model = {}
    for model in args.models:
        print(f"[audio] Evaluating {model} ...")
        per_model[model] = evaluate_model(
            model, items, noise, args.device, args.language, args.max_new_tokens
        )
        m = per_model[model]
        print(f"[audio] {model}: WER={m['wer_raw']} CER={m['cer_raw']} RTF={m['rtf']}")

    main_model = args.models[0]
    payload = {
        "task": "audio",
        "environment": environment(args.device),
        "config": {
            "models": args.models,
            "dataset": source,
            "n": len(items),
            "language": args.language,
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "no_repeat_ngram_size": 3,
                "chunk_length_s": 30,
            },
            "normalization": "lowercase, ё→е, punctuation removed",
            "seed": args.seed,
        },
        "metrics": per_model,
        "summary": {
            "model": main_model,
            "main_metric": "WER (FLEURS ru test)"
            if not args.manifest
            else "WER (собственный набор)",
            "value": per_model[main_model]["wer_raw"],
            "latency_p95_ms": per_model[main_model]["latency_per_utterance"].get("p95_ms"),
            "peak_rss_mb": per_model[main_model]["peak_rss_mb"],
            "rtf": per_model[main_model]["rtf"],
        },
    }
    save_result("audio", payload, args.output_dir)


if __name__ == "__main__":
    main()
