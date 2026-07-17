from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.nlp.text_sentiment import TextSentimentPipeline
from training.dataset_slue import SENTIMENT_LABELS, iter_text_examples, label_names, load_slue_voxceleb, load_slue_voxceleb_audio, normalize_sentiment_label, record_label


def _decode_audio_without_torchcodec(audio_record):
    import numpy as np
    import soundfile as sf

    audio_bytes = audio_record.get("bytes")
    if audio_bytes:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    else:
        path = audio_record.get("path")
        audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def transcribe_training_audio(dataset, cache_path: str, limit: int | None = None) -> Tuple[List[str], List[str]]:
    from app.speech.whisper_service import WhisperService

    cache = Path(cache_path)
    if cache.exists():
        records = json.loads(cache.read_text(encoding="utf-8"))
        texts: List[str] = []
        labels: List[str] = []
        for item in records:
            label = normalize_sentiment_label(item["label"])
            if label in SENTIMENT_LABELS:
                texts.append(item["text"])
                labels.append(label)
        return texts, labels

    whisper = WhisperService(model_name=settings.whisper_model)
    names = label_names(dataset)
    records = []
    size = min(limit, len(dataset)) if limit else len(dataset)
    for index in range(size):
        record = dataset[index]
        audio, sample_rate = _decode_audio_without_torchcodec(record["audio"])
        transcript = whisper.transcribe(audio, sample_rate)
        if transcript.strip():
            label = record_label(record, names)
            if label not in SENTIMENT_LABELS:
                continue
            records.append(
                {
                    "text": transcript,
                    "label": label,
                    "source_index": index,
                }
            )
        if (index + 1) % 25 == 0:
            print(f"transcribed {index + 1}/{size} audio records")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return [item["text"] for item in records], [item["label"] for item in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + MultinomialNB on SLUE-VoxCeleb text")
    parser.add_argument("--output", default=settings.text_sentiment_model_path)
    parser.add_argument("--alpha", type=float, default=settings.sentiment_alpha)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transcribe-audio", action="store_true")
    parser.add_argument("--transcript-cache", default="./training/cache/slue_voxceleb_train_transcripts.json")
    parser.add_argument("--max-transcribe-items", type=int, default=None)
    parser.add_argument(
        "--use-class-prior",
        action="store_true",
        help="Use dataset class frequencies as priors. By default priors are balanced.",
    )
    args = parser.parse_args()

    splits = load_slue_voxceleb_audio(seed=args.seed) if args.transcribe_audio else load_slue_voxceleb(seed=args.seed)
    if args.transcribe_audio:
        train_texts, train_labels = transcribe_training_audio(
            splits.train,
            cache_path=args.transcript_cache,
            limit=args.max_transcribe_items,
        )
    else:
        train_texts, train_labels = iter_text_examples(splits.train)
    if not train_texts:
        raise RuntimeError(
            "No transcript/text field was found in the Hugging Face dataset. "
            "Run with --transcribe-audio to generate Whisper transcripts from the dataset audio."
        )
    pipeline = TextSentimentPipeline(args.output).train(
        train_texts,
        train_labels,
        alpha=args.alpha,
        fit_prior=args.use_class_prior,
    )
    pipeline.save(args.output)
    print(f"Saved TF-IDF MultinomialNB text sentiment model to {args.output}")
    print(TextSentimentPipeline.why_multinomial_nb())


if __name__ == "__main__":
    main()
