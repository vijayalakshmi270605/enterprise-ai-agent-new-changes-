from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from app.config import settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    task: str
    cpu_fallback: bool


def production_model_registry() -> Dict[str, ModelSpec]:
    return {
        "text_sentiment": ModelSpec(
            name="tfidf_multinomial_naive_bayes",
            path=settings.text_sentiment_model_path,
            task="text_sentiment",
            cpu_fallback=True,
        ),
        "voice_emotion_cnn": ModelSpec(
            name="cnn_mfcc",
            path=settings.voice_emotion_model_path,
            task="voice_emotion",
            cpu_fallback=True,
        ),
        "voice_emotion_wav2vec2": ModelSpec(
            name="wav2vec2_audio_classification",
            path=settings.wav2vec_emotion_model_path,
            task="voice_emotion",
            cpu_fallback=False,
        ),
        "speech_to_text": ModelSpec(
            name=f"whisper_{settings.whisper_model}",
            path=settings.whisper_model,
            task="speech_to_text",
            cpu_fallback=True,
        ),
    }

