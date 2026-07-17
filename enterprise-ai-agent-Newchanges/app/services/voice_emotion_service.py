from __future__ import annotations

from app.config import settings
from app.emotion.voice_emotion import VoiceEmotionRecognizer, VoiceEmotionResult


class VoiceEmotionService:
    recognizer: VoiceEmotionRecognizer | None = None

    @classmethod
    def initialize(cls) -> None:
        if cls.recognizer is None:
            cls.recognizer = VoiceEmotionRecognizer(
                model_path=settings.voice_emotion_model_path,
                sample_rate=settings.audio_sample_rate,
            )
        cls.recognizer.initialize()

    @classmethod
    def predict(cls, audio) -> VoiceEmotionResult:
        if cls.recognizer is None:
            cls.initialize()
        return cls.recognizer.predict(audio)

