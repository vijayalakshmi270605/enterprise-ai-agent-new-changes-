from __future__ import annotations

import hashlib
import logging
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    engine: str
    status: str
    text: str
    audio_path: str | None = None
    audio_url: str | None = None
    message: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class TTSService:
    """Local TTS wrapper.

    Kokoro is used when a compatible local package is installed. pyttsx3 is used
    as an offline Windows-friendly fallback. If neither package is installed, the
    service returns a clear unavailable result instead of a fake audio response.
    """

    @classmethod
    def synthesize(cls, text: str, output_dir: str = "./outputs/tts") -> TTSResult:
        clean_text = text.strip()
        if not clean_text:
            return TTSResult(
                engine="none",
                status="skipped",
                text=text,
                message="No text was provided for TTS.",
            )

        output_path = cls._output_path(clean_text, output_dir)
        kokoro = cls._try_kokoro(clean_text, output_path)
        if kokoro.status == "ok":
            return kokoro

        pyttsx3 = cls._try_pyttsx3(clean_text, output_path)
        if pyttsx3.status == "ok":
            return pyttsx3

        return TTSResult(
            engine="unavailable",
            status="unavailable",
            text=clean_text,
            message=(
                "No local TTS engine is installed. Install Kokoro or pyttsx3, "
                "then rerun voice-chat or training/predict.py with --tts."
            ),
        )

    @staticmethod
    def _output_path(text: str, output_dir: str) -> Path:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"tts_{digest}.wav"

    @staticmethod
    def _try_kokoro(text: str, output_path: Path) -> TTSResult:
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            return TTSResult(engine="kokoro", status="unavailable", text=text)

        model_path = getattr(settings, "kokoro_model_path", "")
        voices_path = getattr(settings, "kokoro_voices_path", "")
        if not model_path or not voices_path:
            return TTSResult(
                engine="kokoro",
                status="unavailable",
                text=text,
                message="KOKORO_MODEL_PATH and KOKORO_VOICES_PATH are required for kokoro_onnx.",
            )
        try:
            kokoro = Kokoro(model_path, voices_path)
            samples, sample_rate = kokoro.create(text, voice=getattr(settings, "kokoro_voice", "af_sarah"))
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            return TTSResult(
                engine="kokoro",
                status="ok",
                text=text,
                audio_path=str(output_path),
                audio_url=_output_url(output_path),
            )
        except Exception as exc:
            logger.warning("Kokoro TTS failed: %s", exc)
            return TTSResult(engine="kokoro", status="failed", text=text, message=str(exc))

    @staticmethod
    def _try_pyttsx3(text: str, output_path: Path) -> TTSResult:
        try:
            import pyttsx3
        except ImportError:
            return TTSResult(engine="pyttsx3", status="unavailable", text=text)

        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            if output_path.exists() and output_path.stat().st_size > 0:
                return TTSResult(
                    engine="pyttsx3",
                    status="ok",
                    text=text,
                    audio_path=str(output_path),
                    audio_url=_output_url(output_path),
                )
            return TTSResult(engine="pyttsx3", status="failed", text=text, message="No audio file was created.")
        except Exception as exc:
            logger.warning("pyttsx3 TTS failed: %s", exc)
            return TTSResult(engine="pyttsx3", status="failed", text=text, message=str(exc))


def _output_url(path: Path) -> str:
    normalized = path.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return f"/{normalized}"
