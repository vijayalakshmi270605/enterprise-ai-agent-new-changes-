from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.fusion.engine import EmotionFusionEngine
from app.nlp.text_sentiment import TextSentimentPipeline
from app.services.tts_service import TTSService
from app.services.voice_emotion_service import VoiceEmotionService
from app.speech.preprocessing import SpeechPreprocessor
from app.speech.whisper_service import WhisperService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text/audio emotion prediction")
    parser.add_argument("--text", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--response-text", default=None, help="Optional assistant response text to synthesize.")
    parser.add_argument("--tts", action="store_true", help="Create a local WAV response with Kokoro or pyttsx3.")
    args = parser.parse_args()
    if not args.text and not args.audio:
        raise SystemExit("Provide --text or --audio")

    transcript = args.text or ""
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise SystemExit(
                f"Audio file not found: {audio_path}\n"
                "Use a real path, for example: python training/predict.py --audio C:\\path\\to\\your.wav"
            )
        audio_bytes = audio_path.read_bytes()
        preprocessor = SpeechPreprocessor(target_sr=settings.audio_sample_rate)
        audio, sample_rate = preprocessor.preprocess(audio_bytes)
        transcript = transcript or WhisperService(settings.whisper_model).transcribe(audio, sample_rate)
        voice = VoiceEmotionService.predict(audio)
    else:
        voice = None

    text_result = TextSentimentPipeline(settings.text_sentiment_model_path).load().predict(transcript)
    payload = {
        "transcript": transcript,
        "text_sentiment": text_result.__dict__,
    }
    if voice:
        fused = EmotionFusionEngine().fuse(text_result.label, text_result.confidence, voice.emotion, voice.confidence)
        payload["voice_emotion"] = voice.__dict__
        payload["final_emotion"] = fused.__dict__
    if args.tts:
        tts_text = args.response_text or f"I understood your emotion as {payload.get('final_emotion', {}).get('final_emotion', text_result.label)}."
        payload["tts"] = TTSService.synthesize(tts_text, output_dir=settings.tts_output_dir).to_dict()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
