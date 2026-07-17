from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WhisperService:
    model_name: str = "base"
    language: str = "en"
    model: object | None = None

    def initialize(self):
        if self.model is not None:
            return
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("openai-whisper is required for speech-to-text") from exc
        self.model = whisper.load_model(self.model_name)
        logger.info("Loaded Whisper %s model", self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if sample_rate != 16000:
            raise ValueError("WhisperService expects 16kHz preprocessed audio")
        if self.model is None:
            self.initialize()
        use_fp16 = False
        try:
            import torch

            use_fp16 = torch.cuda.is_available()
        except ImportError:
            use_fp16 = False
        result = self.model.transcribe(
            audio.astype("float32"),
            fp16=use_fp16,
            language=self.language,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            beam_size=3,
            best_of=3,
            verbose=False,
            initial_prompt=(
                "The speaker is a sales lead talking to an AI call assistant in English. "
                "Common words include pricing, live demo, budget, timeline, implementation, "
                "security, automation, enterprise, decision maker, hot lead, warm lead, and cold lead."
            ),
        )
        return str(result.get("text", "")).strip()
