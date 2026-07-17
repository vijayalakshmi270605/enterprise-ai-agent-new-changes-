from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:  # pragma: no cover - plain terminal fallback
    Console = None
    Panel = None


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/v1"
SCORE_FIELDS = (
    "icp_score",
    "intent_score",
    "engagement_score",
    "qualification_score",
    "buying_signal_score",
    "relationship_score",
)


class LiveConsole:
    def __init__(self) -> None:
        self.rich = Console() if Console else None

    def print(self, message: str) -> None:
        if self.rich:
            self.rich.print(message)
        else:
            print(message)

    def panel(self, message: str, title: str) -> None:
        if self.rich and Panel:
            self.rich.print(Panel(message, title=title))
        else:
            print(f"\n[{title}]\n{message}\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def post_json(client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


async def post_audio(client: httpx.AsyncClient, url: str, audio_path: Path) -> dict[str, Any]:
    with audio_path.open("rb") as audio:
        files = {"file": (audio_path.name, audio, "audio/wav")}
        response = await client.post(url, files=files)
    response.raise_for_status()
    return response.json()


def print_scores(console: LiveConsole, scores: dict[str, Any]) -> None:
    lines = [f"{field.replace('_', ' ').title()}: {scores.get(field, 0):>6}" for field in SCORE_FIELDS]
    if "lead_score" in scores:
        lines.append(f"Lead Score: {scores.get('lead_score', 0):>6}")
    if scores.get("lead_category"):
        lines.append(f"Category: {scores['lead_category']}")
    if scores.get("recommendation"):
        lines.append(f"Next Action: {scores['recommendation']}")
    console.panel("\n".join(lines), "Live Lead Scores")


def record_microphone_chunk(seconds: float, sample_rate: int) -> Path:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Microphone mode needs sounddevice. Install it with: pip install sounddevice"
        ) from exc

    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    target = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    sf.write(target, audio, sample_rate)
    return target


def record_microphone_utterance(
    max_seconds: float,
    sample_rate: int,
    silence_seconds: float,
    min_seconds: float,
    energy_threshold: float,
) -> Path:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Microphone mode needs sounddevice. Install it with: pip install sounddevice"
        ) from exc

    block_seconds = 0.15
    block_size = max(1, int(sample_rate * block_seconds))
    max_blocks = max(1, int(max_seconds / block_seconds))
    silence_blocks_needed = max(1, int(silence_seconds / block_seconds))
    min_blocks = max(1, int(min_seconds / block_seconds))
    frames: list[np.ndarray] = []
    heard_voice = False
    silent_blocks = 0

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size) as stream:
        for block_index in range(max_blocks):
            block, _ = stream.read(block_size)
            mono = np.asarray(block, dtype=np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
            is_voice = rms >= energy_threshold

            if is_voice:
                heard_voice = True
                silent_blocks = 0
            elif heard_voice:
                silent_blocks += 1

            if heard_voice or block_index < min_blocks:
                frames.append(mono.copy())

            if heard_voice and block_index >= min_blocks and silent_blocks >= silence_blocks_needed:
                break

    audio = np.concatenate(frames) if frames else np.zeros(int(sample_rate * min_seconds), dtype=np.float32)
    target = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    sf.write(target, audio, sample_rate)
    return target


async def run_turn(
    client: httpx.AsyncClient,
    api_url: str,
    session_id: str,
    transcript: str,
    history: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    infer_payload = {
        "session_id": session_id,
        "prompt": transcript,
        "history": history[-10:],
        "use_rag": True,
        "stream": False,
    }
    score_payload = {
        "session_id": session_id,
        "text": transcript,
        "history": history[-10:],
        "metadata": {"channel": "terminal_live"},
    }
    infer_task = post_json(client, f"{api_url}/infer", infer_payload)
    score_task = post_json(client, f"{api_url}/lead-scoring", score_payload)
    infer_result, score_result = await asyncio.gather(infer_task, score_task)
    return infer_result.get("response", ""), score_result


async def main() -> int:
    parser = argparse.ArgumentParser(description="Live terminal client for the enterprise AI assistant.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--session-id", default="terminal-live")
    parser.add_argument("--mode", choices=("text", "mic"), default="text")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--mic-style", choices=("utterance", "fixed"), default="utterance")
    parser.add_argument("--silence-seconds", type=float, default=1.2)
    parser.add_argument("--min-record-seconds", type=float, default=1.5)
    parser.add_argument("--energy-threshold", type=float, default=0.008)
    args = parser.parse_args()

    console = LiveConsole()
    event_file = LIVE_DIR / f"{args.session_id}.jsonl"
    console.panel(
        f"Session: {args.session_id}\nEvent stream: {event_file}\nSay/type 'exit' to stop.",
        "Follei Live Terminal",
    )

    history: list[dict[str, str]] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        while True:
            try:
                if args.mode == "mic":
                    if args.mic_style == "utterance":
                        console.print(
                            "[cyan]Listening... speak normally. I will stop after your pause.[/cyan]"
                        )
                        audio_path = record_microphone_utterance(
                            max_seconds=args.chunk_seconds,
                            sample_rate=args.sample_rate,
                            silence_seconds=args.silence_seconds,
                            min_seconds=args.min_record_seconds,
                            energy_threshold=args.energy_threshold,
                        )
                    else:
                        console.print(f"[cyan]Listening for {args.chunk_seconds:.1f}s...[/cyan]")
                        audio_path = record_microphone_chunk(args.chunk_seconds, args.sample_rate)
                    stt = await post_audio(client, f"{args.api_url}/speech-to-text", audio_path)
                    audio_path.unlink(missing_ok=True)
                    transcript = stt.get("transcript", "").strip()
                    if not transcript:
                        console.print("[yellow]No clear speech detected.[/yellow]")
                        continue
                else:
                    transcript = input("You: ").strip()

                if transcript.lower() in {"exit", "quit", "stop"}:
                    append_event(event_file, {"type": "session_end", "timestamp": now_iso()})
                    break

                turn_started = time.perf_counter()
                console.panel(transcript, "Transcript")
                answer, scores = await run_turn(client, args.api_url.rstrip("/"), args.session_id, transcript, history)
                latency_ms = round((time.perf_counter() - turn_started) * 1000, 2)

                console.panel(answer, "Assistant")
                print_scores(console, scores)

                history.append({"role": "user", "content": transcript})
                history.append({"role": "assistant", "content": answer})
                append_event(
                    event_file,
                    {
                        "type": "turn",
                        "timestamp": now_iso(),
                        "session_id": args.session_id,
                        "transcript": transcript,
                        "assistant_response": answer,
                        "scores": scores,
                        "latency_ms": latency_ms,
                    },
                )
            except KeyboardInterrupt:
                append_event(event_file, {"type": "session_end", "timestamp": now_iso()})
                break
            except httpx.ConnectError:
                console.print("[red]API is not reachable. Start it with: .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload[/red]")
                return 1
            except Exception as exc:
                encoded = base64.b64encode(str(exc).encode("utf-8")).decode("ascii")
                append_event(event_file, {"type": "error", "timestamp": now_iso(), "error_b64": encoded})
                console.print(f"[red]Live turn failed: {exc}[/red]")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
