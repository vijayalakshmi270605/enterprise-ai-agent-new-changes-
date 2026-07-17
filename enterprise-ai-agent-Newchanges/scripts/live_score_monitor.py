from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - plain terminal fallback
    Console = None
    Live = None
    Panel = None
    Table = None


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
SCORE_FIELDS = (
    "icp_score",
    "intent_score",
    "engagement_score",
    "qualification_score",
    "buying_signal_score",
    "relationship_score",
)


def read_latest_turn(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn":
                latest = event
    return latest


def build_rich_view(event: dict[str, Any] | None, path: Path):
    if event is None:
        return Panel(f"Waiting for events...\n{path}", title="Live Lead Scores")

    scores = event.get("scores", {})
    table = Table(title=f"Session: {event.get('session_id', 'terminal-live')}")
    table.add_column("Signal")
    table.add_column("Score", justify="right")
    for field in SCORE_FIELDS:
        table.add_row(field.replace("_", " ").title(), str(scores.get(field, 0)))
    table.add_row("Lead Score", str(scores.get("lead_score", 0)))
    table.add_row("Conversion %", str(scores.get("conversion_probability_percent", 0)))
    table.add_row("Category", str(scores.get("lead_category", "")))

    transcript = event.get("transcript", "")
    recommendation = scores.get("recommendation", "")
    return Panel(
        table,
        title="Live Lead Scores",
        subtitle=f"Latest: {transcript[:90]} | Next: {recommendation[:90]}",
    )


def print_plain(event: dict[str, Any] | None, path: Path) -> None:
    if event is None:
        print(f"Waiting for events at {path}")
        return
    scores = event.get("scores", {})
    print("\033c", end="")
    print(f"Session: {event.get('session_id', 'terminal-live')}")
    print(f"Transcript: {event.get('transcript', '')}")
    for field in SCORE_FIELDS:
        print(f"{field.replace('_', ' ').title()}: {scores.get(field, 0)}")
    print(f"Lead Score: {scores.get('lead_score', 0)}")
    print(f"Category: {scores.get('lead_category', '')}")
    print(f"Next Action: {scores.get('recommendation', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch live lead scores from the terminal client.")
    parser.add_argument("--session-id", default="terminal-live")
    parser.add_argument("--refresh-seconds", type=float, default=0.5)
    args = parser.parse_args()

    event_file = LIVE_DIR / f"{args.session_id}.jsonl"
    if Console and Live:
        console = Console()
        with Live(build_rich_view(read_latest_turn(event_file), event_file), console=console, refresh_per_second=4) as live:
            while True:
                try:
                    live.update(build_rich_view(read_latest_turn(event_file), event_file))
                    time.sleep(args.refresh_seconds)
                except KeyboardInterrupt:
                    return 0

    while True:
        try:
            print_plain(read_latest_turn(event_file), event_file)
            time.sleep(args.refresh_seconds)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
