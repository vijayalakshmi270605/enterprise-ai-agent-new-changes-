from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf

from training.dataset_slue import label_names, load_slue_voxceleb_audio, record_label
from training.train_text import _decode_audio_without_torchcodec


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a real audio sample from SLUE-VoxCeleb")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="./outputs/audio_samples/slue_sample.wav")
    args = parser.parse_args()

    splits = load_slue_voxceleb_audio()
    dataset = splits.test
    if args.index < 0 or args.index >= len(dataset):
        raise SystemExit(f"--index must be between 0 and {len(dataset) - 1}")

    record = dataset[args.index]
    audio, sample_rate = _decode_audio_without_torchcodec(record["audio"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sample_rate)
    names = label_names(dataset)
    print(f"Saved real dataset audio to {output}")
    print(f"Label: {record_label(record, names)}")
    print(f"Sample rate: {sample_rate}")


if __name__ == "__main__":
    main()

