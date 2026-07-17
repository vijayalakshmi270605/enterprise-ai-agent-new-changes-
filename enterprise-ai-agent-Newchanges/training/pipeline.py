import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
from training.trainer import LoRATrainer
from app.config import settings

logger = logging.getLogger(__name__)


def run_training(train_data_path: str, output_dir: str = None):
    output_dir = output_dir or settings.lora_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer = LoRATrainer(train_path=train_data_path, output_dir=output_dir)
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning pipeline")
    parser.add_argument("--train-data", type=str, required=True, help="Path to instruction dataset JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Adapter output directory")
    args = parser.parse_args()
    run_training(args.train_data, args.output_dir)
