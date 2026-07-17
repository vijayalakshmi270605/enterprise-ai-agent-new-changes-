from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter

from app.config import settings
from app.feature_extraction.audio_features import AudioFeatureExtractor
from app.speech.preprocessing import SpeechPreprocessor
from training.dataset_slue import label_names, load_slue_voxceleb_audio, record_label
from training.train_text import _decode_audio_without_torchcodec


class MFCCCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(32 * 4 * 16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MFCCEmotionDataset(Dataset):
    def __init__(self, hf_dataset, labels: List[str], max_items: int | None = None):
        self.dataset = hf_dataset.select(range(min(max_items, len(hf_dataset)))) if max_items else hf_dataset
        self.labels = labels
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        raw_labels = [record_label(record, labels) for record in self.dataset]
        self.class_counts = Counter(raw_labels)
        self.preprocessor = SpeechPreprocessor(target_sr=settings.audio_sample_rate)
        self.extractor = AudioFeatureExtractor(sample_rate=settings.audio_sample_rate)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        audio, sample_rate = _decode_audio_without_torchcodec(record["audio"])
        audio = self.preprocessor.to_mono(audio)
        audio = self.preprocessor.resample(audio, sample_rate)
        audio = self.preprocessor.normalize(audio)
        mfcc = self.extractor.mfcc_tensor(audio)
        label = self.label_to_id[record_label(record, self.labels)]
        return torch.tensor(mfcc).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN emotion recognizer on MFCC features")
    parser.add_argument("--output", default=settings.voice_emotion_model_path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-items", type=int, default=None)
    args = parser.parse_args()

    splits = load_slue_voxceleb_audio()
    labels = label_names(splits.train)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = MFCCEmotionDataset(splits.train, labels, max_items=args.max_train_items)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = MFCCCNN(num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total = sum(train_dataset.class_counts.values())
    weights = [
        total / max(train_dataset.class_counts.get(label, 1), 1)
        for label in labels
    ]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    history = {"loss": [], "accuracy": []}
    for epoch in range(args.epochs):
        loss, accuracy = train_epoch(model, loader, optimizer, criterion, device)
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)
        print(f"epoch={epoch + 1} loss={loss:.4f} accuracy={accuracy:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "labels": labels, "history": history},
        output,
    )
    print(f"Saved CNN-MFCC emotion model to {output}")


if __name__ == "__main__":
    main()
