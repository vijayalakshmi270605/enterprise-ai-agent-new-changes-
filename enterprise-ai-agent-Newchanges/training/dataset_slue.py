from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from app.config import settings


DATASET_NAME = settings.slue_voxceleb_dataset
SENTIMENT_LABELS = {"positive", "negative", "neutral"}
LABEL_NORMALIZATION = {}


@dataclass
class DatasetSplits:
    train: Any
    validation: Any
    test: Any


def load_slue_voxceleb(seed: int = 42) -> DatasetSplits:
    """Load the required DynamicSuperb SLUE-VoxCeleb dataset from Hugging Face.

    The user requirement is an 80/10/10 split. We use Hugging Face Datasets
    directly and never create a custom dataset.
    """

    from datasets import DatasetDict, concatenate_datasets, load_dataset

    dataset = load_dataset(DATASET_NAME)
    if isinstance(dataset, DatasetDict):
        if {"train", "validation", "test"}.issubset(dataset.keys()):
            combined = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        elif "train" in dataset:
            combined = dataset["train"]
        else:
            first_key = next(iter(dataset.keys()))
            combined = dataset[first_key]
    else:
        combined = dataset

    first = _split(combined, test_size=0.2, seed=seed)
    second = _split(first["test"], test_size=0.5, seed=seed)
    return DatasetSplits(
        train=first["train"],
        validation=second["train"],
        test=second["test"],
    )


def load_slue_voxceleb_audio(seed: int = 42) -> DatasetSplits:
    splits = load_slue_voxceleb(seed=seed)
    return DatasetSplits(
        train=splits.train.cast_column("audio", _audio_feature(decode=False)),
        validation=splits.validation.cast_column("audio", _audio_feature(decode=False)),
        test=splits.test.cast_column("audio", _audio_feature(decode=False)),
    )


def _audio_feature(decode: bool):
    from datasets import Audio

    return Audio(decode=decode)


def _label_column(dataset) -> str | None:
    for candidate in ("label", "sentiment", "emotion"):
        if candidate in dataset.column_names:
            return candidate
    return None


def _split(dataset, test_size: float, seed: int):
    label_column = _label_column(dataset)
    if label_column:
        try:
            return dataset.train_test_split(
                test_size=test_size,
                seed=seed,
                stratify_by_column=label_column,
            )
        except ValueError:
            pass
    return dataset.train_test_split(test_size=test_size, seed=seed)


def label_names(dataset) -> List[str]:
    column = _label_column(dataset)
    if column and hasattr(dataset.features[column], "names"):
        return list(dataset.features[column].names)
    values = sorted({str(value).lower() for value in dataset[column]}) if column else []
    return values


def record_label(record: Dict[str, Any], names: List[str] | None = None) -> str:
    value = record.get("label", record.get("sentiment", record.get("emotion")))
    if isinstance(value, int) and names and value < len(names):
        return normalize_sentiment_label(names[value])
    return normalize_sentiment_label(str(value))


def normalize_sentiment_label(label: str) -> str:
    normalized = label.strip().lower()
    return LABEL_NORMALIZATION.get(normalized, normalized)


def record_text(record: Dict[str, Any]) -> str:
    for key in ("sentence", "text", "transcript", "normalized_text", "utterance"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def record_audio(record: Dict[str, Any]) -> Tuple[Any, int]:
    audio = record.get("audio") or record.get("file") or record.get("speech")
    if isinstance(audio, dict):
        array = audio.get("array")
        sample_rate = int(audio.get("sampling_rate", settings.audio_sample_rate))
        return array, sample_rate
    return audio, settings.audio_sample_rate


def iter_text_examples(dataset) -> Tuple[List[str], List[str]]:
    names = label_names(dataset)
    text_column = next(
        (
            key
            for key in ("sentence", "text", "transcript", "normalized_text", "utterance")
            if key in dataset.column_names
        ),
        None,
    )
    label_column = _label_column(dataset)
    if text_column is None or label_column is None:
        return [], []
    texts: List[str] = []
    labels: List[str] = []
    for text, label_value in zip(dataset[text_column], dataset[label_column]):
        if isinstance(text, str) and text.strip():
            texts.append(text)
            if isinstance(label_value, int) and names and label_value < len(names):
                label = normalize_sentiment_label(names[label_value])
            else:
                label = normalize_sentiment_label(str(label_value))
            if label in SENTIMENT_LABELS:
                labels.append(label)
            else:
                texts.pop()
    return texts, labels
