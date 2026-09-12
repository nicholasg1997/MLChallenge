"""Loading MELD's per-split label CSVs and building dialogue-context text."""
import csv
from dataclasses import dataclass
from pathlib import Path

EMOTIONS = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")
SENTIMENTS = ("neutral", "positive", "negative")


@dataclass(frozen=True)
class Utterance:
    split: str
    dialogue_id: int
    utterance_id: int
    speaker: str
    text: str
    emotion: str
    sentiment: str


def load_split(split: str, labels_dir: Path) -> list[Utterance]:
    path = labels_dir / f"{split}_sent_emo.csv"
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Emotion"] not in EMOTIONS:
                raise ValueError(f"{path.name}: unknown emotion {row['Emotion']!r} "
                                 f"(dia{row['Dialogue_ID']}_utt{row['Utterance_ID']})")
            if row["Sentiment"] not in SENTIMENTS:
                raise ValueError(f"{path.name}: unknown sentiment {row['Sentiment']!r} "
                                 f"(dia{row['Dialogue_ID']}_utt{row['Utterance_ID']})")
            out.append(Utterance(
                split=split,
                dialogue_id=int(row["Dialogue_ID"]),
                utterance_id=int(row["Utterance_ID"]),
                speaker=row["Speaker"],
                text=row["Utterance"],
                emotion=row["Emotion"],
                sentiment=row["Sentiment"],
            ))
    return out


def group_by_dialogue(utterances: list[Utterance]) -> dict[int, list[Utterance]]:
    by_dialogue: dict[int, list[Utterance]] = {}
    for u in utterances:
        by_dialogue.setdefault(u.dialogue_id, []).append(u)
    for group in by_dialogue.values():
        group.sort(key=lambda u: u.utterance_id)
    return by_dialogue


def context_window(by_dialogue: dict[int, list[Utterance]], dialogue_id: int,
                   utterance_id: int, k: int) -> list[str]:
    """Up to k previous utterance texts (oldest first) followed by the current one."""
    dialogue = by_dialogue[dialogue_id]
    idx = next(i for i, u in enumerate(dialogue) if u.utterance_id == utterance_id)
    start = max(0, idx - k)
    return [u.text for u in dialogue[start:idx + 1]]
