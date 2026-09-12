"""Per-split video file indexing.

(Dialogue_ID, Utterance_ID) is NOT a unique key across MELD's splits -- IDs
restart at 0 in each split. build_video_index must always be called with ONE
split's own directory. Never merge indices from multiple splits.
"""
from pathlib import Path


def build_video_index(split_dir: Path) -> dict[tuple[int, int], Path]:
    index = {}
    for p in split_dir.glob("dia*_utt*.*"):
        try:
            dia_part, utt_part = p.stem.split("_")
            key = (int(dia_part.replace("dia", "")), int(utt_part.replace("utt", "")))
        except ValueError:
            continue
        index[key] = p
    return index
