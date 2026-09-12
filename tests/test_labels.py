import csv

import pytest

FIELDNAMES = ["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment",
              "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"]


def _row(dia, utt, text, emotion="neutral", sentiment="neutral", speaker="Chandler"):
    return {"Sr No.": "1", "Utterance": text, "Speaker": speaker, "Emotion": emotion,
            "Sentiment": sentiment, "Dialogue_ID": str(dia), "Utterance_ID": str(utt),
            "Season": "1", "Episode": "1", "StartTime": "00:00:00,000", "EndTime": "00:00:01,000"}


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_load_split_parses_rows_into_utterances(tmp_path):
    from meld_emotion.data.labels import load_split
    _write_csv(tmp_path / "dev_sent_emo.csv", [
        _row(0, 0, "Hello there", emotion="joy", sentiment="positive"),
        _row(0, 1, "Not really", emotion="anger", sentiment="negative"),
    ])
    utterances = load_split("dev", labels_dir=tmp_path)
    assert len(utterances) == 2
    assert utterances[0].text == "Hello there"
    assert utterances[0].emotion == "joy"
    assert utterances[0].sentiment == "positive"
    assert utterances[0].split == "dev"
    assert utterances[1].dialogue_id == 0
    assert utterances[1].utterance_id == 1


def test_load_split_rejects_an_unknown_emotion_label(tmp_path):
    from meld_emotion.data.labels import load_split
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 0, "hi", emotion="confused")])
    with pytest.raises(ValueError, match="confused"):
        load_split("dev", labels_dir=tmp_path)


def test_context_window_returns_previous_k_plus_current(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue, context_window
    _write_csv(tmp_path / "dev_sent_emo.csv", [
        _row(0, 0, "line zero"), _row(0, 1, "line one"),
        _row(0, 2, "line two"), _row(0, 3, "line three"),
    ])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert context_window(by_dialogue, dialogue_id=0, utterance_id=3, k=2) == [
        "line one", "line two", "line three"
    ]


def test_context_window_clamps_at_dialogue_start(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue, context_window
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 0, "line zero"), _row(0, 1, "line one")])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert context_window(by_dialogue, dialogue_id=0, utterance_id=1, k=4) == ["line zero", "line one"]


def test_context_window_k_zero_returns_only_current(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue, context_window
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 0, "line zero"), _row(0, 1, "line one")])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert context_window(by_dialogue, dialogue_id=0, utterance_id=1, k=0) == ["line one"]


def test_group_by_dialogue_sorts_by_utterance_id(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 2, "second"), _row(0, 0, "first"), _row(0, 1, "middle")])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert [u.text for u in by_dialogue[0]] == ["first", "middle", "second"]
