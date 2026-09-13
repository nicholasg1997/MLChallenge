import json


def test_sample_response_rows_reads_done_events_and_adds_empty_rubric_fields(tmp_path):
    from scripts.response_rubric import sample_response_rows
    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"turn_id": f"dia{i}_utt0", "phase": "done", "response": f"response {i}"}) for i in range(5)]
    path.write_text("\n".join(lines) + "\n")
    rows = sample_response_rows(path, n=3, seed=0)
    assert len(rows) == 3
    for row in rows:
        assert set(row) == {"turn_id", "prompt", "response", "tag_consistent", "uses_visual_cue", "concise",
                            "in_character", "no_invented_facts"}
        assert all(row[f] is None for f in ("tag_consistent", "uses_visual_cue", "concise", "in_character", "no_invented_facts"))


def test_sample_response_rows_is_capped_at_the_available_count(tmp_path):
    from scripts.response_rubric import sample_response_rows
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"turn_id": "dia0_utt0", "phase": "done", "response": "hi"}) + "\n")
    assert len(sample_response_rows(path, n=50, seed=0)) == 1
