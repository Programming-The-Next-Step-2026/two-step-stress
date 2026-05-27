"""Tests for the trial CSV writer and session JSON sidecar."""

import csv
import json

from two_step_stress.io.data_logging import TRIAL_COLUMNS, TrialWriter


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_csv_created_with_header(tmp_path):
    with TrialWriter("01", "20260525-120000", output_dir=tmp_path) as w:
        path = w.path
    assert path.exists()
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == list(TRIAL_COLUMNS)


def test_write_row_values(tmp_path):
    row = {
        "participant_id": "07",
        "block": 1,
        "block_type": "load",
        "trial_in_block": 3,
        "trial_global": 3,
        "stage1_choice": 0,
        "stage1_rt": 0.512,
        "transition": "common",
        "stage2_state": 0,
        "stage2_choice": 1,
        "stage2_rt": 0.733,
        "reward": 1,
        "reward_probs": [0.25, 0.5, 0.75, 0.6],
        "nback_letter": "B",
        "nback_is_match": False,
        "nback_response": "m",
        "nback_correct": True,
        "nback_rt": 0.44,
        "timestamp": "2026-05-25T12:00:00",
    }
    with TrialWriter("07", "ts", output_dir=tmp_path) as w:
        w.write_row(row)
        path = w.path

    rows = _read_rows(path)
    assert len(rows) == 1
    r = rows[0]
    assert r["participant_id"] == "07"
    assert r["block_type"] == "load"
    assert r["transition"] == "common"
    assert r["stage2_state"] == "0"
    assert r["reward"] == "1"
    assert r["stage1_rt"] == "0.512"
    assert json.loads(r["reward_probs"]) == [0.25, 0.5, 0.75, 0.6]
    assert r["nback_is_match"] == "False"
    assert r["nback_correct"] == "True"


def test_missing_fields_become_na(tmp_path):
    with TrialWriter("02", "ts", output_dir=tmp_path) as w:
        w.write_row({"participant_id": "02", "block": 2, "block_type": "no_load"})
        path = w.path

    r = _read_rows(path)[0]
    assert r["stage1_choice"] == "NA"
    assert r["nback_letter"] == "NA"
    assert r["reward_probs"] == "NA"


def test_no_overwrite_suffix(tmp_path):
    writers = [TrialWriter("01", "samets", output_dir=tmp_path) for _ in range(3)]
    names = [w.path.name for w in writers]
    for w in writers:
        w.close()

    assert names == [
        "sub-01_samets.csv",
        "sub-01_samets_1.csv",
        "sub-01_samets_2.csv",
    ]
    assert len(set(names)) == 3


def test_session_json_sidecar(tmp_path):
    with TrialWriter("01", "ts", output_dir=tmp_path) as w:
        json_path = w.write_session_json({"rng_seed": 12345, "frame_rate": 60.0})

    assert json_path.name == "sub-01_ts_session.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["rng_seed"] == 12345
    assert data["frame_rate"] == 60.0


def test_session_json_matches_suffixed_csv(tmp_path):
    first = TrialWriter("09", "dup", output_dir=tmp_path)
    second = TrialWriter("09", "dup", output_dir=tmp_path)  # collides → _1 suffix
    json_path = second.write_session_json({"ok": True})
    first.close()
    second.close()

    assert second.path.name == "sub-09_dup_1.csv"
    assert json_path.name == "sub-09_dup_1_session.json"
