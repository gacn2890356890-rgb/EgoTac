"""Sequence planning and batch math for full runs. No GPU, data, or official code required."""
import pytest

from egotac.config import accumulation_steps, load_config
from egotac.data.splits import PAPER_TABLE_7, cap_test_rows, plan_splits


def test_oakink_plan_matches_table_7():
    names = [f"seq{i}" for i in range(627)]
    plan = plan_splits("oakink2", names)
    assert sum(v == "train" for v in plan.values()) == 565
    assert sum(v == "test" for v in plan.values()) == 62
    assert plan == plan_splits("oakink2", list(reversed(names))), "must not depend on listing order"


def test_taco_plan_selects_paper_size_and_ignores_the_rest():
    plan = plan_splits("taco", [f"seq{i}" for i in range(2311)])
    assert sum(v == "train" for v in plan.values()) == 839
    assert sum(v == "test" for v in plan.values()) == 120


def test_partial_download_still_holds_out_sequences():
    plan = plan_splits("taco", [f"seq{i}" for i in range(40)])
    assert len(plan) == 40 and 1 <= sum(v == "test" for v in plan.values()) < 40


def test_test_clips_are_capped_and_training_clips_are_not():
    rows = [{"dataset": "taco", "split": "test", "path": f"t{i}"} for i in range(900)]
    rows += [{"dataset": "taco", "split": "train", "path": f"r{i}"} for i in range(900)]
    kept = cap_test_rows(rows)
    assert sum(r["split"] == "test" for r in kept) == PAPER_TABLE_7["taco"][1]
    assert sum(r["split"] == "train" for r in kept) == 900


@pytest.mark.parametrize("batch,world,expected", [(8, 8, 1), (2, 8, 4), (8, 1, 8), (3, 4, 6)])
def test_accumulation_reaches_the_total_batch(batch, world, expected):
    assert accumulation_steps({"batch_size": batch, "total_batch_size": 64}, world) == expected


def test_config_expands_environment_and_rejects_unset(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    path.write_text('[a]\nb = "${EGOTAC_TEST_ROOT}/x"\n')
    with pytest.raises(ValueError, match="EGOTAC_TEST_ROOT"):
        load_config(path, [])
    monkeypatch.setenv("EGOTAC_TEST_ROOT", "/r")
    assert load_config(path, [])["a"]["b"] == "/r/x"
    assert load_config(path, ['a.b="/o"', "t.n=3"]) == {"a": {"b": "/o"}, "t": {"n": 3}}
