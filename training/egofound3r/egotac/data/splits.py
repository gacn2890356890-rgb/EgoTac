"""Sequence selection at the data sizes of EgoFound3R Table 7.

The table gives counts, not names, so the sequences themselves are our deterministic choice.
TACO publishes 2317 sequences against the paper's 839 training sequences: the authors' selection rule
is unpublished, so this choice can differ from theirs. Test sequences are chosen first, then capped to
the table's evaluation clips by the manifest.
"""
from __future__ import annotations

import hashlib

# dataset: (training sequences, evaluation clips), from Table 7.
PAPER_TABLE_7 = {
    "oakink2": (565, 400),
    "taco": (839, 400),
    "hoi4d": (1431, 461),
    "h2o": (138, 283),
    "hot3d_aria": (136, 400),
    "arctic": (226, 434),
}
# Enough sequences to fill the evaluation clips. OakInk-v2 has 627 in all, which leaves exactly 62.
TEST_SEQUENCES = {"oakink2": 62, "taco": 120, "hoi4d": 70, "hot3d_aria": 30, "arctic": 40}
# STERA-10M: every sequence trains and none evaluates (Table 7). No converter yet: its format is behind access approval.
SUBSET_TEST_FRACTION = 0.05


def _order(dataset: str, sequences: list[str]) -> list[str]:
    return sorted(sequences, key=lambda name: hashlib.sha1(f"{dataset}:{name}".encode()).hexdigest())


def plan_splits(dataset: str, sequences: list[str], is_fixed_test=None) -> dict[str, str]:
    """Map sequence to 'train' or 'test'. Sequences beyond the paper's size are left out.

    is_fixed_test marks sequences a dataset's own protocol reserves for testing, as H2O's subject 4.
    """
    ordered = _order(dataset, sorted(set(sequences)))
    train_count, _ = PAPER_TABLE_7[dataset]
    if is_fixed_test is not None:
        plan = {name: "test" for name in ordered if is_fixed_test(name)}
        rest = [name for name in ordered if name not in plan]
        if len(rest) < train_count:
            print(f"warning: {dataset} has {len(rest)} training sequences, below the {train_count} planned", flush=True)
        plan.update({name: "train" for name in rest[:train_count]})
        return plan
    test_count = TEST_SEQUENCES[dataset]
    if len(ordered) < train_count + test_count:
        # A partial download. Still usable, but it is not the paper's size, so say so.
        held_out = max(1, round(len(ordered) * SUBSET_TEST_FRACTION))
        print(f"warning: {dataset} has {len(ordered)} sequences, below the {train_count} + {test_count} planned; holding out {held_out}", flush=True)
        return {name: ("test" if index < held_out else "train") for index, name in enumerate(ordered)}
    plan = {name: "test" for name in ordered[:test_count]}
    plan.update({name: "train" for name in ordered[test_count : test_count + train_count]})
    return plan


def cap_test_rows(rows: list[dict]) -> list[dict]:
    """Keep each dataset's evaluation clips at the table's count; training rows pass through."""
    kept = [row for row in rows if row["split"] != "test" or row["dataset"] not in PAPER_TABLE_7]
    for dataset, (_, clips) in PAPER_TABLE_7.items():
        tests = [row for row in rows if row["split"] == "test" and row["dataset"] == dataset]
        tests.sort(key=lambda row: hashlib.sha1(row["path"].encode()).hexdigest())
        kept.extend(tests[:clips])
    return sorted(kept, key=lambda row: row["path"])
