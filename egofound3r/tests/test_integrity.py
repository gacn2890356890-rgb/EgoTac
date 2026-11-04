"""Majority voting must repair a corrupted weight and refuse when there is no majority."""
import pytest
import torch

from egotac import integrity


def test_vote_recovers_single_bad_read(tmp_path, monkeypatch):
    good = {"a": torch.arange(4.0), "b": torch.ones(3)}
    bad = {"a": torch.arange(4.0), "b": torch.tensor([1.0, float("nan"), 1.0])}
    reads = iter([good, bad, good])
    monkeypatch.setattr(integrity, "_read_state", lambda _path: next(reads))
    voted, disagreed = integrity.voted_state(tmp_path / "x.pt")
    assert disagreed == 1 and torch.equal(voted["b"], good["b"])


def test_vote_refuses_without_majority(tmp_path, monkeypatch):
    reads = iter([{"a": torch.tensor([float(i)])} for i in range(3)])
    monkeypatch.setattr(integrity, "_read_state", lambda _path: next(reads))
    with pytest.raises(RuntimeError, match="all 3 reads"):
        integrity.voted_state(tmp_path / "x.pt")


def test_repair_overwrites_corrupted_live_weight(tmp_path):
    path = tmp_path / "w.pt"
    module = torch.nn.Linear(2, 2)
    torch.save({"model": {k: v.clone() for k, v in module.state_dict().items()}}, path)
    original = module.weight.data.clone()
    module.weight.data[0, 0] = float("nan")
    report = integrity.repair_module(module, [(path, "")])
    assert report["tensors_repaired"] == ["weight"]
    assert torch.equal(module.weight.data, original)
    assert integrity.repair_module(module, [(path, "")])["tensors_repaired"] == []
