import json

import pytest

from base_model_research.config import canonical_json
from base_model_research.search import generate_trials, load_trials, summarize


def test_shared_trials_reproducible_and_in_range(tmp_path):
    first = generate_trials("configs/search.yaml", tmp_path / "a.json")
    second = generate_trials("configs/search.yaml", tmp_path / "b.json")
    assert canonical_json(first) == canonical_json(second)
    assert len(first["trials"]) == 32
    for trial in first["trials"]:
        for name, value in trial["hyperparameters"].items():
            low, high = first["settings"]["ranges"][name]
            assert low <= value <= high
    assert generate_trials("configs/search.yaml", tmp_path / "a.json") == first


def test_manifest_tampering_rejected(tmp_path):
    path = tmp_path / "trials.json"
    manifest = generate_trials("configs/search.yaml", path)
    manifest["trials"][0]["hyperparameters"]["learning_rate"] *= 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        load_trials(path)


def test_pending_trials_do_not_produce_final_winner(tmp_path):
    manifest = generate_trials("configs/search.yaml", tmp_path / "trials.json")
    summarize(tmp_path / "standard", manifest, "standard")
    assert (tmp_path / "standard" / "results.csv").exists()
    assert not (tmp_path / "standard" / "best_hparams.json").exists()


@pytest.mark.parametrize("damage", ["missing_metadata", "wrong_fingerprint", "short_budget"])
def test_selection_rejects_unrelated_or_incomplete_results(tmp_path, damage):
    trials = [{"id": f"trial_{i:03d}", "hyperparameters": {"learning_rate": 0.001}} for i in range(2)]
    for trial in trials:
        directory = tmp_path / trial["id"]
        directory.mkdir()
        metadata = {
            "model": {"family": "standard"},
            "training": {},
            "data_fingerprint": "data",
            "fingerprint": trial["id"],
            "hyperparameters": trial["hyperparameters"],
            "token_budget": 100,
            "environment": {},
        }
        result = {
            "status": "complete",
            "fingerprint": trial["id"],
            "consumed_tokens": 100,
            "validation_loss": 1.0,
        }
        if trial is trials[-1]:
            if damage == "wrong_fingerprint":
                result["fingerprint"] = trials[0]["id"]
            elif damage == "short_budget":
                result["consumed_tokens"] -= 1
        if trial is trials[0] or damage != "missing_metadata":
            (directory / "run.json").write_text(json.dumps(metadata))
        (directory / "result.json").write_text(json.dumps(result))

    with pytest.raises(ValueError, match="trial_001"):
        summarize(tmp_path, {"trials": trials, "fingerprint": "manifest"}, "standard")
    assert not (tmp_path / "best_hparams.json").exists()
