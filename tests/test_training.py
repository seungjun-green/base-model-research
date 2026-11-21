import json
import os
import random
import shutil
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from base_model_research.data import TokenCorpus
from base_model_research.evaluation import load_trained_model, score_tokens
from base_model_research.storage import load_checkpoint, save_checkpoint
from base_model_research.training import CosineSchedule, train, validation_steps


def test_data_crosses_shards_and_masks_exact_tail(corpus):
    data = TokenCorpus(corpus, "train", verify=True)
    assert data.read(data.total_tokens, 0).shape == (0,)
    assert data.read(29, 8).tolist() == (np.arange(29, 37) % 8).tolist()
    x, y = data.batch(29, 11, 8)
    assert x.shape == (2, 8)
    assert y.flatten()[:11].tolist() == (np.arange(30, 41) % 8).tolist()
    assert (y.flatten()[11:] == -100).all()
    assert y[0, -1] == x[1, 0]


@pytest.mark.parametrize("family", ["standard", "modified"])
def test_proxy_restarts_at_seed_and_never_saves_weights(
    tmp_path, corpus, tiny_config, hp, training_config, family
):
    cfg = replace(tiny_config, family=family)
    cached = shutil.copytree(corpus, tmp_path / "ssd")
    complete = train(cfg, hp, training_config, cached, tmp_path / "complete", device="cpu")
    paused = train(cfg, hp, training_config, corpus, tmp_path / "resumed", device="cpu", stop_after_updates=2)
    assert paused["status"] == "paused"
    assert paused["checkpoint"] is None
    assert not (tmp_path / "resumed" / "checkpoints").exists()
    reopened = train(cfg, hp, training_config, cached, tmp_path / "resumed", device="cpu", resume=True)
    assert reopened == complete
    assert reopened["status"] == "complete" and reopened["consumed_tokens"] == reopened["token_budget"]
    assert not list(tmp_path.rglob("*.pt"))
    metrics = [json.loads(line) for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == list(range(1, reopened["step"] + 1))


@pytest.mark.parametrize("steps", [5069, 5078, 45475, 45615, 11, 1])
def test_one_pass_validation_schedule(steps):
    checks = sorted(validation_steps(steps, 20))
    assert len(checks) == min(20, steps)
    assert checks[-1] == steps
    gaps = [b - a for a, b in zip([0, *checks], checks)]
    assert min(gaps) >= 1 and max(gaps) - min(gaps) <= 1


def test_checkpoint_pruning_keeps_current_file_when_old_timestamps_are_newer(tmp_path):
    future = time.time() + 3600
    for step in (1, 2):
        path = save_checkpoint(tmp_path, {"step": step})
        os.utime(path, (future, future))
    save_checkpoint(tmp_path, {"step": 3})
    assert load_checkpoint(tmp_path)["step"] == 3
    assert len(list(tmp_path.glob("step-*.pt"))) == 2


def test_resume_rejects_changed_protocol(tmp_path, corpus, tiny_config, hp, training_config):
    output = tmp_path / "run"
    train(tiny_config, hp, training_config, corpus, output, device="cpu", stop_after_updates=1)
    with pytest.raises(ValueError, match="Resume refused"):
        train(tiny_config, hp, replace(training_config, seed=43), corpus, output, device="cpu", resume=True)


def test_schedule_preserves_group_ratios_and_endpoints():
    parameters = [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))]
    optimizer = torch.optim.Adam(
        [{"params": parameters[:1], "lr": 0.001}, {"params": parameters[1:], "lr": 0.004}]
    )
    schedule = CosineSchedule(optimizer, 1000, 0.01, 0.1)
    assert schedule.warmup_steps == 10
    assert schedule.factor(0) == 0.1
    assert schedule.factor(9) == 1
    assert schedule.factor(999) == 0.1
    for step in (0, 9, 250, 999):
        schedule.apply(step)
        assert optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"] == 4


def test_likelihood_matches_manual_shifted_logits(tiny_config, hp):
    from base_model_research.parameterization import build_model

    model = build_model(tiny_config, hp).eval()
    context, continuation = [1, 2, 3], [4, 5]
    logits = model(torch.tensor([context + continuation[:-1]]))[0, -2:]
    expected = logits.log_softmax(-1)[torch.arange(2), torch.tensor(continuation)].sum().item()
    actual, greedy = score_tokens(model, context, continuation, max_length=8, chunk_size=1)
    assert actual == pytest.approx(expected, abs=1e-6)
    assert greedy == bool(torch.equal(logits.argmax(-1), torch.tensor(continuation)))


def test_likelihood_long_continuation_scores_every_token(tiny_config, hp):
    from base_model_research.parameterization import build_model

    model = build_model(tiny_config, hp).eval()
    context, continuation = [1], [2, 3, 4, 5, 6, 7, 0, 1, 2]
    actual, _ = score_tokens(model, context, continuation, max_length=4)
    expected = 0
    for start in range(0, len(continuation), 4):
        end = min(start + 4, len(continuation))
        combined = (context + continuation[:end])[-5:]
        logits = model(torch.tensor([combined[:-1]]))[0, -(end - start) :]
        expected -= torch.nn.functional.cross_entropy(
            logits, torch.tensor(continuation[start:end]), reduction="sum"
        ).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_interrupted_proxy_reports_updates_without_saving(
    tmp_path, corpus, tiny_config, hp, training_config, monkeypatch
):
    from base_model_research import training

    apply_update = training._optimizer_update

    def interrupt_on_fourth_update(model, optimizer, data, config, device, start, count):
        if start == 3 * count:
            raise KeyboardInterrupt
        return apply_update(model, optimizer, data, config, device, start, count)

    monkeypatch.setattr(training, "_optimizer_update", interrupt_on_fourth_update)
    output = tmp_path / "run"
    result = train(tiny_config, hp, training_config, corpus, output, device="cpu")
    assert result["status"] == "interrupted" and result["step"] == 3
    assert result["checkpoint"] is None and not list(output.rglob("*.pt"))


def test_main_saves_all_validations_including_last_optimizer_update(
    tmp_path, corpus, tiny_config, hp, training_config, monkeypatch
):
    from base_model_research import training

    cfg = tiny_config.at_width(4)
    output = tmp_path / "main"
    directory = output / "checkpoints"
    losses = [8.0, 4.0, 6.0, 9.0, 2.0, 4.0] + [7.0] * 14
    values = iter(losses)
    monkeypatch.setattr(training, "validate", lambda *args: next(values))
    update = training._optimizer_update
    snapshots = []

    def record_update(model, *args):
        result = update(model, *args)
        snapshots.append(model.embedding.weight.detach().clone())
        return result

    monkeypatch.setattr(training, "_optimizer_update", record_update)
    result = train(cfg, hp, training_config, corpus, output, device="cpu")
    assert result["status"] == "complete" and result["validation_loss"] == losses[-1]
    files = sorted(directory.glob("*.pt"))
    assert len(files) == 20
    steps = sorted(validation_steps(len(snapshots), 20))
    for path, step, expected_loss in zip(files, steps, losses, strict=True):
        payload = load_checkpoint(path)
        assert payload["validation_loss"] == expected_loss
        assert payload["step"] == step
        assert torch.equal(payload["model"]["embedding.weight"], snapshots[step - 1])
        assert all(s["step"].item() == step for s in payload["optimizer"]["state"].values())
    assert not (directory / "best.json").exists()
    assert result["checkpoint"] == str(files[-1])
    final, metadata = load_trained_model(directory, "cpu")
    assert metadata["step"] == result["step"] == len(snapshots)
    assert metadata["consumed_tokens"] == result["token_budget"]
    assert torch.equal(final.embedding.weight, snapshots[-1])
    assert final.embedding.weight is final.readout.weight
    (output / "result.json").unlink()  # Runtime lost after final checkpoint publication.
    assert train(cfg, hp, training_config, corpus, output, device="cpu", resume=True) == result
    assert train(cfg, hp, training_config, corpus, output, device="cpu", resume=True) == result
    assert len(snapshots) == result["step"]


@pytest.mark.parametrize("family", ["standard", "modified"])
def test_main_resumes_exactly_after_interrupted_checkpoint(
    tmp_path, corpus, tiny_config, hp, training_config, monkeypatch, family
):
    from base_model_research import storage, training

    cfg = replace(tiny_config, family=family).at_width(4)
    runs = [tmp_path / name for name in ("control", "resumed")]

    # Exercise all CPU RNGs even though the production model currently has no dropout.
    def stochastic_validation(*args):
        return random.random() + np.random.random() + torch.rand(()).item()

    monkeypatch.setattr(training, "validate", stochastic_validation)
    train(cfg, hp, training_config, corpus, runs[0], device="cpu")
    publish = storage.dump_json
    count = 0

    def disconnect_at_validation(path, value):
        nonlocal count
        count += 1
        if count == 4:
            raise OSError("Drive disconnected")
        publish(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "dump_json", disconnect_at_validation)
        with pytest.raises(OSError, match="Drive disconnected"):
            train(cfg, hp, training_config, corpus, runs[1], device="cpu")

    directory = runs[1] / "checkpoints"
    latest = load_checkpoint(directory)
    assert latest["step"] == sorted(validation_steps(latest["identity"]["total_steps"], 20))[2]
    if family == "modified":
        (runs[1] / "metrics.jsonl").unlink()  # One checkpoint can also reconstruct a lost log.
    starts = []
    update = training._optimizer_update

    def record_start(model, optimizer, data, config, device, start, count):
        starts.append(start)
        return update(model, optimizer, data, config, device, start, count)

    monkeypatch.setattr(training, "_optimizer_update", record_start)
    result = train(cfg, hp, training_config, corpus, runs[1], device="cpu", resume=True)
    assert starts[0] == latest["consumed_tokens"]
    assert result["status"] == "complete"
    assert len(list(directory.glob("*.pt"))) == 20
    expected, actual = [load_checkpoint(run / "checkpoints") for run in runs]
    for key in ("model", "optimizer", "peak_lrs"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in ("python", "numpy", "cuda"):
        assert actual["rng"][key] == expected["rng"][key]
    assert torch.equal(actual["rng"]["torch"], expected["rng"]["torch"])
    logs = []
    for run in runs:
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        logs.append([{k: v for k, v in row.items() if k != "tokens_per_second"} for row in rows])
    assert logs[0] == logs[1]
