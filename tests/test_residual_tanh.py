import csv
import json
import math
import sys
from dataclasses import asdict, replace
from functools import partial
from io import StringIO
from pathlib import Path

import pytest
import torch
import yaml

from base_model_research import training
from base_model_research.config import DEFAULT_TOKENIZER, ModelConfig
from base_model_research.data import TokenCorpus, write_manifest
from base_model_research.gate_gradients import GateGradientLog
from base_model_research.models import ResearchLM, parameter_count
from base_model_research.parameterization import build_model


def test_residual_tanh_matches_original_gates_plus_skip_including_first_layer(tiny_config, hp):
    original = build_model(replace(tiny_config, family="modified", n_layers=2), hp)
    bounded = build_model(replace(original.config, family="residual_tanh"), hp)
    bounded.load_state_dict(original.state_dict())
    x = torch.randn(1, 4, tiny_config.d_model)
    a_prev = f_prev = None
    for ref, layer in zip(original.layers, bounded.layers):
        a = ref.attention(ref.input_norm(x))
        h = a + (x + ref.attention_gate(x, x if a_prev is None else a_prev).tanh())
        f = ref.ffn(ref.post_attention_norm(h))
        expected = f + (h + ref.ffn_gate(h, h if f_prev is None else f_prev).tanh())
        actual, actual_a, actual_f = layer(x, a_prev, f_prev)
        torch.testing.assert_close((actual, actual_a, actual_f), (expected, a, f), rtol=0, atol=0)
        x, a_prev, f_prev = expected, a, f

    proxy = replace(ModelConfig.load("configs/models/modified_proxy.yaml"), family="residual_tanh")
    with torch.device("meta"):
        real = ResearchLM(proxy, hp)
    assert parameter_count(real) == 13_362_624 and len(real.layers) == 28


def test_gate_log_distinguishes_zero_tiny_large_nonfinite_and_missing(tiny_config, hp):
    model = build_model(replace(tiny_config, family="residual_tanh"), hp)
    stream = StringIO()
    log = GateGradientLog(stream, model)
    for p, value in zip(log.projections, (0.0, 1e-30, 1e30, float("inf"))):
        p.grad = torch.full_like(p, value)
    log.write(1)
    log.projections[0].grad = None
    log.write(2)
    first, second = csv.DictReader(stream.getvalue().splitlines())
    assert first["attention_current"] == "0.0" and second["attention_current"] == ""
    assert float(first["attention_previous"]) == pytest.approx(1e-30, rel=1e-6, abs=0)
    assert float(first["ffn_current"]) == pytest.approx(1e30)
    assert math.isinf(float(first["ffn_previous"]))


def test_restarted_proxy_replaces_old_trace_and_keeps_failed_gradients(
    tmp_path, tiny_config, hp, training_config, corpus, monkeypatch
):
    cfg = replace(tiny_config, family="residual_tanh", n_layers=2)
    out = tmp_path / "trial_000"
    args = (cfg, hp, training_config, corpus, out)
    assert training.train(*args, device="cpu", stop_after_updates=2)["status"] == "paused"
    original_build = training.build_model

    def inject_nonfinite(*args, **kwargs):
        model = original_build(*args, **kwargs)
        model.layers[-1].ffn_gate.previous.weight.register_hook(lambda grad: grad * float("nan"))
        return model

    monkeypatch.setattr(training, "build_model", inject_nonfinite)
    result = training.train(*args, device="cpu", resume=True)
    assert result["status"] == "diverged" and result["failed_at_step"] == 1 and result["step"] == 0
    with (out / "gate_gradients.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["update"], r["layer"]) for r in rows] == [("1", "1"), ("1", "2")]
    assert math.isnan(float(rows[-1]["ffn_previous"]))


def test_residual_tanh_notebook_trains_logs_and_plots(
    tmp_path, tiny_config, training_config, corpus, monkeypatch
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    path = Path("notebooks/trial_000_residual_tanh_proxy.ipynb")
    cells = [c["source"] for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    assert 'drive.mount("/content/drive")' in cells[0] and "proxy_residual_tanh" in cells[0]
    folder = tmp_path / "configs/models"
    folder.mkdir(parents=True)
    (folder / "modified_proxy.yaml").write_text(
        yaml.safe_dump(asdict(replace(tiny_config, family="modified", n_layers=2)))
    )
    (folder.parent / "training.yaml").write_text(yaml.safe_dump(asdict(training_config)))
    (tmp_path / "experiments").symlink_to(Path("experiments").resolve())
    splits = TokenCorpus(corpus, "train").manifest["splits"]
    write_manifest(corpus, splits, tokenizer=DEFAULT_TOKENIZER, tokenizer_max_id=7)
    env = {
        "REPO": tmp_path,
        "DATA": tmp_path / "ssd" / "data",
        "SAVED_DATA": corpus,
        "RUNS": tmp_path / "runs",
        "sys": sys,
        "json": json,
    }

    def execute(source):
        exec(compile(source, str(path), "exec"), env)  # noqa: S102 - exercise generated notebook cells

    execute(cells[1])
    assert env["MICROBATCH_SIZE"] == 16
    env["MICROBATCH_SIZE"] = training_config.microbatch_size
    execute(cells[2][cells[2].index("sys.path.insert(") :])
    execute(cells[3])
    execute(cells[4])
    out = env["OUTPUT"]
    assert out.name == "trial_000" and env["PROXY_DATA"].is_relative_to(tmp_path / "ssd")
    assert env["trial"]["id"] == "trial_000" and env["model"].family == "residual_tanh"
    monkeypatch.setattr(training, "train", partial(training.train, device="cpu"))
    execute(cells[5])
    assert env["result"]["status"] == "complete"
    grad_path = out / "gate_gradients.csv"
    with grad_path.open() as stream:
        rows = list(csv.DictReader(stream))
    num_layers, updates = env["model"].n_layers, env["updates"]
    assert [(int(r["update"]), int(r["layer"])) for r in rows] == [
        (s, l) for s in range(1, updates + 1) for l in range(1, num_layers + 1)
    ]
    # An independent, unsplit backward pass gives the accumulated pre-clipping gradients at update 1.
    training.seed_everything(training_config.seed)
    reference = build_model(env["model"], env["hp"])
    n = training_config.context_length * training_config.global_batch_size
    x, y = TokenCorpus(corpus, "train").batch(0, n, training_config.context_length)
    (reference.loss_sum(x, y) / n).backward()
    for row, layer in zip(rows[:num_layers], reference.layers):
        for gate_name in ("attention", "ffn"):
            for branch in ("current", "previous"):
                p = getattr(getattr(layer, gate_name + "_gate"), branch).weight
                expected = p.grad.double().square().mean().sqrt().item()
                assert float(row[f"{gate_name}_{branch}"]) == pytest.approx(expected, rel=2e-5, abs=1e-15)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    checks = [r for r in records if "validation_loss" in r]
    assert {r["step"] for r in checks} == training.validation_steps(
        updates, training_config.validations_per_run
    )
    assert checks[-1]["validation_loss"] == env["result"]["validation_loss"]
    execute(cells[6])
    fig = env["figure"]
    assert sum(len(ax.images) for ax in fig.axes) == 4
    assert list(next(ax for ax in fig.axes if ax.get_ylabel() == "Validation loss").lines[0].get_ydata()) == [
        r["validation_loss"] for r in checks
    ]
    assert (out / "gate_gradients_and_validation.png").stat().st_size > 0
    plt.close(fig)
    saved = grad_path.read_bytes()
    execute(cells[5])
    assert grad_path.read_bytes() == saved
