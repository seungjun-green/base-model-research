import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from base_model_research.config import load_mapping
from base_model_research.data import TokenCorpus, prepare_data
from base_model_research.evaluation import make_harness_model
from base_model_research.parameterization import build_model
from base_model_research.search import generate_trials, sweep


def test_copy_prepared_data_recovers_an_interrupted_transfer(tmp_path, corpus, monkeypatch):
    import shutil

    from base_model_research.storage import copy_prepared_data

    (corpus / "tokenizer").mkdir()
    (corpus / "tokenizer" / "tokenizer.json").write_text("{}")
    saved = tmp_path / "drive" / "corpus"
    fresh = tmp_path / "fresh-runtime" / "corpus"
    original_copy = shutil.copyfile
    copied = []

    def interrupt_second_shard(src, dst):
        copied.append(src.name)
        if src.name == "train-00001.bin":
            dst.write_bytes(b"partial")
            raise OSError("Transfer interrupted")
        return original_copy(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", interrupt_second_shard)
        with pytest.raises(OSError, match="interrupted"):
            copy_prepared_data(corpus, saved)
    assert not saved.exists()
    copied.clear()

    def record_copy(src, dst):
        copied.append(Path(src).name)
        return original_copy(src, dst)

    monkeypatch.setattr(shutil, "copyfile", record_copy)
    copy_prepared_data(corpus, saved)
    assert "train-00000.bin" not in copied
    assert "train-00001.bin" in copied
    copy_prepared_data(saved, fresh)
    assert (fresh / "tokenizer" / "tokenizer.json").read_bytes() == b"{}"
    assert TokenCorpus(fresh, "train", verify=True).fingerprint == TokenCorpus(corpus, "train").fingerprint
    copied.clear()
    copy_prepared_data(saved, fresh)
    assert not copied
    (fresh / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Different dataset"):
        copy_prepared_data(saved, fresh)


@pytest.mark.parametrize("family", ["standard", "modified"])
def test_main_cli_transfers_winner_and_records_provenance(
    tmp_path, monkeypatch, tiny_config, hp, training_config, corpus, family
):
    from dataclasses import asdict

    from base_model_research.cli import main
    from base_model_research.storage import load_checkpoint

    proxy = replace(tiny_config, family=family)
    model = proxy.at_width(4)
    model_path, training_path, winner_path = (
        tmp_path / name for name in ("main.yaml", "training.yaml", "best_hparams.json")
    )
    model_path.write_text(yaml.safe_dump(asdict(model)))
    training_path.write_text(yaml.safe_dump(asdict(training_config)))
    winner = {
        "family": family,
        "trial_id": "trial_000",
        "selection": "final_cross_entropy_at_full_token_budget",
        "model": asdict(proxy),
        "training": asdict(training_config),
        "hyperparameters": asdict(hp),
        "data_fingerprint": TokenCorpus(corpus, "train").fingerprint,
    }
    winner_path.write_text(json.dumps(winner))
    output = tmp_path / "main"
    monkeypatch.setattr(
        "sys.argv",
        [
            "bmr",
            "train",
            "--model",
            str(model_path),
            "--training",
            str(training_path),
            "--hyperparameters",
            str(winner_path),
            "--data",
            str(corpus),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    main()
    assert list((tmp_path / "cache" / winner["data_fingerprint"]).glob("validation-*.bin"))
    checkpoint = load_checkpoint(output / "checkpoints")
    result = json.loads((output / "result.json").read_text())
    assert result["consumed_tokens"] == checkpoint["identity"]["token_budget"]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    losses = [row["validation_loss"] for row in metrics if "validation_loss" in row]
    assert len(losses) == 20 and checkpoint["validation_loss"] == losses[-1]
    assert checkpoint["consumed_tokens"] == result["token_budget"]
    assert checkpoint["identity"]["selection"] == json.loads(winner_path.read_text())
    assert checkpoint["identity"]["model"]["d_model"] == 4 * proxy.d_model


def test_sweeps_select_separate_winners_from_shared_trials(tmp_path, tiny_config, training_config, corpus):
    config = load_mapping("configs/search.yaml")
    config["trials"] = 3
    search_path = tmp_path / "search.yaml"
    search_path.write_text(yaml.safe_dump(config, sort_keys=False))
    trials_path = tmp_path / "trials.json"
    manifest = generate_trials(search_path, trials_path)
    for family in ("standard", "modified"):
        cfg = replace(tiny_config, family=family)
        rows = sweep(cfg, training_config, corpus, tmp_path / "sweeps", trials_path, device="cpu")
        assert all(row["status"] in ("complete", "diverged") for row in rows)
        root = tmp_path / "sweeps" / family
        winner = json.loads((root / "best_hparams.json").read_text())
        assert winner["family"] == family
        assert winner["trial_manifest_fingerprint"] == manifest["fingerprint"]
        assert winner["validation_loss"] == min(
            r["validation_loss"] for r in rows if r["status"] == "complete"
        )
        before = (root / "results.csv").read_text()
        sweep(cfg, training_config, corpus, tmp_path / "sweeps", trials_path, device="cpu")
        assert before == (root / "results.csv").read_text()


def test_preparation_pins_revisions_and_splits_whole_documents(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    hub = pytest.importorskip("huggingface_hub")
    transformers = pytest.importorskip("transformers")

    class Tokenizer:
        eos_token_id = 7

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[ord(char) % 7 for char in text] for text in texts]}

        def get_vocab(self):
            return {str(i): i for i in range(32_000)}

        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True)

    monkeypatch.setattr(hub.HfApi, "dataset_info", lambda *a, **k: SimpleNamespace(sha="data-sha"))
    monkeypatch.setattr(hub.HfApi, "model_info", lambda *a, **k: SimpleNamespace(sha="tokenizer-sha"))
    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **k: Tokenizer())
    received = {}

    def load(*args, **kwargs):
        received.update(kwargs)
        return [{"text": str(i), "id": str(i)} for i in range(100)]

    monkeypatch.setattr(datasets, "load_dataset", load)
    output = tmp_path / "prepared"
    with monkeypatch.context() as patch:
        from base_model_research.data import TokenWriter

        write = TokenWriter.write

        def interrupt(writer, tokens):
            write(writer, tokens)
            if writer.split == "train":
                raise KeyboardInterrupt

        patch.setattr(TokenWriter, "write", interrupt)
        with pytest.raises(KeyboardInterrupt):
            prepare_data(output, max_training_tokens=25, validation_documents=2, shard_tokens=7)
    assert not output.exists() and not list(tmp_path.glob("prepared.preparing*"))
    increments = []
    manifest = prepare_data(
        output, max_training_tokens=25, validation_documents=2, shard_tokens=7, on_progress=increments.append
    )
    assert sum(increments) == 26
    assert manifest["dataset_revision"] == received["revision"] == "data-sha"
    assert manifest["tokenizer_revision"] == "tokenizer-sha"
    assert manifest["tokenizer"] == "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    assert manifest["tokenizer_max_id"] == 31_999
    assert manifest["splits"]["train"]["tokens"] == 26
    assert manifest["splits"]["validation"]["tokens"] == 4
    validation = TokenCorpus(output, "validation", verify=True)
    assert validation.read(0, 4).tolist() == [ord("0") % 7, 7, ord("1") % 7, 7]
    train = TokenCorpus(output, "train", verify=True)
    assert train.read(0, 2).tolist() == [ord("2") % 7, 7]


def test_actual_harness_evaluates_offline_task(monkeypatch, tiny_config, hp):
    lm_eval = pytest.importorskip("lm_eval")
    datasets = pytest.importorskip("datasets")
    from lm_eval.api.task import ConfigurableTask

    class Tokenizer:
        eos_token_id = 0
        name_or_path = "synthetic"

        def encode(self, text, **kwargs):
            return [ord(char) % 7 + 1 for char in text]

        def decode(self, ids, **kwargs):
            return "".join(chr(i + 96) for i in ids)

    data = datasets.DatasetDict(
        {"test": datasets.Dataset.from_dict({"prompt": ["a", "b"], "answer": ["c", "d"]})}
    )

    def download(self, *args, **kwargs):
        self.dataset = data

    monkeypatch.setattr(ConfigurableTask, "download", download)
    task = ConfigurableTask(
        config={
            "task": "bmr_offline_smoke",
            "dataset_path": "unused",
            "test_split": "test",
            "output_type": "loglikelihood",
            "doc_to_text": "{{prompt}}",
            "doc_to_target": "{{answer}}",
            "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
        }
    )
    model = build_model(tiny_config, hp).eval()
    adapter = make_harness_model(model, Tokenizer(), max_length=8, precision="fp32")
    result = lm_eval.simple_evaluate(
        model=adapter,
        tasks=[task],
        num_fewshot=0,
        bootstrap_iters=0,
        log_samples=False,
        apply_chat_template=False,
    )
    assert "bmr_offline_smoke" in result["results"]
    assert result["n-samples"]["bmr_offline_smoke"]["effective"] == 2


def test_notebook_cells_are_valid_python_and_have_no_saved_outputs():
    notebooks = list(Path("notebooks").glob("*.ipynb"))
    assert len(notebooks) == 9
    for path in [*notebooks, Path("notebooks/local/01_prepare_data.ipynb")]:
        notebook = json.loads(path.read_text())
        if path.parent.name == "local":
            assert "colab" not in notebook["metadata"]
        if path.name.startswith("01_"):
            assert "accelerator" not in notebook["metadata"]
            assert "CPU" in notebook["cells"][0]["source"]
        else:
            assert notebook["metadata"]["accelerator"] == "GPU"
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and not cell["outputs"]
                compile(cell["source"], str(path), "exec")


def test_timing_notebook_trains_one_random_main_without_a_sweep(
    tmp_path, tiny_config, training_config, monkeypatch, capsys
):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from base_model_research import timing_data, training
    from base_model_research.config import DEFAULT_TOKENIZER, ModelConfig
    from base_model_research.data import TokenWriter, write_manifest

    def tokenizer(words):
        backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")

    source = tokenizer(["<unk>", *map(str, range(10)), "one", "two", "<eos>"])
    target = tokenizer(["<unk>", "<eos>", "one", "two", "1", "2", "3", "4"])
    corpus = tmp_path / "data" / "fineweb_edu"
    source.save_pretrained(corpus / "tokenizer")
    splits = {}
    for split, repeats in (("train", 1500), ("validation", 10)):
        writer = TokenWriter(corpus, split, shard_tokens=31)
        writer.write([11, 12, source.eos_token_id] * repeats)
        splits[split] = writer.finish()
    write_manifest(corpus, splits, tokenizer=timing_data.QWEN_TOKENIZER, tokenizer_max_id=source.eos_token_id)
    load = timing_data.load_corpus_tokenizer
    monkeypatch.setattr(
        timing_data,
        "load_corpus_tokenizer",
        lambda name, **kw: target if name == DEFAULT_TOKENIZER else load(name, **kw),
    )
    monkeypatch.setattr(timing_data, "TINYLLAMA_VOCAB_SIZE", len(target))

    streams = (TokenCorpus(corpus, "train"), TokenCorpus(corpus, "validation"))
    converted, valid = timing_data.timing_corpora(*streams, 4500, run_dir=tmp_path)
    converted.block_size = 7
    assert converted.read(2100, 20).tolist() == ([2, 3, 1] * 7)[:20]
    assert valid.total_tokens == 30 and valid.read(0, 30).tolist() == [2, 3, 1] * 10
    resumed, _ = timing_data.timing_corpora(*streams, 4500, run_dir=tmp_path)
    resumed.block_size = 7
    assert len(resumed.positions) > 1 and resumed.read(2102, 4).tolist() == [1, 2, 3, 1]

    path = Path("notebooks/time_standard_main.ipynb")
    cells = [cell["source"] for cell in json.loads(path.read_text())["cells"] if cell["cell_type"] == "code"]
    assert 'SAVED_DATA = storage / "data" / "fineweb_edu"' in cells[0]
    env = {"REPO": Path.cwd(), "RUNS": tmp_path / "runs", "SAVED_DATA": corpus, "DATA": corpus, "json": json}

    def run_cell(source):
        exec(compile(source, str(path), "exec"), env)  # noqa: S102 - execute the generated notebook cell

    run_cell(cells[-2])
    assert env["model"] == ModelConfig.load("configs/models/standard_main.yaml")
    assert env["settings"].tokens_per_parameter == 20
    assert env["output"] == env["RUNS"] / env["trial"]["id"]

    # Exercise the notebook's full training cell on the small CPU fixtures.
    env.update(model=tiny_config.at_width(2), settings=training_config)
    real_train = training.train
    monkeypatch.setattr(training, "train", lambda *a, **kw: real_train(*a, device="cpu", **kw))
    run_cell(cells[-1])
    result = env["result"]
    run = json.loads(env["record"].read_text())
    assert result["status"] == "complete" and result["consumed_tokens"] == run["token_budget"]
    assert run["hyperparameters"] == env["trial"]["hyperparameters"]
    assert run["selection"]["trial_id"] == env["trial"]["id"]
    assert f"{run['total_steps']}/{run['total_steps']}" in capsys.readouterr().err
    run_cell(cells[-1])
    assert env["result"]["step"] == run["total_steps"]


def test_gate_comparison_notebook_records_and_reuses_each_outcome(
    tmp_path, corpus, tiny_config, training_config, monkeypatch
):
    import csv
    from dataclasses import asdict
    from functools import partial

    from base_model_research import training
    from base_model_research.config import DEFAULT_TOKENIZER, GATE_VARIANTS
    from base_model_research.data import write_manifest

    notebook = Path("notebooks/compare_gate_variants.ipynb")
    cells = [c["source"] for c in json.loads(notebook.read_text())["cells"] if c["cell_type"] == "code"]
    assert 'drive.mount("/content/drive")' in cells[0]
    config_dir = tmp_path / "configs/models"
    config_dir.mkdir(parents=True)
    (config_dir / "modified_proxy.yaml").write_text(yaml.safe_dump(asdict(replace(tiny_config, n_layers=2))))
    (config_dir.parent / "training.yaml").write_text(yaml.safe_dump(asdict(training_config)))
    (tmp_path / "experiments").symlink_to(Path("experiments").resolve())
    splits = TokenCorpus(corpus, "train").manifest["splits"]
    write_manifest(corpus, splits, tokenizer_max_id=7, tokenizer=DEFAULT_TOKENIZER)
    env = {
        "REPO": tmp_path,
        "DATA": tmp_path / "ssd" / "tokens",
        "SAVED_DATA": corpus,
        "RUNS": tmp_path / "runs",
        "sys": sys,
        "json": json,
    }

    def execute(source):
        exec(compile(source, str(notebook), "exec"), env)  # noqa: S102 - exercise the generated cells

    execute(cells[1])
    assert env["MICROBATCH_SIZE"] == 16
    env["MICROBATCH_SIZE"] = 2
    execute(cells[2][cells[2].index("sys.path.insert(") :])
    execute(cells[3])
    execute(cells[4])
    assert env["PROXY_DATA"].is_relative_to(tmp_path / "ssd")
    monkeypatch.setattr(training, "train", partial(training.train, device="cpu"))
    update, validate = training._optimizer_update, training.validate

    def update_with_bad_gradient(model, *args):
        if model.config.family == "add_previous":
            model.embedding.weight.register_hook(lambda gradient: gradient * float("nan"))
        return update(model, *args)

    def validate_with_bad_loss(model, *args):
        return float("nan") if model.config.family == "multiply_previous" else validate(model, *args)

    monkeypatch.setattr(training, "_optimizer_update", update_with_bad_gradient)
    monkeypatch.setattr(training, "validate", validate_with_bad_loss)
    execute(cells[5])
    execute(cells[6])
    comparison = env["RUNS"] / "comparison.csv"
    with comparison.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["variant"] for r in rows] == list(GATE_VARIANTS)
    assert [r["status"] for r in rows] == ["diverged", "complete", "diverged"]
    assert (rows[0]["completed_updates"], rows[0]["failed_at_update"]) == ("0", "1")
    assert rows[0]["failure_reason"] == "Nonfinite gradient norm"
    assert (rows[2]["completed_updates"], rows[2]["failed_at_update"]) == ("1", "1")
    assert rows[2]["failure_reason"] == "Nonfinite validation loss"
    assert float(rows[1]["final_validation_loss"]) > 0
    for row in rows:
        record = json.loads((env["RUNS"] / row["variant"] / "trial_000" / "run.json").read_text())
        assert record["hyperparameters"] == env["trial"]["hyperparameters"]
        assert record["training"]["seed"] == training_config.seed
        assert record["parameters"] == int(row["parameters"])

    original = comparison.read_bytes()
    monkeypatch.setattr(training, "_optimizer_update", lambda *args: pytest.fail("Repeated training"))
    execute(cells[5])
    execute(cells[6])
    assert comparison.read_bytes() == original


@pytest.mark.parametrize(("benchmark", "extras"), [(False, "data"), (True, "data,eval")])
def test_colab_uses_kernel_python_and_preserves_torch(tmp_path, monkeypatch, benchmark, extras):
    import runpy
    import subprocess
    from unittest.mock import Mock

    setup = runpy.run_path("scripts/colab_setup.py")["setup_colab"]
    repo, runs = tmp_path / "checkout", tmp_path / "existing_experiment"
    runs.mkdir()
    snapshot = runs / "requirements.freeze.txt"
    snapshot.write_text("torch==2.13.0\n")
    monkeypatch.setitem(setup.__globals__, "_checkout", lambda workspace, runs: repo)
    monkeypatch.setitem(setup.__globals__, "version", lambda name: "2.12.0+cu130")
    install = Mock()
    monkeypatch.setattr(subprocess, "run", install)

    assert setup(tmp_path, runs, benchmark=benchmark) == repo
    install.assert_called_once()
    (command,) = install.call_args.args
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert "torch==2.12.0+cu130" in command and command[-1] == f"{repo}[{extras}]"
    assert "-r" not in command and "--upgrade" not in command
    assert snapshot.read_text() == "torch==2.13.0\n" and not (tmp_path / "envs").exists()


@pytest.mark.parametrize("family", ["standard", "modified"])
def test_proxy_notebook_smoke_finishes_one_trial_and_full_skips_it(
    tmp_path, corpus, tiny_config, training_config, family, monkeypatch
):
    from dataclasses import asdict

    from base_model_research import progress

    path = next(Path("notebooks").glob(f"*_{family}_proxy_search.ipynb"))
    cells = [c["source"] for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    assert 'drive.mount("/content/drive")' in cells[0]
    model = replace(tiny_config, family=family)
    training = replace(training_config, tokens_per_parameter=20)
    config_dir = tmp_path / "configs"
    (config_dir / "models").mkdir(parents=True)
    (config_dir / "models" / f"{family}_proxy.yaml").write_text(yaml.safe_dump(asdict(model)))
    (config_dir / "training.yaml").write_text(yaml.safe_dump(asdict(training)))
    search = tmp_path / "search.yaml"
    search.write_text(yaml.safe_dump({**load_mapping("configs/search.yaml"), "trials": 2}))
    trials = tmp_path / "trials.json"
    generate_trials(search, trials)
    runs = tmp_path / "runs"
    runs.mkdir()
    # A notebook override must take precedence over the profile.
    profile = runs / f"{family}_proxy_profile.json"
    profile.write_text(json.dumps({"training": {"microbatch_size": 1, "activation_checkpointing": False}}))

    calls = []

    def run_with_progress(command, *, cwd):
        data = Path(command[command.index("--data") + 1])
        size = int(command[command.index("--microbatch-size") + 1])
        assert data.is_relative_to(tmp_path / "runtime")
        assert data != corpus and size == 2 and "--cache-dir" not in command
        assert "--stop-after-updates" not in command and "--activation-checkpointing" not in command
        calls.append(data)
        sweep(
            model,
            replace(training, microbatch_size=size),
            data,
            command[command.index("--output") + 1],
            trials,
            device="cpu",
            end=int(command[command.index("--end") + 1]) if "--end" in command else None,
        )

    monkeypatch.setattr(progress, "run_with_progress", run_with_progress)
    env = {
        "RUNS": runs,
        "SAVED_DATA": corpus,
        "DATA": tmp_path / "runtime" / "fineweb_edu",
        "REPO": tmp_path,
        "sys": sys,
        "json": json,
    }

    def execute(source):
        exec(compile(source, str(path), "exec"), env)  # noqa: S102 - generated notebook code

    execute(cells[1])
    assert env["MICROBATCH_SIZE"] == 4
    env.update(MICROBATCH_SIZE=2, RUN_FULL=True)
    # Skip checkout and pip installation, as Colab setup has its own test.
    execute(cells[2][cells[2].index("sys.path.insert(") :])
    execute(cells[3])
    assert env["PROXY_DATA"].is_dir() and not calls
    env["stage_proxy_data"] = lambda *args: pytest.fail("Training tried to stage the data again")
    execute(cells[4])
    execute(cells[5])
    family_runs = env["FAMILY_RUNS"]
    first_trial = family_runs / "trial_000"
    second_trial = family_runs / "trial_001"
    result_path = first_trial / "result.json"
    result = json.loads(result_path.read_text())
    run = json.loads((first_trial / "run.json").read_text())
    assert result["status"] == "complete" and result["step"] == run["total_steps"]
    assert result["consumed_tokens"] == 20 * run["parameters"]
    assert run["data_fingerprint"] == TokenCorpus(corpus, "train").fingerprint
    metrics = [json.loads(line) for line in (first_trial / "metrics.jsonl").read_text().splitlines()]
    assert sum("validation_loss" in row for row in metrics) == 20
    assert not second_trial.exists() and not (family_runs / "best_hparams.json").exists()
    first_save = result_path.stat().st_mtime_ns

    # A fresh session must keep the saved activation setting and refuse to mix microbatch sizes.
    profile.write_text(json.dumps({"training": {"microbatch_size": 1, "activation_checkpointing": True}}))
    env["MICROBATCH_SIZE"] = 1
    with pytest.raises(ValueError, match="sweep uses MICROBATCH_SIZE = 2"):
        execute(cells[6])
    env["MICROBATCH_SIZE"] = 2
    execute(cells[6])
    assert result_path.stat().st_mtime_ns == first_save
    assert json.loads((second_trial / "result.json").read_text())["status"] == "complete"
    assert len(calls) == 2 and calls[0] == calls[1]
    assert not list(family_runs.rglob("*.pt"))
    assert (family_runs / "best_hparams.json").is_file()


def test_proxy_stages_only_its_prefix_and_validation_and_recovers_bad_copies(tmp_path, corpus, monkeypatch):
    import runpy
    import shutil

    stage = runpy.run_path("scripts/colab_proxy.py")["stage_proxy_data"]
    data = tmp_path / "runtime" / "fineweb_edu"
    copied = []
    copyfile = shutil.copyfile

    def copy(src, dst):
        copied.append(src.name)
        if len(copied) == 2:
            Path(dst).write_bytes(b"partial")
            return
        return copyfile(src, dst)

    monkeypatch.setattr(shutil, "copyfile", copy)
    # A 31-target budget ends exactly at a shard boundary; the next shard supplies the lookahead.
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage(corpus, data, 31)
    assert copied == ["train-00000.bin", "train-00001.bin"]
    local = stage(corpus, data, 31)
    assert copied == [
        "train-00000.bin",
        "train-00001.bin",
        "train-00001.bin",
        "validation-00000.bin",
        "validation-00001.bin",
    ]
    assert not (local / "train-00001.bin").is_symlink()
    assert (local / "train-00002.bin").is_symlink()
    assert (local / "manifest.json").read_bytes() == (corpus / "manifest.json").read_bytes()
    copied.clear()
    stage(corpus, data, 31)
    assert not copied
    stage(corpus, data, 62)
    assert copied == ["train-00002.bin"] and not (local / "train-00002.bin").is_symlink()
    # The original trainer checks all shard sizes; every path in its manifest must still resolve.
    assert (
        TokenCorpus(local, "train").read(0, 63).tolist() == TokenCorpus(corpus, "train").read(0, 63).tolist()
    )


def test_changed_sweep_stops_before_starting_next_trial(tmp_path, corpus, training_config, tiny_config):
    trials = tmp_path / "trials.json"
    generate_trials("configs/search.yaml", trials)
    output = tmp_path / "sweep"
    sweep(tiny_config, training_config, corpus, output, trials, device="cpu", end=1)
    # Simulate a sweep created before protocol.json existed.
    (output / "standard" / "protocol.json").unlink()
    with pytest.raises(ValueError, match="different model, dataset, or training configuration"):
        sweep(
            tiny_config,
            replace(training_config, seed=43),
            corpus,
            output,
            trials,
            device="cpu",
            start=1,
            end=2,
        )
    assert not (output / "standard" / "trial_001").exists()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_benchmark_loads_once_and_identifies_the_scored_checkpoint(
    tmp_path, tiny_config, hp, training_config, corpus, monkeypatch, version
):
    from dataclasses import asdict
    from unittest.mock import Mock

    import torch

    from base_model_research import evaluation
    from base_model_research.data import file_sha256
    from base_model_research.storage import save_checkpoint, save_validation_checkpoint

    lm_eval = pytest.importorskip("lm_eval")
    transformers = pytest.importorskip("transformers")
    config = tiny_config.at_width(4)
    model = build_model(config, hp)
    payload = {
        "identity": {
            "model": asdict(config),
            "hyperparameters": asdict(hp),
            "training": asdict(training_config),
            "data_fingerprint": TokenCorpus(corpus, "train").fingerprint,
            "token_budget": 32,
        },
        "model": model.state_dict(),
        "step": 1,
        "consumed_tokens": 32,
        "fingerprint": "test-run",
    }
    directory = tmp_path / "checkpoints"
    if version == 2:
        payload.update(format_version=2, consumed_tokens=16, validation_loss=2.0)
    elif version == 3:
        payload.update(format_version=3, validation_loss=2.0)

    def publish(record):
        if version == 3:
            return save_validation_checkpoint(directory, record)
        if version == 1:
            return save_checkpoint(directory, record)
        # Recreate the archived weights-only format.
        directory.mkdir(exist_ok=True)
        path = directory / f"ranked-{record['step']}.pt"
        torch.save(record, path)
        (directory / "best.json").write_text(json.dumps([{"file": path.name, "sha256": file_sha256(path)}]))
        return path

    original = publish(payload)
    loader = Mock(wraps=evaluation.load_trained_model)
    monkeypatch.setattr(evaluation, "load_trained_model", loader)

    def move_pointer(*args, **kwargs):
        # Another process publishes a checkpoint while evaluation initializes.
        publish({**payload, "step": 2, "validation_loss": 1.0})
        return SimpleNamespace(name_or_path="test-tokenizer", eos_token_id=0)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", move_pointer)
    monkeypatch.setattr(
        lm_eval, "simple_evaluate", lambda **kwargs: {"results": {"fixture": {"acc,none": 1.0}}}
    )
    settings = tmp_path / "eval.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "precision": "fp32",
                "max_length": 8,
                "bootstrap_iters": 0,
                "tasks": [{"name": "fixture", "fewshot": 0}],
            }
        )
    )
    output = tmp_path / "benchmark"
    if version in (2, 3):
        with pytest.raises(ValueError, match="completes its token budget"):
            evaluation.benchmark(
                settings,
                output,
                "cpu",
                standard=directory,
                tokenizer_path=corpus / "tokenizer",
                include_references=False,
            )
        loader.reset_mock()
        (tmp_path / "result.json").write_text(
            json.dumps({"status": "complete", "consumed_tokens": 32, "fingerprint": payload["fingerprint"]})
        )
    evaluation.benchmark(
        settings,
        output,
        "cpu",
        standard=directory,
        tokenizer_path=corpus / "tokenizer",
        include_references=False,
    )
    loader.assert_called_once()
    record = json.loads((output / "standard" / "fixture.json").read_text())
    assert record["identity"]["model"]["model_sha256"] == file_sha256(original)
