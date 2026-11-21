import json
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
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    main()
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
            return {str(i): i for i in range(8)}

        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True)

    monkeypatch.setattr(hub.HfApi, "dataset_info", lambda *a, **k: SimpleNamespace(sha="data-sha"))
    monkeypatch.setattr(hub.HfApi, "model_info", lambda *a, **k: SimpleNamespace(sha="tokenizer-sha"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    received = {}

    def load(*args, **kwargs):
        received.update(kwargs)
        return [{"text": str(i), "id": str(i)} for i in range(100)]

    monkeypatch.setattr(datasets, "load_dataset", load)
    output = tmp_path / "prepared"
    manifest = prepare_data(output, max_training_tokens=25, validation_documents=2, shard_tokens=7)
    assert manifest["dataset_revision"] == received["revision"] == "data-sha"
    assert manifest["tokenizer_revision"] == "tokenizer-sha"
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
    assert len(notebooks) == 7
    for path in notebooks:
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and not cell["outputs"]
                compile(cell["source"], str(path), "exec")


@pytest.mark.parametrize("family", ["standard", "modified"])
def test_proxy_notebook_smoke_then_full_uses_the_same_trial(
    tmp_path, corpus, tiny_config, training_config, family
):
    import shutil

    notebook = next(Path("notebooks").glob(f"*_{family}_proxy_search.ipynb"))
    cells = [cell["source"] for cell in json.loads(notebook.read_text())["cells"]]
    search = tmp_path / "search.yaml"
    search.write_text(yaml.safe_dump({**load_mapping("configs/search.yaml"), "trials": 1}))
    trials = tmp_path / "trials.json"
    generate_trials(search, trials)
    cached = tmp_path / "ssd"
    paths = []

    def run_training(command, *args):
        assert command == "sweep"
        data = args[args.index("--data") + 1]
        paths.append(data)
        limited = "--stop-after-updates" in args
        sweep(
            replace(tiny_config, family=family),
            replace(training_config, tokens_per_parameter=5),
            data,
            args[args.index("--output") + 1],
            trials,
            device="cpu",
            stop_after_updates=args[args.index("--stop-after-updates") + 1] if limited else None,
        )

    env = {
        "RUNS": tmp_path / "runs",
        "SAVED_DATA": corpus,
        "MICROBATCH_SIZE": 2,
        "RUN_SMOKE": True,
        "RUN_FULL": True,
        "run_training": run_training,
        "training_data": lambda: shutil.copytree(corpus, cached),
    }

    def run_cell(marker):
        source = next(cell for cell in cells if marker in cell)
        exec(compile(source, str(notebook), "exec"), env)  # noqa: S102 - run the generated notebook cell

    run_cell("def run_proxy(")
    run_cell("if RUN_SMOKE:")
    result_path = env["FAMILY_RUNS"] / "trial_000" / "result.json"
    paused = json.loads(result_path.read_text())
    assert paused["status"] == "paused" and paused["step"] == 10
    assert paused["checkpoint"] is None
    assert paths == [corpus] and not cached.exists()
    run_cell("if RUN_FULL:")
    finished = json.loads(result_path.read_text())
    assert paths == [corpus, cached] and finished["status"] == "complete"
    assert paused["fingerprint"] == finished["fingerprint"]
    assert not list(env["FAMILY_RUNS"].rglob("*.pt"))
    assert (env["FAMILY_RUNS"] / "best_hparams.json").is_file()


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
