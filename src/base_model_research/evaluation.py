"""LM Evaluation Harness integration. Optional dependencies are imported on use."""

import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import (
    SHARED_TRAINING_FIELDS,
    Hyperparameters,
    ModelConfig,
    TrainConfig,
    canonical_json,
    dump_json,
    json_fingerprint,
    load_mapping,
)
from .data import load_corpus_tokenizer
from .parameterization import build_model
from .storage import load_checkpoint
from .training import autocast_context, environment_report


def load_trained_model(checkpoint_path, device="cuda"):
    payload, checksum = load_checkpoint(checkpoint_path, return_checksum=True)
    identity = payload["identity"]
    model = build_model(
        ModelConfig(**identity["model"]),
        Hyperparameters(**identity["hyperparameters"]),
        device=device,
        initialize=False,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    # Do not retain a second CPU copy of the model or its large Adam state during evaluation.
    metadata = {k: v for k, v in payload.items() if k not in ("model", "optimizer", "rng", "metrics_log")}
    metadata["checkpoint_sha256"] = checksum
    return model, metadata


@torch.no_grad()
def score_tokens(model, context, continuation, max_length=2048, precision="fp32", chunk_size=128):
    """Score each continuation token exactly once, with causal shifted indexing."""
    device = next(model.parameters()).device
    if not context:
        raise ValueError("Scoring requires at least one context token")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    total, greedy = 0.0, True
    previous = list(context)
    for start in range(0, len(continuation), max_length):
        segment = continuation[start : start + max_length]
        tokens = (previous + segment)[-(max_length + 1) :]
        inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        with autocast_context(device, precision):
            hidden = model.hidden_states(inputs)[0, -len(segment) :]
            for offset in range(0, len(segment), chunk_size):
                logits = model.readout(hidden[offset : offset + chunk_size]).float()
                target = torch.tensor(segment[offset : offset + chunk_size], device=device)
                total += -F.cross_entropy(logits, target, reduction="sum").item()
                greedy = greedy and bool(torch.equal(logits.argmax(-1), target))
        previous.extend(segment)
    return total, greedy


def make_harness_model(model, tokenizer, max_length=2048, precision="bf16"):
    from lm_eval.api.model import TemplateLM

    class ResearchHarnessLM(TemplateLM):
        def __init__(self):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self._device = next(model.parameters()).device

        @property
        def eot_token_id(self):
            return tokenizer.eos_token_id

        @property
        def tokenizer_name(self):
            return tokenizer.name_or_path.replace("/", "__")

        def tok_encode(self, string, add_special_tokens=None, **kwargs):
            return tokenizer.encode(string, add_special_tokens=bool(add_special_tokens))

        def _loglikelihood_tokens(self, requests, **kwargs):
            results = []
            for key, context, continuation in requests:
                result = score_tokens(model, context, continuation, max_length, precision)
                results.append(result)
                if key is not None:
                    self.cache_hook.add_partial("loglikelihood", key, result)
            return results

        def loglikelihood_rolling(self, requests, **kwargs):
            return [
                score_tokens(model, [self.eot_token_id], self.tok_encode(req.args[0]), max_length, precision)[
                    0
                ]
                for req in requests
            ]

        @torch.no_grad()
        def generate_until(self, requests, **kwargs):
            results = []
            for request in requests:
                text, settings = request.args
                if settings.get("do_sample", False):
                    raise ValueError("This research adapter supports greedy generation only")
                ids = self.tok_encode(text) or [self.eot_token_id]
                generated = []
                stops = settings.get("until", [])
                if isinstance(stops, str):
                    stops = [stops]
                for _ in range(settings.get("max_gen_toks", 256)):
                    device = next(model.parameters()).device
                    x = torch.tensor([ids[-max_length:]], device=device)
                    with autocast_context(device, precision):
                        hidden = model.hidden_states(x)[:, -1:]
                        token = model.readout(hidden).argmax(-1).item()
                    if token == self.eot_token_id:
                        break
                    ids.append(token)
                    generated.append(token)
                    if any(stop in tokenizer.decode(generated) for stop in stops):
                        break
                decoded = tokenizer.decode(generated)
                for stop in stops:
                    decoded = decoded.split(stop)[0]
                results.append(decoded)
            return results

    return ResearchHarnessLM()


def _serializable(value):
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if hasattr(value, "item"):
        return _serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def benchmark(
    config_path,
    output,
    device="cuda",
    standard=None,
    modified=None,
    tokenizer_path=None,
    limit=None,
    include_references=True,
):
    from huggingface_hub import HfApi
    from lm_eval import simple_evaluate

    settings = load_mapping(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_report()
    specifications = []
    comparison_protocol = None
    for family, path in (("standard", standard), ("modified", modified)):
        if path:
            if not tokenizer_path:
                raise ValueError("Supply the prepared dataset's tokenizer directory for our checkpoints")
            specifications.append((family, path, None))
    if include_references:
        api = HfApi()
        for name, identifier in settings["references"].items():
            specifications.append((name, identifier, {"revision": api.model_info(identifier).sha}))
    if not specifications:
        raise ValueError("No models selected for evaluation")
    for name, path, metadata in specifications:
        if name in ("standard", "modified"):
            model, payload = load_trained_model(path, device)
            if model.config.family != name:
                raise ValueError(f"Wrong checkpoint family for {name}")
            version = payload.get("format_version", 1)
            if version in (2, 3):
                directory = Path(path) if Path(path).is_dir() else Path(path).parent
                result_path = directory.parent / "result.json"
                result = json.loads(result_path.read_text()) if result_path.is_file() else {}
                if (
                    result.get("status") != "complete"
                    or result.get("fingerprint") != payload["fingerprint"]
                    or result.get("consumed_tokens") != payload["identity"]["token_budget"]
                ):
                    raise ValueError("Benchmark checkpoints after the main run completes its token budget")
            if version != 2 and payload["consumed_tokens"] != payload["identity"]["token_budget"]:
                raise ValueError("Main benchmark comparison requires a completed full-budget checkpoint")
            if model.config.width_mult != 4:
                raise ValueError("Main benchmark comparison requires width multiplier 4")
            training = TrainConfig.from_record(payload["identity"]["training"])
            protocol = canonical_json(
                {
                    "data_fingerprint": payload["identity"]["data_fingerprint"],
                    "training": {key: getattr(training, key) for key in SHARED_TRAINING_FIELDS},
                }
            )
            if comparison_protocol is not None and protocol != comparison_protocol:
                raise ValueError("Main models used different datasets or shared training settings")
            comparison_protocol = protocol
            tokenizer_metadata = json.loads((Path(tokenizer_path).parent / "manifest.json").read_text())
            if tokenizer_metadata["fingerprint"] != payload["identity"]["data_fingerprint"]:
                raise ValueError("Tokenizer dataset manifest differs from the training run")
            tokenizer = load_corpus_tokenizer(
                tokenizer_path, tokenizer_name=tokenizer_metadata.get("tokenizer"), local_files_only=True
            )
            adapter = make_harness_model(model, tokenizer, settings["max_length"], settings["precision"])
            checkpoint_identity = {
                "run_fingerprint": payload["fingerprint"],
                "step": payload["step"],
                "model_sha256": payload["checkpoint_sha256"],
            }
            if "validation_loss" in payload:
                checkpoint_identity["validation_loss"] = payload["validation_loss"]
        else:
            from lm_eval.models.huggingface import HFLM

            adapter = HFLM(
                pretrained=path,
                revision=metadata["revision"],
                device=device,
                dtype="bfloat16" if settings["precision"] == "bf16" else "float32",
                max_length=settings["max_length"],
                batch_size=1,
            )
            checkpoint_identity = {"pretrained": path, "revision": metadata["revision"]}
        for task in settings["tasks"]:
            identity = {
                "model": checkpoint_identity,
                "task": task,
                "settings": settings,
                "limit": limit,
                "environment": environment,
            }
            fingerprint = json_fingerprint(identity)
            destination = output / name / f"{task['name']}.json"
            if destination.exists():
                if json.loads(destination.read_text())["fingerprint"] != fingerprint:
                    raise ValueError(f"Existing evaluation has different settings: {destination}")
                continue
            result = simple_evaluate(
                model=adapter,
                tasks=[task["name"]],
                num_fewshot=task["fewshot"],
                limit=limit,
                random_seed=42,
                numpy_random_seed=42,
                torch_random_seed=42,
                fewshot_random_seed=42,
                log_samples=False,
                apply_chat_template=False,
                bootstrap_iters=settings["bootstrap_iters"],
            )
            dump_json(
                destination,
                {"fingerprint": fingerprint, "identity": identity, "result": _serializable(result)},
            )
        del adapter
        if name in ("standard", "modified"):
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return collect_results(output)


def collect_results(output):
    output = Path(output)
    rows = []
    for path in sorted(output.glob("*/*.json")):
        record = json.loads(path.read_text())
        for task, metrics in record["result"]["results"].items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    rows.append(
                        {
                            "model": path.parent.name,
                            "task": task,
                            "metric": metric,
                            "value": value,
                            "limited_run": record["identity"]["limit"] is not None,
                        }
                    )
    with open(output / "comparison.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "task", "metric", "value", "limited_run"])
        writer.writeheader()
        writer.writerows(rows)
    return rows
