"""Proxy notebook helpers, embedded so they work with an existing experiment's pinned trainer."""

import json
import shutil
from pathlib import Path

import torch
from tqdm.auto import tqdm

from base_model_research.config import ModelConfig, TrainConfig, dump_json
from base_model_research.data import TokenCorpus, file_sha256
from base_model_research.diagnostics import DEFAULT_HP
from base_model_research.models import ResearchLM, parameter_count
from base_model_research.storage import run_lock


def stage_proxy_data(source, data, training_targets):
    """Stage the training prefix (including lookahead) and every validation shard."""
    manifest = source / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Upload the prepared FineWeb-Edu folder to {source} before training.")
    local_manifest = data / "manifest.json"
    if source == data or (local_manifest.is_file() and local_manifest.read_bytes() == manifest.read_bytes()):
        print(f"Using the prepared local dataset at {data}.", flush=True)
        return data

    train = TokenCorpus(source, "train")
    validation = TokenCorpus(source, "validation")
    if train.total_tokens < training_targets + 1:
        raise ValueError(f"This proxy needs {training_targets + 1:,} training tokens including lookahead.")
    local = data.parent / "proxy_shards" / train.fingerprint
    with run_lock(local, blocking=True):
        required = []
        for corpus, limit in ((train, training_targets + 1), (validation, validation.total_tokens)):
            for start, entry in zip(corpus.starts, corpus.entries):
                name = entry["path"]
                if Path(name).name != name or name in ("", ".", ".."):
                    raise ValueError(f"Expected a shard filename: {name!r}")
                if start < limit:
                    required.append(entry)
                elif not (local / name).exists():
                    # Pinned trainers stat every shard listed in the original manifest.
                    # Only shards beyond this proxy's training budget are links to Drive.
                    (local / name).symlink_to(source / name)

        print(f"Preparing {sum(e['tokens'] * 4 for e in required) / 1e9:.2f} GB at {local}", flush=True)
        for entry in tqdm(required, desc="Copying or verifying proxy shards", unit="shard"):
            target = local / entry["path"]
            if not target.is_symlink() and target.is_file() and file_sha256(target) == entry["sha256"]:
                continue
            temporary = target.with_suffix(".copying")
            try:
                shutil.copyfile(source / entry["path"], temporary)
                if file_sha256(temporary) != entry["sha256"]:
                    raise ValueError(f"Shard checksum mismatch: {entry['path']}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        # Keep the full corpus identity for saved trials and the main-model transfer.
        dump_json(local / "manifest.json", train.manifest)
    return local


def proxy_training_targets(repo, family):
    model = ModelConfig.load(repo / "configs" / "models" / f"{family}_proxy.yaml")
    config = TrainConfig.load(repo / "configs" / "training.yaml")
    with torch.device("meta"):
        parameters = parameter_count(ResearchLM(model, DEFAULT_HP))
    return parameters * config.tokens_per_parameter


def proxy_sweep_args(repo, runs, family, microbatch_size, local, *, reuse_trial_000=None):
    model_name = f"{family}_proxy"
    model_path = repo / "configs" / "models" / f"{model_name}.yaml"
    protocol = runs / "proxy" / ModelConfig.load(model_path).family / "protocol.json"
    settings_path = runs / f"{model_name}_profile.json"
    if reuse_trial_000 is not None and (reuse_trial_000 / "run.json").is_file():
        settings_path = reuse_trial_000 / "run.json"
    if protocol.is_file():
        settings_path = protocol
    settings = json.loads(settings_path.read_text())["training"] if settings_path.is_file() else {}
    if microbatch_size is None and reuse_trial_000 is not None:
        microbatch_size = settings.get("microbatch_size", 16)
    if type(microbatch_size) is not int:
        raise ValueError("MICROBATCH_SIZE must be a positive integer that divides the global batch size.")
    config = TrainConfig.load(repo / "configs" / "training.yaml", microbatch_size=microbatch_size)
    if settings_path == protocol and settings["microbatch_size"] != microbatch_size:
        raise ValueError(
            f"This {family} sweep uses MICROBATCH_SIZE = {settings['microbatch_size']}. "
            f"Use that value to continue, or set a new EXPERIMENT name to start a sweep at {microbatch_size}."
        )
    accumulation = config.global_batch_size // microbatch_size
    print(f"{model_name}: {accumulation} microbatches of {microbatch_size} sequences per update.", flush=True)
    args = [
        "--model",
        str(model_path),
        "--data",
        str(local),
        "--output",
        str(runs / "proxy"),
        "--microbatch-size",
        str(microbatch_size),
        "--progress-events",
    ]
    if settings.get("activation_checkpointing", config.activation_checkpointing):
        args.append("--activation-checkpointing")
    if reuse_trial_000 is not None:
        args.extend(["--reuse-trial-000", str(reuse_trial_000)])
    return args
