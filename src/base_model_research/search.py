import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .config import Hyperparameters, canonical_json, dump_json, json_fingerprint, load_mapping
from .data import TokenCorpus
from .storage import run_lock
from .training import train


def generate_trials(search_config, output):
    settings = load_mapping(search_config)
    if settings["sampling"] != "log_uniform":
        raise ValueError("Only log_uniform sampling is supported")
    rng = np.random.default_rng(settings["seed"])
    trials = []
    for idx in range(settings["trials"]):
        values = {}
        for name, (low, high) in settings["ranges"].items():
            if low <= 0 or high < low:
                raise ValueError(f"Invalid log-uniform bounds for {name}")
            values[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        trials.append({"id": f"trial_{idx:03d}", "hyperparameters": asdict(Hyperparameters(**values))})
    manifest = {"format_version": 1, "settings": settings, "rng": "numpy.PCG64", "trials": trials}
    manifest["fingerprint"] = json_fingerprint(manifest)
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError("Refusing to overwrite a different trial manifest")
    else:
        dump_json(output, manifest)
    return manifest


def load_trials(path):
    manifest = json.loads(Path(path).read_text())
    raw = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if json_fingerprint(raw) != manifest["fingerprint"]:
        raise ValueError("Trial manifest fingerprint mismatch")
    ids = [t["id"] for t in manifest["trials"]]
    if not ids:
        raise ValueError("Trial manifest is empty")
    if len(set(ids)) != len(ids) or any(Path(x).name != x for x in ids):
        raise ValueError("Invalid trial IDs")
    return manifest


def _read_trial(directory, trial, family):
    run_path = directory / "run.json"
    result_path = directory / "result.json"
    pending = {"status": "pending"}
    if not run_path.exists():
        if result_path.exists():
            raise ValueError(f"{directory}: result exists without run metadata")
        return None, pending

    run = json.loads(run_path.read_text())
    if run["model"]["family"] != family or run["hyperparameters"] != trial["hyperparameters"]:
        raise ValueError(f"{directory}: run disagrees with the family or trial hyperparameters")
    if not result_path.exists():
        return run, pending
    result = json.loads(result_path.read_text())
    if result["fingerprint"] != run["fingerprint"]:
        raise ValueError(f"{directory}: result belongs to a different run")
    if result["status"] == "complete" and (
        result["consumed_tokens"] != run["token_budget"] or not math.isfinite(result["validation_loss"])
    ):
        raise ValueError(f"{directory}: completed trial requires its full budget and finite validation loss")
    return run, result


def _trial_protocol(run):
    return {key: run[key] for key in ("model", "training", "data_fingerprint")}


def _reuse_trial_000(source, root, trial, expected):
    """Import a finished standalone run atomically while its files are locked."""
    target = root / trial["id"]
    if trial["id"] != "trial_000" or target.exists() or not (source / "result.json").is_file():
        return
    with run_lock(source):
        run = json.loads((source / "run.json").read_text())
        if (
            canonical_json(_trial_protocol(run)) != expected
            or run["hyperparameters"] != trial["hyperparameters"]
        ):
            print("Standalone trial 000 used different settings or data. Training a fresh trial.", flush=True)
            return
        _, result = _read_trial(source, trial, run["model"]["family"])
        if result["status"] != "complete":
            return
        if result["step"] != run["total_steps"] or run["token_budget"] != (
            run["parameters"] * run["training"]["tokens_per_parameter"]
        ):
            raise ValueError(f"{source}: inconsistent standalone trial budget")
        with TemporaryDirectory(dir=root) as tmp:
            staged = Path(tmp) / trial["id"]
            staged.mkdir()
            for name in ("run.json", "result.json", "metrics.jsonl", "parameter_groups.json"):
                shutil.copyfile(source / name, staged / name)
            staged.rename(target)
    print(
        f"Reused completed standalone trial 000 (validation loss {result['validation_loss']:.4f}).",
        flush=True,
    )


def summarize(root, manifest, family):
    root = Path(root)
    with run_lock(root, blocking=True):
        return _summarize_locked(root, manifest, family)


def _summarize_locked(root, manifest, family):
    rows, candidates, identities = [], [], set()
    for trial in manifest["trials"]:
        run, result = _read_trial(root / trial["id"], trial, family)
        row = {
            "trial_id": trial["id"],
            **trial["hyperparameters"],
            "status": result["status"],
            "validation_loss": result.get("validation_loss"),
        }
        rows.append(row)
        if run is not None:
            identities.add(canonical_json(_trial_protocol(run)))
        if result["status"] == "complete":
            candidates.append((result["validation_loss"], trial["id"], trial, run))
    if len(identities) > 1:
        raise ValueError("Trials used different training or validation protocols")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "results.csv")
    # Never promote a provisional winner while trials are pending or interrupted.
    finished = all(row["status"] in ("complete", "diverged") for row in rows)
    best_path = root / "best_hparams.json"
    if finished and candidates:
        loss, trial_id, trial, run = min(candidates, key=lambda item: (item[0], item[1]))
        dump_json(
            best_path,
            {
                "family": family,
                "trial_id": trial_id,
                "validation_loss": loss,
                "hyperparameters": trial["hyperparameters"],
                "trial_manifest_fingerprint": manifest["fingerprint"],
                "data_fingerprint": run["data_fingerprint"],
                "training": run["training"],
                "model": run["model"],
                "environment": run["environment"],
                "selection": "final_cross_entropy_at_full_token_budget",
            },
        )
    elif best_path.exists():
        best_path.unlink()
    return rows


def sweep(
    model_config,
    training,
    data_dir,
    output,
    trials_path,
    device="cuda",
    start=0,
    end=None,
    stop_after_updates=None,
    remote=None,
    progress=None,
    cache_dir=None,
    *,
    reuse_trial_000=None,
):
    manifest = load_trials(trials_path)
    end = len(manifest["trials"]) if end is None else end
    if not 0 <= start < end <= len(manifest["trials"]):
        raise ValueError("Trial range must satisfy 0 <= start < end <= number of trials")
    root = Path(output) / model_config.family
    expected = canonical_json(
        {
            "model": asdict(model_config),
            "training": asdict(training),
            "data_fingerprint": TokenCorpus(data_dir, "train", cache_dir=cache_dir).fingerprint,
        }
    )
    manifest_path = root / "trial_manifest.json"
    selected = manifest["trials"][start:end]
    terminal_ids = set()
    with run_lock(root, blocking=True):
        protocol_path = root / "protocol.json"
        if protocol_path.exists() and canonical_json(json.loads(protocol_path.read_text())) != expected:
            raise ValueError(
                "This sweep directory already uses a different model, dataset, or training configuration"
            )
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("This sweep directory already belongs to a different trial manifest")
        if reuse_trial_000 is not None:
            _reuse_trial_000(Path(reuse_trial_000), root, manifest["trials"][0], expected)
        # Validate the whole sweep before skipping finished runs or starting a costly new one.
        for trial in manifest["trials"]:
            run, previous = _read_trial(root / trial["id"], trial, model_config.family)
            if previous["status"] in ("complete", "diverged"):
                terminal_ids.add(trial["id"])
            if run is not None and canonical_json(_trial_protocol(run)) != expected:
                raise ValueError(
                    f"{trial['id']}: existing trial uses a different model, dataset, or training configuration"
                )
        if not protocol_path.exists():
            dump_json(protocol_path, json.loads(expected))
        if not manifest_path.exists():
            dump_json(manifest_path, manifest)
    finished = sum(trial["id"] in terminal_ids for trial in selected)
    show_trials = progress is not None and len(selected) > 1
    label = f"{model_config.family} proxy search"
    if show_trials:
        progress(label, "trial", finished, len(selected))
    for trial in selected:
        directory = root / trial["id"]
        if trial["id"] in terminal_ids:
            continue
        _, result = _read_trial(directory, trial, model_config.family)
        if result["status"] not in ("complete", "diverged"):
            hp = Hyperparameters(**trial["hyperparameters"])
            result = train(
                model_config,
                hp,
                training,
                data_dir,
                directory,
                device=device,
                resume=(directory / "run.json").exists(),
                stop_after_updates=stop_after_updates,
                remote=f"{remote.rstrip('/')}/{model_config.family}/{trial['id']}" if remote else None,
                progress=progress,
                cache_dir=cache_dir,
            )
            summarize(root, manifest, model_config.family)
            if result["status"] in ("paused", "interrupted"):
                break
        finished += 1
        if show_trials:
            progress(label, "trial", finished, len(selected), f"{trial['id']} {result['status']}")
    rows = summarize(root, manifest, model_config.family)
    if remote:
        from .storage import sync_to_remote

        sync_to_remote(root, f"{remote.rstrip('/')}/{model_config.family}")
    return rows
