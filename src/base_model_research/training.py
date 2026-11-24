import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import dump_json, json_fingerprint
from .data import TokenCorpus
from .models import parameter_count
from .parameterization import build_model, make_optimizer, parameter_report
from .storage import (
    load_checkpoint,
    run_lock,
    save_validation_checkpoint,
    sync_to_remote,
    validation_checkpoint_name,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state(device):
    algorithm, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), pos, has_gauss, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state, device):
    if (state["cuda"] is not None) != (device.type == "cuda"):
        raise ValueError("Resume requires the same device type as the saved checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def environment_report():
    packages = {}
    for name in ("torch", "mup", "numpy", "transformers", "datasets", "lm_eval"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        commit, dirty = None, None
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "git_dirty": dirty,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        report.update(
            gpu=torch.cuda.get_device_name(),
            vram_bytes=torch.cuda.get_device_properties(0).total_memory,
            capability=list(torch.cuda.get_device_capability()),
        )
    return report


class CosineSchedule:
    """Apply one common factor while preserving MuAdam's group LR ratios."""

    def __init__(self, optimizer, total_steps, warmup_fraction, min_lr_ratio):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = math.ceil(total_steps * warmup_fraction)
        self.min_lr_ratio = min_lr_ratio
        self.peak_lrs = [group["lr"] for group in optimizer.param_groups]

    def factor(self, update):
        if update < self.warmup_steps:
            return (update + 1) / self.warmup_steps
        denominator = max(1, self.total_steps - self.warmup_steps - 1)
        progress = min(1.0, max(0.0, (update - self.warmup_steps) / denominator))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    def apply(self, update):
        factor = self.factor(update)
        for group, peak in zip(self.optimizer.param_groups, self.peak_lrs):
            group["lr"] = peak * factor


def autocast_context(device, precision):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.device(device).type == "cuda" and precision == "bf16"
        else nullcontext()
    )


def validation_steps(total_steps, count):
    """The first optimizer boundary at or after each fraction of one full run."""
    count = min(total_steps, count)
    return {(i * total_steps + count - 1) // count for i in range(1, count + 1)}


@torch.no_grad()
def validate(model, corpus, config, device):
    was_training = model.training
    model.eval()
    count, total = 0, 0.0
    capacity = config.microbatch_size * config.context_length
    while count < corpus.total_tokens - 1:
        take = min(capacity, corpus.total_tokens - 1 - count)
        x, y = corpus.batch(count, take, config.context_length)
        with autocast_context(device, config.precision):
            total += model.loss_sum(x.to(device), y.to(device), config.loss_chunk_size).item()
        count += take
    model.train(was_training)
    return total / count


def _optimizer_update(model, optimizer, corpus, config, device, start, target_count):
    """Accumulate the mean gradient over the actual number of targets in this update."""
    optimizer.zero_grad(set_to_none=True)
    capacity = config.microbatch_size * config.context_length
    summed_loss = 0.0
    for offset in range(0, target_count, capacity):
        take = min(capacity, target_count - offset)
        x, y = corpus.batch(start + offset, take, config.context_length)
        with autocast_context(device, config.precision):
            loss = model.loss_sum(x.to(device), y.to(device), config.loss_chunk_size)
        value = loss.detach().item()
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite training loss")
        (loss / target_count).backward()
        summed_loss += value

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm")
    optimizer.step()
    return summed_loss / target_count, norm


def train(
    model_config,
    hp,
    config,
    data_dir,
    run_dir,
    device="cuda",
    resume=False,
    stop_after_updates=None,
    remote=None,
    selection=None,
    progress=None,
    cache_dir=None,
    *,
    adapt_corpora=None,
):
    """Train once over the assigned token budget, checkpointing main runs at every validation."""
    config.require_schedule()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Connect the GPU runtime, or use --device cpu for checks")
    if device.type != "cuda" and config.precision == "bf16":
        raise ValueError("Use precision=fp32 for CPU checks; production BF16 training requires CUDA")
    if device.type == "cuda" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA runtime does not support BF16")
    run_dir = Path(run_dir)
    with run_lock(run_dir):
        return _train_locked(
            model_config,
            hp,
            config,
            data_dir,
            run_dir,
            device,
            resume,
            stop_after_updates,
            remote,
            selection,
            progress,
            cache_dir,
            adapt_corpora,
        )


def _train_locked(
    model_config,
    hp,
    config,
    data_dir,
    run_dir,
    device,
    resume,
    stop_after_updates,
    remote,
    selection,
    progress,
    cache_dir,
    adapt_corpora,
):
    corpus = TokenCorpus(data_dir, "train", cache_dir=cache_dir)
    validation = TokenCorpus(data_dir, "validation", cache_dir=cache_dir)
    checkpoint_dir = run_dir / "checkpoints"
    is_main = model_config.width_mult != 1
    seed_everything(config.seed)
    model = build_model(model_config, hp, device=device)
    model.activation_checkpointing = config.activation_checkpointing
    model.train()
    parameters = parameter_count(model)
    budget = parameters * config.tokens_per_parameter
    if adapt_corpora is not None:
        corpus, validation = adapt_corpora(corpus, validation, budget + 1)
    if corpus.manifest.get("tokenizer_max_id", 0) >= model_config.vocab_size:
        raise ValueError("Prepared tokenizer has token IDs outside the model vocabulary")
    if corpus.total_tokens < budget + 1:
        raise ValueError(
            f"Need {budget + 1:,} training tokens including lookahead; have {corpus.total_tokens:,}"
        )
    tokens_per_update = config.global_batch_size * config.context_length
    total_steps = math.ceil(budget / tokens_per_update)
    checks = validation_steps(total_steps, config.validations_per_run)
    optimizer = make_optimizer(model, config)
    schedule = CosineSchedule(optimizer, total_steps, config.warmup_fraction, config.min_lr_ratio)
    identity = {
        "model": asdict(model_config),
        "hyperparameters": asdict(hp),
        "training": asdict(config),
        "data_fingerprint": corpus.fingerprint,
        "parameters": parameters,
        "token_budget": budget,
        "total_steps": total_steps,
        "selection": selection,
    }
    fingerprint = json_fingerprint(identity)
    step, consumed = 0, 0
    final_validation = None
    latest_path = checkpoint_dir / "latest.json"
    initial_environment = environment_report()
    result_path = run_dir / "result.json"
    metrics_path = run_dir / "metrics.jsonl"
    if (run_dir / "run.json").exists():
        if not resume:
            raise FileExistsError(f"Run exists: {run_dir}; pass --resume")
        metadata = json.loads((run_dir / "run.json").read_text())
        if metadata["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: model, hyperparameters, training settings, or dataset changed")
        if is_main and metadata.get("checkpoint_format") != 3:
            raise ValueError("This run predates per-validation checkpoints. Use a new main run directory.")
        # Training reads prepared token IDs and uses only these numerical packages.
        previous_packages = metadata["environment"]["packages"]
        current_packages = initial_environment["packages"]
        for package in ("torch", "mup", "numpy"):
            previous = previous_packages.get(package)
            current = current_packages.get(package)
            if previous != current:
                raise ValueError(f"Resume requires {package}=={previous}. This runtime has {current}.")
        if metadata["environment"]["git_commit"] != initial_environment["git_commit"]:
            raise ValueError("Resume refused: repository commit changed")
        initial_environment = metadata["environment"]
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous["fingerprint"] != fingerprint:
                raise ValueError("Result belongs to a different run")
            if previous["status"] in ("complete", "diverged"):
                if is_main and previous["status"] == "complete":
                    final_name = validation_checkpoint_name(total_steps)
                    if json.loads(latest_path.read_text())["file"] != final_name or not all(
                        (checkpoint_dir / validation_checkpoint_name(s)).is_file() for s in checks
                    ):
                        raise ValueError("Completed main run is missing a validation checkpoint")
                    previous["checkpoint"] = str(checkpoint_dir / final_name)
                if remote:
                    sync_to_remote(run_dir, remote)
                return previous
    else:
        if resume:
            raise FileNotFoundError(f"No run to resume in {run_dir}")
        if checkpoint_dir.exists():
            raise ValueError("Found an existing checkpoint without its run metadata")
        metadata = {**identity, "fingerprint": fingerprint, "environment": initial_environment}
        if is_main:
            metadata["checkpoint_format"] = 3
        dump_json(run_dir / "run.json", metadata)
    if resume:
        if is_main and latest_path.is_file():
            saved = load_checkpoint(checkpoint_dir)
            step, consumed = saved["step"], saved["consumed_tokens"]
            if (
                saved.get("format_version") != 3
                or saved["fingerprint"] != fingerprint
                or step not in checks
                or consumed != min(budget, step * tokens_per_update)
            ):
                raise ValueError("Resume checkpoint does not match this run or its validation schedule")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            # The next schedule factor is a pure function of the restored step and run config.
            schedule.peak_lrs = saved["peak_lrs"]
            _restore_rng(saved["rng"], device)
            final_validation = saved["validation_loss"]
            # Restore the small metrics log to exactly the checkpoint boundary.
            temporary = metrics_path.with_suffix(".jsonl.tmp")
            temporary.write_bytes(saved["metrics_log"])
            temporary.replace(metrics_path)
            del saved
            print(f"Resuming after validation at update {step:,} ({consumed:,} tokens).", flush=True)
        else:
            print("Restarting the unfinished run from its original seed.", flush=True)
            metrics_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
    if not (run_dir / "parameter_groups.json").exists():
        dump_json(run_dir / "parameter_groups.json", parameter_report(model, optimizer))

    stop_step = (
        total_steps if stop_after_updates is None else min(total_steps, step + max(0, stop_after_updates))
    )
    label = f"{model_config.family} · {run_dir.name}"
    if progress:
        progress(label, "update", step, stop_step)
    failure = {}
    try:
        while step < stop_step:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            schedule.apply(step)
            update_targets = min(tokens_per_update, budget - consumed)
            update_loss, gradient_norm = _optimizer_update(
                model, optimizer, corpus, config, device, consumed, update_targets
            )
            consumed += update_targets
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            metrics = {
                "step": step,
                "consumed_tokens": consumed,
                "training_loss": update_loss,
                "gradient_norm": gradient_norm,
                "lr": [g["lr"] for g in optimizer.param_groups],
                "tokens_per_second": update_targets / elapsed,
            }
            if progress:
                detail = (
                    f"step {step} · loss {update_loss:.4f} · grad norm {gradient_norm:.2e} · "
                    f"{metrics['tokens_per_second']:,.0f} tok/s"
                )
                progress(label, "update", step, stop_step, detail)
            if device.type == "cuda":
                metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
            if step in checks:
                final_validation = validate(model, validation, config, device)
                if not math.isfinite(final_validation):
                    status = "diverged"
                    failure = {"failed_at_step": step, "failure_reason": "Nonfinite validation loss"}
                    break
                metrics["validation_loss"] = final_validation
            if step % config.log_interval == 0 or consumed == budget or "validation_loss" in metrics:
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                if not progress or "validation_loss" in metrics:
                    print(json.dumps(metrics), flush=True)
                if is_main and step in checks:
                    save_validation_checkpoint(
                        checkpoint_dir,
                        {
                            "format_version": 3,
                            "fingerprint": fingerprint,
                            "identity": identity,
                            "environment": initial_environment,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "peak_lrs": schedule.peak_lrs,
                            "rng": _rng_state(device),
                            "step": step,
                            "consumed_tokens": consumed,
                            "validation_loss": final_validation,
                            "metrics_log": metrics_path.read_bytes(),
                        },
                    )
                    if remote:
                        sync_to_remote(run_dir, remote)
        else:
            status = "complete" if consumed == budget else "paused"
    except FloatingPointError as error:
        status = "diverged"
        failure = {"failed_at_step": step + 1, "failure_reason": str(error)}
    except KeyboardInterrupt:
        status = "interrupted"
    if failure:
        print(f"Run diverged at update {failure['failed_at_step']}: {failure['failure_reason']}", flush=True)
    latest = json.loads(latest_path.read_text()) if is_main and latest_path.is_file() else None
    result = {
        "status": status,
        "step": step,
        "consumed_tokens": consumed,
        "token_budget": budget,
        "validation_loss": final_validation if status == "complete" else None,
        "fingerprint": fingerprint,
        "checkpoint": str(checkpoint_dir / latest["file"]) if latest else None,
        **failure,
    }
    dump_json(result_path, result)
    if remote:
        sync_to_remote(run_dir, remote)
    return result
