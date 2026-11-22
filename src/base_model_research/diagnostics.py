import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import Hyperparameters, ModelConfig, TrainConfig, dump_json
from .models import ResearchLM, parameter_count
from .parameterization import build_model, make_optimizer
from .training import autocast_context, environment_report, seed_everything

DEFAULT_HP = Hyperparameters(0.001, 1.0, 1.0, 1.0, 1.0)
EXPECTED_COUNTS = {
    "standard_proxy": 13_326_336,
    "modified_proxy": 13_362_624,
    "standard_main": 157_814_784,
    "modified_main": 158_395_392,
}


def check_parameter_counts(config_directory="configs/models"):
    results = {}
    for name, expected in EXPECTED_COUNTS.items():
        config = ModelConfig.load(Path(config_directory) / f"{name}.yaml")
        with torch.device("meta"):
            model = ResearchLM(config, DEFAULT_HP)
        actual = parameter_count(model)
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
        results[name] = {
            "parameters": actual,
            "training_tokens": 20 * actual,
            "optimizer_updates": math.ceil(20 * actual / (128 * 2048)),
        }
    return results


def smoke_check(device="cpu"):
    results = {}
    for family in ("standard", "modified"):
        seed_everything(42)
        config = ModelConfig(
            family, 16, 2, 2, 1, 8, 32, vocab_size=64, base_d_model=16, base_head_dim=8, base_d_ff=32
        )
        model = build_model(config, DEFAULT_HP, device)
        optimizer = make_optimizer(model, TrainConfig())
        x = torch.randint(0, 64, (2, 8), device=device)
        y = torch.randint(0, 64, (2, 8), device=device)
        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss_sum(x, y, 8) / y.numel()
            loss.backward()
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite smoke-test loss")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        if losses[-1] >= losses[0]:
            raise AssertionError(f"Tiny {family} model failed to reduce training loss")
        results[family] = {"losses": losses, "parameters": parameter_count(model)}
    return results


def coordinate_check(config, output, device="cuda", widths=None, steps=3):
    """Record activation magnitudes across widths; inspect for growth/collapse.

    This is a diagnostic, not a proof of mu-transfer for the modified architecture.
    It preserves the family's layer/head counts and proportional head/FFN widths.
    """
    if widths is None:
        widths = [config.base_d_model * multiple for multiple in (1, 2, 4)]
    records = []
    for width in widths:
        seed_everything(42)
        variant = config.at_width(width / config.base_d_model)
        model = build_model(variant, DEFAULT_HP, device)
        optimizer = make_optimizer(model, TrainConfig())
        current = {}
        hooks = []
        for name, module in model.named_modules():
            if (
                name == "embedding"
                or name == "final_norm"
                or name.endswith((".attention", ".ffn", ".attention_gate", ".ffn_gate"))
            ):

                def record(_module, _inputs, result, key=name, values=current):
                    values[key] = result.detach().float().abs().mean().item()

                hooks.append(module.register_forward_hook(record))
        x = torch.randint(0, config.vocab_size, (1, 16), device=device)
        y = torch.randint(0, config.vocab_size, (1, 16), device=device)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss_sum(x, y, 16) / y.numel()
            records.extend(
                {"width": width, "step": step, "module": key, "mean_absolute": value}
                for key, value in current.items()
            )
            print(f"coordinate check width={width} step={step} loss={loss.item():.5f}", flush=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        for hook in hooks:
            hook.remove()
        del model, optimizer
    dump_json(output, {"model": asdict(config), "hyperparameters": asdict(DEFAULT_HP), "records": records})
    return records


def profile(config, training, output, device="cuda", warmup_updates=2, measured_updates=3):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Throughput profiling requires the actual CUDA training runtime")
    seed_everything(42)
    model = build_model(config, DEFAULT_HP, device)
    model.activation_checkpointing = training.activation_checkpointing
    optimizer = make_optimizer(model, training)
    x = torch.randint(
        0, config.vocab_size, (training.microbatch_size, training.context_length), device=device
    )
    y = torch.randint(0, config.vocab_size, x.shape, device=device)
    accumulation = training.global_batch_size // training.microbatch_size
    times = []
    torch.cuda.reset_peak_memory_stats()
    for update in range(warmup_updates + measured_updates):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with autocast_context(device, training.precision):
                loss = model.loss_sum(x, y, training.loss_chunk_size) / (x.numel() * accumulation)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"profile update={update + 1} seconds={elapsed:.3f}", flush=True)
        if update >= warmup_updates:
            times.append(elapsed)
    tokens_per_update = training.global_batch_size * training.context_length
    throughput = tokens_per_update / (sum(times) / len(times))
    report = {
        "environment": environment_report(),
        "model": asdict(config),
        "training": asdict(training),
        "seconds_per_update": times,
        "tokens_per_second": throughput,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "estimated_training_hours": parameter_count(model)
        * training.tokens_per_parameter
        / throughput
        / 3600,
        "note": "Synthetic batches; excludes data loading, validation, checkpointing, and Drive transfer",
    }
    dump_json(output, report)
    return report
