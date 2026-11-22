import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_TRAINING_TARGETS,
    SHARED_TRAINING_FIELDS,
    Hyperparameters,
    ModelConfig,
    TrainConfig,
    dump_json,
)


def main():
    parser = argparse.ArgumentParser(prog="bmr")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("trials", help="Create/verify the shared deterministic trial manifest")
    p.add_argument("--config", default="configs/search.yaml")
    p.add_argument("--output", default="experiments/trials_seed42.json")
    p = commands.add_parser("prepare-data")
    p.add_argument("--output", required=True)
    p.add_argument("--training-tokens", type=int, default=DEFAULT_TRAINING_TARGETS)
    p.add_argument("--validation-documents", type=int, default=2048)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-100BT")
    p.add_argument("--dataset-revision")
    p.add_argument("--tokenizer-revision")
    p = commands.add_parser("copy-data", help="Copy a prepared corpus and verify its token shards")
    p.add_argument("--source", required=True)
    p.add_argument("--destination", required=True)
    p = commands.add_parser("check")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output")
    p = commands.add_parser("environment")
    p.add_argument("--output")
    for name in ("train", "sweep", "profile", "coordinate-check"):
        p = commands.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--output", required=True)
        if name == "coordinate-check":
            p.add_argument("--widths", nargs="+", type=int, help="Default: 1×, 2×, and 4× the proxy width")
            p.add_argument("--steps", type=int, default=3)
            continue
        p.add_argument("--training", default="configs/training.yaml")
        p.add_argument("--microbatch-size", type=int)
        p.add_argument("--precision", choices=["fp32", "bf16"])
        p.add_argument("--warmup-fraction", type=float)
        p.add_argument("--min-lr-ratio", type=float)
        p.add_argument("--activation-checkpointing", action="store_true", default=None)
        if name == "profile":
            continue
        p.add_argument("--data", required=True)
        p.add_argument(
            "--cache-dir", help="Local SSD directory for verified token shards copied on first use"
        )
        p.add_argument("--stop-after-updates", type=int)
        p.add_argument(
            "--progress-events", action="store_true", help="Stream counters for notebook progress bars"
        )
        p.add_argument("--remote", help="rclone destination; runtime credentials stay outside the repository")
        if name == "train":
            p.add_argument("--hyperparameters", required=True, help="A best_hparams.json or plain HP object")
            p.add_argument(
                "--resume",
                action="store_true",
                help="Continue a main run from its latest validation checkpoint",
            )
        else:
            p.add_argument("--trials", default="experiments/trials_seed42.json")
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--end", type=int)
    p = commands.add_parser("summarize")
    p.add_argument("--output", required=True, help="Family directory, e.g. runs/proxy/standard")
    p.add_argument("--family", required=True, choices=["standard", "modified"])
    p.add_argument("--trials", default="experiments/trials_seed42.json")
    p = commands.add_parser("benchmark")
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--standard", help="Standard main checkpoint directory or file")
    p.add_argument("--modified", help="Modified main checkpoint directory or file")
    p.add_argument("--tokenizer", help="Prepared data/tokenizer directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, help="Debug samples per task; results are labelled limited")
    p.add_argument("--no-references", action="store_true")
    p = commands.add_parser("sync")
    p.add_argument("--source", required=True)
    p.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.command == "trials":
        from .search import generate_trials

        result = generate_trials(args.config, args.output)
        print(f"{len(result['trials'])} trials; fingerprint={result['fingerprint']}")
    elif args.command == "prepare-data":
        from .data import prepare_data

        result = prepare_data(
            args.output,
            args.training_tokens,
            args.validation_documents,
            args.dataset,
            args.dataset_config,
            dataset_revision=args.dataset_revision,
            tokenizer_revision=args.tokenizer_revision,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "copy-data":
        from .storage import copy_prepared_data

        copy_prepared_data(args.source, args.destination)
        print(f"Prepared corpus available at {args.destination}")
    elif args.command in ("check", "environment"):
        from .diagnostics import check_parameter_counts, smoke_check
        from .training import environment_report

        result = environment_report()
        if args.command == "check":
            result = {
                "environment": result,
                "parameter_counts": check_parameter_counts(),
                "smoke": smoke_check(args.device),
            }
        if args.output:
            dump_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command in ("train", "sweep", "profile", "coordinate-check"):
        model = ModelConfig.load(args.model)
        if args.command == "coordinate-check":
            from .diagnostics import coordinate_check

            coordinate_check(model, args.output, args.device, args.widths, args.steps)
            return
        training = TrainConfig.load(
            args.training,
            microbatch_size=args.microbatch_size,
            precision=args.precision,
            warmup_fraction=args.warmup_fraction,
            min_lr_ratio=args.min_lr_ratio,
            activation_checkpointing=args.activation_checkpointing,
        )
        if args.command == "profile":
            from .diagnostics import profile

            print(json.dumps(profile(model, training, args.output, args.device), indent=2))
        elif args.command == "sweep":
            from .progress import emit_progress

            if model.width_mult != 1:
                raise ValueError("Proxy sweeps require the family's base width")
            from .search import sweep

            sweep(
                model,
                training,
                args.data,
                args.output,
                args.trials,
                args.device,
                args.start,
                args.end,
                args.stop_after_updates,
                args.remote,
                progress=emit_progress if args.progress_events else None,
                cache_dir=args.cache_dir,
            )
        else:
            from .progress import emit_progress
            from .training import train

            saved = json.loads(Path(args.hyperparameters).read_text())
            hp = Hyperparameters(**saved.get("hyperparameters", saved))
            if "family" in saved and saved["family"] != model.family:
                raise ValueError("Winner belongs to the other model family")
            if model.width_mult != 1:
                if "selection" not in saved:
                    raise ValueError("Main runs require a finalized proxy best_hparams.json")
                proxy_training = TrainConfig.from_record(saved["training"])
                # Batch geometry and the training objective remain fixed; microbatch
                # and logging intervals may differ for a larger model.
                for key in SHARED_TRAINING_FIELDS:
                    if getattr(training, key) != getattr(proxy_training, key):
                        raise ValueError(f"Main/proxy training protocol mismatch: {key}")
                from .data import TokenCorpus

                if (
                    TokenCorpus(args.data, "train", cache_dir=args.cache_dir).fingerprint
                    != saved["data_fingerprint"]
                ):
                    raise ValueError("Main run must use the same prepared dataset as proxy selection")
                if model.at_width(1) != ModelConfig(**saved["model"]):
                    raise ValueError("Main model does not match the selected proxy's architecture family")
            result = train(
                model,
                hp,
                training,
                args.data,
                args.output,
                args.device,
                args.resume,
                args.stop_after_updates,
                args.remote,
                selection=saved if "selection" in saved else None,
                progress=emit_progress if args.progress_events else None,
                cache_dir=args.cache_dir,
            )
            print(json.dumps(result, indent=2))
            if result["status"] == "diverged":
                raise SystemExit(2)
    elif args.command == "summarize":
        from .search import load_trials, summarize

        print(json.dumps(summarize(args.output, load_trials(args.trials), args.family), indent=2))
    elif args.command == "benchmark":
        from .evaluation import benchmark

        result = benchmark(
            args.config,
            args.output,
            args.device,
            args.standard,
            args.modified,
            args.tokenizer,
            args.limit,
            not args.no_references,
        )
        print(f"Saved {len(result)} metric rows")
    else:
        from .storage import sync_to_remote

        sync_to_remote(args.source, args.destination)


if __name__ == "__main__":
    main()
