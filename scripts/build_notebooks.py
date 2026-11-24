"""Generate the experiment notebooks from the shared Python implementation."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(text).strip() + "\n",
    }


def storage_cell(experiment, data_folder):
    return code(f"""
        # Experiment storage
        USE_GOOGLE_DRIVE = True

        from pathlib import Path
        import os

        default_workspace = "/content/bmr" if Path("/content").is_dir() else "~/bmr-workspace"
        WORKSPACE = Path(os.environ.get("BMR_WORKSPACE", default_workspace)).expanduser().resolve()
        EXPERIMENT = "{experiment}"
        DATA = WORKSPACE / "data" / "{data_folder}"
        if USE_GOOGLE_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")
            my_drive = Path("/content/drive/MyDrive")
            if not my_drive.is_dir():
                raise RuntimeError("Mount Google Drive before continuing.")
            storage = my_drive / "bmr"
        else:
            storage = WORKSPACE
        RUNS = storage / "runs" / EXPERIMENT
        SAVED_DATA = storage / "data" / "{data_folder}"
        print("Experiment outputs:", RUNS)
    """)


BOOTSTRAP = """
    def run(*arguments):
        subprocess.run([sys.executable, "-m", "base_model_research.cli", *map(str, arguments)],
                       cwd=REPO, check=True)
"""


TRAINING_LAUNCHER = """
    # Training runs in a subprocess of the same Python; tqdm renders in the notebook.
    sys.path.insert(0, str(REPO / "src"))
    from base_model_research.progress import run_with_progress

    def training_data():
        print(f"Checking the prepared dataset at {SAVED_DATA}…", flush=True)
        if not (SAVED_DATA / "manifest.json").is_file():
            raise FileNotFoundError(f"Run notebook 01 to prepare the shared dataset at {SAVED_DATA}.")
        if SAVED_DATA == DATA:
            return ["--data", SAVED_DATA]
        local_manifest = DATA / "manifest.json"
        if local_manifest.is_file() and local_manifest.read_bytes() == (SAVED_DATA / "manifest.json").read_bytes():
            print("Using the prepared copy on the runtime SSD.", flush=True)
            return ["--data", DATA]
        from inspect import signature
        from base_model_research.data import TokenCorpus
        if "cache_dir" in signature(TokenCorpus).parameters:
            print("Training will cache each shard on the runtime SSD when it is first read.", flush=True)
            return ["--data", SAVED_DATA, "--cache-dir", DATA.parent / "cached_shards"]
        print("The pinned trainer requires a full local copy. Copying the prepared corpus…", flush=True)
        run("copy-data", "--source", SAVED_DATA, "--destination", DATA)
        return ["--data", DATA]

    def run_training(command_name, model_name, run_record, *arguments):
        # Existing runs keep their settings. New runs use the profile from 00 or the training config.
        print(f"Reading saved settings for {model_name}…", flush=True)
        settings_file = run_record if run_record.is_file() else RUNS / f"{model_name}_profile.json"
        memory_options = []
        if settings_file.is_file():
            settings = json.loads(settings_file.read_text())["training"]
            memory_options = ["--microbatch-size", str(settings["microbatch_size"])]
            if settings["activation_checkpointing"]:
                memory_options.append("--activation-checkpointing")
        command = [sys.executable, "-u", "-m", "base_model_research.cli", command_name,
                   "--model", f"configs/models/{model_name}.yaml", *map(str, arguments),
                   *memory_options, "--progress-events"]
        print(f"Launching {model_name}. The update bar appears after the trainer checks "
              "saved runs and initializes the model.", flush=True)
        run_with_progress(command, cwd=REPO)
"""


def write_notebook(name, cells, metadata):
    content = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            **metadata,
        },
        "cells": cells,
    }
    for index, cell in enumerate(content["cells"]):
        cell["id"] = f"cell-{index:03d}"
    path = ROOT / "notebooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


def notebook(
    name,
    title,
    cells,
    *,
    proxy_controls=None,
    training=False,
    benchmark=False,
    gpu=True,
    experiment="standard_vs_gated_160m_tinyllama",
    data_folder="fineweb_edu_tinyllama",
):
    setup = (ROOT / "scripts" / "colab_setup.py").read_text()
    bootstrap = (
        "# Set up experiment runtime\n"
        + setup
        + f"\nREPO = setup_colab(WORKSPACE, RUNS, benchmark={benchmark})\n"
    )
    if proxy_controls is not None:
        bootstrap += textwrap.dedent("""
            sys.path.insert(0, str(REPO / "src"))
            from base_model_research.progress import run_with_progress

        """)
        bootstrap += (ROOT / "scripts" / "colab_proxy.py").read_text()
    else:
        bootstrap += textwrap.dedent(BOOTSTRAP)
        if training:
            bootstrap += textwrap.dedent(TRAINING_LAUNCHER)
    runtime = "Select the RTX PRO 6000 GPU" if gpu else "Select a CPU runtime"
    write_notebook(
        name,
        [
            markdown(
                f"# {title}\n\n{runtime} in Colab, run the cells in order, "
                "and approve the Google Drive mount when prompted."
            ),
            storage_cell(experiment, data_folder),
            *([proxy_controls] if proxy_controls is not None else []),
            code(bootstrap),
            *cells,
        ],
        {**({"accelerator": "GPU"} if gpu else {}), "colab": {"name": name}},
    )


notebook(
    "00_setup_and_checks.ipynb",
    "Setup, correctness checks, and GPU profiling",
    [
        code("""
        subprocess.run(["nvidia-smi"], check=True)
        run("environment", "--output", RUNS / "environment.json")
        run("check", "--device", "cpu", "--output", RUNS / "checks.json")
    """),
        markdown(
            "Run these GPU diagnostics before full sweeps. Inspect activation magnitudes across widths "
            "for collapse or growth; the new gated architecture has no guaranteed transfer result."
        ),
        code("""
        for family in ("standard", "modified"):
            run("coordinate-check", "--model", f"configs/models/{family}_proxy.yaml",
                "--output", RUNS / f"{family}_coordinates.json")
    """),
        markdown(
            "This cell profiles all four models using synthetic data. If a model runs out of GPU memory, "
            "reduce the microbatch size and rerun the cell. Training uses each model's saved settings."
        ),
        code("""
        MICROBATCH_SIZE = 4  # Must divide the global batch size of 128.
        for model in ("standard_proxy", "modified_proxy", "standard_main", "modified_main"):
            print(f"Profiling {model}…", flush=True)
            run("profile", "--model", f"configs/models/{model}.yaml",
                "--microbatch-size", MICROBATCH_SIZE, "--output", RUNS / f"{model}_profile.json")
    """),
    ],
)

notebook(
    "01_prepare_data.ipynb",
    "Prepare the shared FineWeb-Edu token corpus",
    [
        markdown(
            "Prepare once and reuse for every run. Default: 2,048 held-out whole documents, followed by "
            "3,167,907,840 training targets plus one lookahead token. uint32 tokens occupy about 12.7 GB "
            "in decimal units, plus validation, tokenizer and caches. If preparation stops, rerun this cell. "
            "Tokenization starts over if needed. A partial Drive copy resumes from verified shards."
        ),
        code("""
        import shutil
        from base_model_research.config import DEFAULT_TOKENIZER, DEFAULT_TRAINING_TARGETS

        print("Free disk GB:", shutil.disk_usage(WORKSPACE).free / 1e9)
        prepared = SAVED_DATA if (SAVED_DATA / "manifest.json").is_file() else DATA
        if not (prepared / "manifest.json").is_file():
            run("prepare-data", "--output", prepared)
        manifest = json.loads((prepared / "manifest.json").read_text())
        if (manifest["tokenizer"] != DEFAULT_TOKENIZER
                or manifest["splits"]["train"]["tokens"] < DEFAULT_TRAINING_TARGETS + 1):
            raise ValueError(f"Expected the full TinyLlama-tokenized experiment corpus at {prepared}.")
        if prepared != SAVED_DATA:
            run("copy-data", "--source", prepared, "--destination", SAVED_DATA)
        print("Prepared corpus:", SAVED_DATA)
    """),
    ],
    gpu=False,
)

write_notebook(
    "local/01_prepare_data.ipynb",
    [
        markdown("""
            # Prepare FineWeb-Edu on your Mac

            Open this notebook in the project with your local Python 3 kernel and choose **Run All**.
            Your CPU streams FineWeb-Edu, tokenizes with the TinyLlama 32k tokenizer, and saves the corpus in
            **Downloads → fineweb_edu_tinyllama**. Allow about 12.7 GB for tokens plus room for caches.
            After the validation documents, the progress bar shows training tokens per second and time remaining.
            Keep the Mac awake while this runs. Interrupting tokenization discards the partial run.
        """),
        code("""
            from pathlib import Path
            import shutil
            import sys

            # Change this path if you want to use an external disk.
            OUTPUT = Path.home() / "Downloads" / "fineweb_edu_tinyllama"
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            print(f"Output folder: {OUTPUT}")
            print(f"Free disk space: {shutil.disk_usage(OUTPUT.parent).free / 1e9:.1f} GB")

            cwd = Path.cwd()
            repo = next((p for p in (cwd, *cwd.parents) if (p / "src/base_model_research/data.py").is_file()), None)
            if repo is None:
                raise RuntimeError("Open this notebook inside the base-model-research project folder.")
            sys.path.insert(0, str(repo / "src"))
            from base_model_research.config import DEFAULT_TOKENIZER, DEFAULT_TRAINING_TARGETS
            from base_model_research.data import TokenCorpus, prepare_data
            from tqdm.auto import tqdm
        """),
        code("""
            if not (OUTPUT / "manifest.json").is_file():
                print("Opening FineWeb-Edu. The first 2,048 documents go to validation.", flush=True)
                with tqdm(
                    total=DEFAULT_TRAINING_TARGETS + 1, desc="Training tokens", unit="tok", unit_scale=True, mininterval=1
                ) as progress:
                    prepare_data(OUTPUT, on_progress=progress.update)

            train = TokenCorpus(OUTPUT, "train")
            validation = TokenCorpus(OUTPUT, "validation")
            manifest = train.manifest
            if (
                train.total_tokens < DEFAULT_TRAINING_TARGETS + 1
                or manifest.get("dataset") != "HuggingFaceFW/fineweb-edu"
                or manifest.get("dataset_config") != "sample-100BT"
                or manifest.get("tokenizer") != DEFAULT_TOKENIZER
                or manifest.get("validation_documents") != 2048
                or not (OUTPUT / "tokenizer" / "tokenizer_config.json").is_file()
            ):
                raise ValueError(f"The folder at {OUTPUT} does not contain the full experiment corpus.")
            print(f"Training targets: {train.total_tokens - 1:,}   Validation tokens: {validation.total_tokens:,}")
            print(f"Ready to upload: {OUTPUT}")
        """),
        markdown("""
            When the cell prints **Ready to upload**, drag the entire **fineweb_edu_tinyllama** folder from Finder
            into **My Drive → bmr → data** in your browser. After the upload finishes, the manifest should
            be at `My Drive/bmr/data/fineweb_edu_tinyllama/manifest.json`. Continue with **02_standard_proxy_search**
            in Colab.
        """),
    ],
    {},
)

for family, number in (("standard", "02"), ("modified", "03")):
    notebook(
        f"{number}_{family}_proxy_search.ipynb",
        f"{family.title()} 13.3M proxy: 32 shared trials",
        [
            code(f"""
            # Copy proxy data from Google Drive to /content
            PROXY_DATA = stage_proxy_data(SAVED_DATA, DATA, proxy_training_targets(REPO, "{family}"))
            print(f"Proxy data ready at {{PROXY_DATA}}.")
        """),
            code(f"""
            FAMILY_RUNS = RUNS / "proxy" / "{family}"

            def run_proxy(*, first_only=False):
                sweep_args = proxy_sweep_args(REPO, RUNS, "{family}", MICROBATCH_SIZE, PROXY_DATA)
                options = ["--end", 1] if first_only else []
                print("Starting the {family} proxy trainer…", flush=True)
                run_with_progress([sys.executable, "-u", "-m", "base_model_research.cli", "sweep",
                                   *sweep_args, *map(str, options)], cwd=REPO)
        """),
            code("""
            # Smoke run · train trial 000 on 20 tokens per parameter
            if RUN_SMOKE:
                run_proxy(first_only=True)
                result = json.loads((FAMILY_RUNS / "trial_000" / "result.json").read_text())
                print(f"Trial 000 is {result['status']} at step {result['step']}.")
            else:
                print("Smoke run disabled.")
        """),
            code("""
            # Full run · all 32 trials
            if RUN_FULL:
                run_proxy()
                winner = FAMILY_RUNS / "best_hparams.json"
                if winner.exists():
                    best = json.loads(winner.read_text())
                    print(f"Winner: {best['trial_id']} · validation loss {best['validation_loss']:.4f}")
                    print(best["hyperparameters"])
                else:
                    print("No winner yet. See", FAMILY_RUNS / "results.csv")
            else:
                print("Full run disabled.")
        """),
        ],
        proxy_controls=code("""
            # Run controls
            MICROBATCH_SIZE = 4  # Sequences per GPU microbatch. Choose a divisor of 128.
            RUN_SMOKE = True
            RUN_FULL = False
            # RUN_SMOKE trains one full trial (~267M tokens, 20 validations).
            # RUN_FULL runs the 32-trial search and skips completed trials, including trial 000.
        """),
    )

notebook(
    "compare_gate_variants.ipynb",
    "Compare three gated proxies on trial 000",
    [
        markdown("""
            Each proxy uses **trial 000**, the settings under which the original gated proxy diverged.
            All three use seed 42, the same ordered TinyLlama-tokenized FineWeb-Edu data, and the same
            held-out validation set. They run sequentially for 20 training tokens per parameter with
            20 validations per full run. The parameter counts and total updates can differ.

            In the table, `x` is the input to the sublayer, `y` its output, and `p` the previous
            layer's output from the **same kind of sublayer**. Multiplication `⊙` is elementwise.

            | Variant | Output passed to the next sublayer |
            |---|---|
            | 1 · add_previous | GELU(W₁x) ⊙ (W₂y) + p |
            | 2 · scale_previous | GELU(W₁x) ⊙ (W₂y) + αp |
            | 3 · multiply_previous | GELU(ReLU(x + y)) ⊙ (Wp) |

            The first layer uses ordinary residual addition because it has no previous layer.
            Every later layer applies its formula to both attention and the FFN, with independent
            weights. Each α in variant 2 is a learned scalar initialized to 1.
        """),
        code("""
            from dataclasses import replace
            from base_model_research.config import DEFAULT_TOKENIZER, GATE_VARIANTS, Hyperparameters
            from base_model_research.search import load_trials

            trial = next(t for t in load_trials(REPO / "experiments/trials_seed42.json")["trials"]
                         if t["id"] == "trial_000")
            hp = Hyperparameters(**trial["hyperparameters"])
            settings = TrainConfig.load(REPO / "configs/training.yaml", microbatch_size=MICROBATCH_SIZE)
            proxy = ModelConfig.load(REPO / "configs/models/modified_proxy.yaml")
            models = [replace(proxy, family=name) for name in GATE_VARIANTS]
            with torch.device("meta"):
                counts = {m.family: parameter_count(ResearchLM(m, hp)) for m in models}
            print(f"Shared settings: {trial['id']}, seed {settings.seed}, microbatch {settings.microbatch_size}")
            print(json.dumps(trial["hyperparameters"], indent=2))
            for name, count in counts.items():
                print(f"{name}: {count:,} parameters, {count * settings.tokens_per_parameter:,} training tokens")
        """),
        code("""
            # Copy the shards needed by the largest variant to the Colab SSD.
            manifest = SAVED_DATA / "manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"Expected your prepared TinyLlama corpus at {manifest}.")
            if json.loads(manifest.read_text())["tokenizer"] != DEFAULT_TOKENIZER:
                raise ValueError(f"Expected the TinyLlama-tokenized FineWeb-Edu corpus at {SAVED_DATA}.")
            PROXY_DATA = stage_proxy_data(SAVED_DATA, DATA, max(counts.values()) * settings.tokens_per_parameter)
        """),
        code("""
            from contextlib import chdir
            from math import ceil
            from base_model_research.training import train

            def train_variant(model):
                output = RUNS / model.family / trial["id"]
                total = ceil(counts[model.family] * settings.tokens_per_parameter /
                             (settings.global_batch_size * settings.context_length))
                print(f"Checking saved results and initializing {model.family}…", flush=True)
                with chdir(REPO), tqdm(total=total, desc=model.family, unit="update", delay=1) as bar:
                    def show_update(label, unit, completed, target, detail=""):
                        bar.set_postfix_str(detail, refresh=False)
                        bar.update(completed - bar.n)

                    return train(model, hp, settings, PROXY_DATA, output,
                                 resume=(output / "run.json").is_file(),
                                 selection={"trial_id": trial["id"]}, progress=show_update)

            for model in models:
                result = train_variant(model)
                print(f"{model.family}: {result['status']} after {result['step']:,} completed updates.")
                if result["status"] == "diverged":
                    print(f"Update {result['failed_at_step']}: {result['failure_reason']}.")
                if result["status"] == "interrupted":
                    break
        """),
        markdown("""
            **Read the results.** A nonfinite training loss, gradient norm, or validation loss marks
            that variant as diverged. The next variant then starts. Compare final validation losses
            among completed runs. A loss near 10.37 is what uniform predictions give with this
            32,000-token vocabulary, so a finite loss alone can hide a model that learned very little.
            Watch the gradient norm in the progress bar. If it stays at zero while loss stays near
            10.37, the model may be stuck predicting every token with equal probability.

            Rerunning the training cell reuses completed and diverged results. An interrupted variant
            restarts from its seed. You can run the summary cell separately to inspect saved results.
        """),
        code("""
            import pandas as pd

            rows = []
            for model in models:
                folder = RUNS / model.family / trial["id"]
                path = folder / "result.json"
                result = json.loads(path.read_text()) if path.is_file() else {}
                metrics = folder / "metrics.jsonl"
                logged = metrics.read_text().splitlines() if result and metrics.is_file() else []
                last = json.loads(logged[-1]) if logged else {}
                rows.append({
                    "variant": model.family, "parameters": counts[model.family],
                    "status": result.get("status", "no result yet"),
                    "completed_updates": result.get("step"), "failed_at_update": result.get("failed_at_step"),
                    "failure_reason": result.get("failure_reason"), "training_tokens": result.get("consumed_tokens"),
                    "last_logged_gradient_norm": last.get("gradient_norm"),
                    "final_validation_loss": result.get("validation_loss"),
                })
            summary = pd.DataFrame(rows).convert_dtypes()
            summary_path = RUNS / "comparison.csv"
            temporary = summary_path.with_suffix(".csv.tmp")
            with run_lock(RUNS, blocking=True):
                summary.to_csv(temporary, index=False)
                temporary.replace(summary_path)
            print(f"Saved comparison: {summary_path}")
            summary
        """),
    ],
    proxy_controls=code("""
        MICROBATCH_SIZE = 16  # Sequences per GPU microbatch. Must divide 128.
    """),
    experiment="proxy_gate_variants",
)


def trial_000_proxy_cells(*, family, backbone, label, startup_message):
    return [
        code(f"""
            from dataclasses import replace
            from math import ceil
            from base_model_research.config import DEFAULT_TOKENIZER, Hyperparameters
            from base_model_research.search import load_trials

            trial = next(t for t in load_trials(REPO / "experiments/trials_seed42.json")["trials"]
                         if t["id"] == "trial_000")
            hp = Hyperparameters(**trial["hyperparameters"])
            model = replace(ModelConfig.load(REPO / "configs/models/{backbone}_proxy.yaml"), family="{family}")
            settings = TrainConfig.load(REPO / "configs/training.yaml", microbatch_size=MICROBATCH_SIZE)
            with torch.device("meta"):
                budget = parameter_count(ResearchLM(model, hp)) * settings.tokens_per_parameter
            updates = ceil(budget / (settings.context_length * settings.global_batch_size))
            OUTPUT = RUNS / trial["id"]
            print(f"{{trial['id']}} · seed {{settings.seed}} · {{budget:,}} tokens · {{updates:,}} optimizer updates")
            print(json.dumps(trial["hyperparameters"], indent=2))
            print(f"Results folder: {{OUTPUT}}")
        """),
        code("""
            # Copy the proxy's training prefix and validation shards to the runtime SSD.
            manifest = SAVED_DATA / "manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"Expected the prepared TinyLlama corpus at {manifest}.")
            if json.loads(manifest.read_text())["tokenizer"] != DEFAULT_TOKENIZER:
                raise ValueError(f"Expected TinyLlama-tokenized FineWeb-Edu at {SAVED_DATA}.")
            PROXY_DATA = stage_proxy_data(SAVED_DATA, DATA, budget)
        """),
        code(f"""
            from contextlib import chdir
            from base_model_research.training import train

            print("{startup_message}", flush=True)
            with chdir(REPO), tqdm(total=updates, desc="{label} · trial 000", unit="update", delay=1) as bar:
                def show_update(label, unit, completed, total, detail=""):
                    bar.set_postfix_str(detail, refresh=False)
                    bar.update(completed - bar.n)

                result = train(model, hp, settings, PROXY_DATA, OUTPUT,
                               resume=(OUTPUT / "run.json").is_file(),
                               selection={{"trial_id": trial["id"]}}, progress=show_update)
            print(f"{{result['status']}} after {{result['step']:,}} completed updates.")
            if result.get("failure_reason"):
                print(f"Update {{result['failed_at_step']}}: {{result['failure_reason']}}")
        """),
    ]


notebook(
    "trial_000_residual_tanh_proxy.ipynb",
    "Trial 000 · modified proxy with a direct residual and tanh gate",
    [
        markdown("""
            Both attention and the FFN use **x + y + tanh(GELU(W₁x) ⊙ W₂p)**. Here y is the current
            sublayer output and p is the previous layer's output from the same kind of sublayer.
            In layer 1, p = x. Each sublayer has its own W₁ and W₂.

            This notebook trains one 13,362,624-parameter proxy on TinyLlama-tokenized FineWeb-Edu
            with trial 000's hyperparameters and seed 42. A full run uses 267,252,480 training targets
            and evaluates the held-out data 20 times. The first validation is at update 51.
        """),
        *trial_000_proxy_cells(
            family="residual_tanh",
            backbone="modified",
            label="Residual + tanh",
            startup_message="Initializing trial 000. Each update records all 112 gate matrices before clipping.",
        ),
        markdown("""
            **Inspect the run.** `gate_gradients.csv` stores one row per update and layer.
            The four columns give the RMS of the accumulated weight gradients before clipping.
            W₁ projects the current input and W₂ projects the previous output. The console prints
            examples from layers 1, 14, and 28 early in training. The heatmaps show every layer and update.

            Validation losses are saved to `metrics.jsonl` and printed as each evaluation finishes.
            The plot cell also works after an interruption. Rerunning training reuses a completed or
            diverged result. An interrupted proxy restarts from its seed and records a fresh trace.
        """),
        code("""
            import matplotlib.pyplot as plt
            from base_model_research.gate_gradients import plot_gate_training

            figure = plot_gate_training(OUTPUT)
            if figure is not None:
                plt.show()
        """),
    ],
    proxy_controls=code("""
        MICROBATCH_SIZE = 16  # Sequences per GPU microbatch. Choose a divisor of 128.
    """),
    experiment="proxy_residual_tanh",
)

notebook(
    "trial_000_previous_attention_relu_proxy.ipynb",
    "Trial 000 · ReLU proxy with previous-layer attention",
    [
        markdown(r"""
            This notebook trains one proxy with trial 000's hyperparameters and seed 42.
            Each attention sublayer computes $h_\ell = \operatorname{ReLU}(x_\ell + a_\ell + a_{\ell-1})$.
            Here $a_{\ell-1}$ is the preceding layer's projected attention output, with $a_0=0$.
            The FFN uses the standard SwiGLU and residual addition.

            The 28-layer proxy uses the standard backbone with FFN width 432, 16 query heads
            and 8 KV heads. It has 13,326,336 parameters and trains on 266,526,720 TinyLlama-tokenized
            FineWeb-Edu targets (20 per parameter), with 20 held-out validations.
        """),
        *trial_000_proxy_cells(
            family="previous_attention_relu",
            backbone="standard",
            label="Previous attention + ReLU",
            startup_message="Initializing the previous-attention ReLU proxy for trial 000.",
        ),
        code("""
            if result["validation_loss"] is not None:
                print(f"Final validation loss {result['validation_loss']:.4f}")
            print(f"Run result: {OUTPUT / 'result.json'}")
            print(f"Training and validation metrics: {OUTPUT / 'metrics.jsonl'}")
        """),
    ],
    proxy_controls=code("""
        MICROBATCH_SIZE = 16  # Sequences per GPU microbatch. Choose a divisor of 128.
    """),
    experiment="proxy_previous_attention_relu",
)

for family, number, proxy_number in (("standard", "04", "02"), ("modified", "05", "03")):
    notebook(
        f"{number}_train_{family}_main.ipynb",
        f"Train the {family} ~160M main model from scratch",
        [
            markdown(
                "Trains once through this family's token budget and validates at 5% intervals. "
                "Each of the 20 validations saves the model weights and optimizer state to Drive. "
                "The 20th save includes the final optimizer update. Rerun this notebook to continue "
                "from the latest completed save. Allow about 38 GB on Drive per main model."
            ),
            code(f'''
            OUTPUT = RUNS / "main" / "{family}"
            WINNER = RUNS / "proxy" / "{family}" / "best_hparams.json"
            if not WINNER.is_file():
                raise FileNotFoundError("Enable RUN_FULL in notebook {proxy_number} "
                                        f"and finish the {family} proxy search to create {{WINNER}}.")
            options = ["--resume"] if (OUTPUT / "run.json").exists() else []
            run_training("train", "{family}_main", OUTPUT / "run.json", "--hyperparameters", WINNER,
                *training_data(), "--output", OUTPUT, *options)
        '''),
        ],
        training=True,
    )

notebook(
    "time_standard_main.ipynb",
    "Time one standard 158M training run",
    [
        markdown(
            "Uses your existing `My Drive/bmr/data/fineweb_edu` upload. Colab reads its Qwen tokens "
            "and converts documents to TinyLlama tokens in small groups as training needs them. "
            "It draws one of the 32 hyperparameter combinations and trains on 3,156,295,680 targets "
            "(20 per parameter). You can run it before the proxy searches. The tqdm bar estimates "
            "the remaining time from completed optimizer updates. Give it several updates to settle. "
            "Validation and Drive writes can add time beyond the early estimate."
        ),
        code("""
            import random
            import sys

            sys.path.insert(0, str(REPO / "src"))
            from base_model_research.config import Hyperparameters, ModelConfig, TrainConfig
            from base_model_research.search import load_trials

            DRAW_SEED = 42  # The same seed selects the same hyperparameters when you reopen Colab.
            trials = load_trials(REPO / "experiments/trials_seed42.json")["trials"]
            trial = random.Random(DRAW_SEED).choice(trials)
            model = ModelConfig.load(REPO / "configs/models/standard_main.yaml")
            hp = Hyperparameters(**trial["hyperparameters"])
            output = RUNS / trial["id"]
            record = output / "run.json"
            profile = RUNS.parent / "standard_vs_gated_160m_tinyllama" / "standard_main_profile.json"
            if record.is_file():
                settings = TrainConfig.from_record(json.loads(record.read_text())["training"])
            else:
                memory = json.loads(profile.read_text())["training"] if profile.is_file() else {}
                settings = TrainConfig.load(REPO / "configs/training.yaml",
                                            microbatch_size=memory.get("microbatch_size"),
                                            activation_checkpointing=memory.get("activation_checkpointing"))

            print(f"Selected {trial['id']} with a microbatch of {settings.microbatch_size} sequences.")
            print(json.dumps(trial["hyperparameters"], indent=2))
            print(f"Timing run folder: {output}")
        """),
        markdown(
            "You can stop the next cell once you have a time estimate, or leave it running to finish. "
            "Each of the 20 validations saves weights and optimizer state to the timing run folder "
            "(about 38 GB total for a full run). Rerunning resumes from the latest completed save. "
            "Before the first save, it restarts from the original seed."
        ),
        code("""
            from contextlib import chdir, ExitStack
            from functools import partial
            from base_model_research.timing_data import timing_corpora
            from base_model_research.training import train
            from tqdm.auto import tqdm

            def train_one_model():
                if not (SAVED_DATA / "manifest.json").is_file():
                    raise FileNotFoundError(f"Expected the original FineWeb-Edu upload at {SAVED_DATA / 'manifest.json'}.")
                cache = DATA.parent / "cached_shards" if DATA != SAVED_DATA else None
                with chdir(REPO), ExitStack() as display:
                    bar = None

                    def show_update(label, unit, completed, total, detail=""):
                        nonlocal bar
                        if bar is None:
                            bar = display.enter_context(tqdm(desc=label, unit=unit, initial=completed, total=total))
                        bar.set_postfix_str(detail, refresh=False)
                        bar.update(completed - bar.n)

                    return train(model, hp, settings, SAVED_DATA, output, resume=record.is_file(),
                                 selection={"method": "random", "trial_id": trial["id"]},
                                 progress=show_update, cache_dir=cache,
                                 adapt_corpora=partial(timing_corpora, run_dir=output))

            print(f"Reading {SAVED_DATA}. The ETA includes on-demand data conversion.", flush=True)
            result = train_one_model()
            print(f"Run {result['status']} at update {result['step']:,}.")
        """),
    ],
    experiment="standard_158m_timing_fineweb_edu",
    data_folder="fineweb_edu",
)

notebook(
    "06_benchmarks.ipynb",
    "Compare our two main models and three pretrained references",
    [
        markdown(
            "Uses the final validation checkpoint from each completed main run. "
            "Initial config: zero-shot on all seven tasks, LAMBADA OpenAI variant, context cap 2048, "
            "base completion formatting. Each reference uses its own tokenizer. Change settings before "
            "starting the comparison and keep them identical across models. Existing model/task results "
            "are reused only when their identities and settings match."
        ),
        code("""
        # Use 10 for a quick check or None for the full benchmarks. Results are stored separately.
        LIMIT = None
        for family, notebook_number in (("standard", "04"), ("modified", "05")):
            result = RUNS / "main" / family / "result.json"
            if not result.is_file() or json.loads(result.read_text())["status"] != "complete":
                raise RuntimeError(f"Finish notebook {notebook_number} to benchmark the {family} main model.")
        OUTPUT = RUNS / ("benchmarks_debug" if LIMIT is not None else "benchmarks")
        options = ["--limit", LIMIT] if LIMIT is not None else []
        run("environment", "--output", RUNS / "environment.benchmark.json")
        run("benchmark", "--standard", RUNS / "main" / "standard" / "checkpoints",
            "--modified", RUNS / "main" / "modified" / "checkpoints",
            "--tokenizer", SAVED_DATA / "tokenizer", "--output", OUTPUT, *options)
        print((OUTPUT / "comparison.csv").read_text())
    """),
    ],
    benchmark=True,
)
