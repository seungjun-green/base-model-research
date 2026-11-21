"""Generate small, reviewable Colab launchers; training lives in the package."""

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


STORAGE = code("""
    #@title Experiment storage
    from pathlib import Path
    import os

    default_workspace = "/content/bmr" if Path("/content").is_dir() else "~/bmr-workspace"
    WORKSPACE = Path(os.environ.get("BMR_WORKSPACE", default_workspace)).expanduser().resolve()
    EXPERIMENT = "standard_vs_gated"
    DATA = WORKSPACE / "data" / "fineweb_edu"
    USE_GOOGLE_DRIVE = True  #@param {type:"boolean"}
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
    SAVED_DATA = storage / "data" / "fineweb_edu"
    print("Experiment outputs:", RUNS)
""")

BOOTSTRAP = """
    def run(*arguments):
        subprocess.run([str(PYTHON), "-m", "base_model_research.cli", *map(str, arguments)],
                       cwd=REPO, check=True)
"""


TRAINING_LAUNCHER = """
    # tqdm renders in the notebook kernel while training uses the isolated environment.
    from importlib.util import find_spec
    if find_spec("tqdm") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "tqdm>=4.66,<5"], check=True)
    sys.path.insert(0, str(REPO / "src"))
    from base_model_research.progress import run_with_progress

    def training_data(*, local_copy=True):
        if not (SAVED_DATA / "manifest.json").is_file():
            raise FileNotFoundError(f"Run notebook 01 to prepare the shared dataset at {SAVED_DATA}.")
        if local_copy and SAVED_DATA != DATA:
            run("copy-data", "--source", SAVED_DATA, "--destination", DATA)
        return DATA if local_copy else SAVED_DATA

    def run_training(command_name, model_name, run_record, *arguments):
        # Existing runs keep their settings. New runs use the profile from 00 or the training config.
        settings_file = run_record if run_record.is_file() else RUNS / f"{model_name}_profile.json"
        memory_options = []
        if settings_file.is_file():
            settings = json.loads(settings_file.read_text())["training"]
            memory_options = ["--microbatch-size", str(settings["microbatch_size"])]
            if settings["activation_checkpointing"]:
                memory_options.append("--activation-checkpointing")
        command = [str(PYTHON), "-u", "-m", "base_model_research.cli", command_name,
                   "--model", f"configs/models/{model_name}.yaml", *map(str, arguments),
                   *memory_options, "--progress-events"]
        run_with_progress(command, cwd=REPO)
"""


def notebook(name, title, cells, *, controls=None, training=False, benchmark=False, gpu=True):
    setup = (ROOT / "scripts" / "colab_setup.py").read_text()
    bootstrap = (
        "#@title Set up experiment runtime\n"
        + setup
        + f"\nREPO, PYTHON = setup_colab(WORKSPACE, RUNS, benchmark={benchmark})\n"
        + textwrap.dedent(BOOTSTRAP)
    )
    if training:
        bootstrap += textwrap.dedent(TRAINING_LAUNCHER)
    runtime = "Select the RTX PRO 6000 GPU" if gpu else "Select a CPU runtime"
    content = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            **({"accelerator": "GPU"} if gpu else {}),
            "colab": {"name": name},
        },
        "cells": [
            markdown(
                f"# {title}\n\n{runtime} in Colab, run the cells in order, "
                "and approve the Google Drive mount when prompted."
            ),
            *([controls] if controls is not None else []),
            STORAGE,
            code(bootstrap),
            *cells,
        ],
    }
    for index, cell in enumerate(content["cells"]):
        cell["id"] = f"cell-{index:03d}"
    path = ROOT / "notebooks" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


notebook(
    "00_setup_and_checks.ipynb",
    "Setup, correctness checks, and GPU profiling",
    [
        code("""
        subprocess.run(["nvidia-smi"], check=True)
        run("environment", "--output", RUNS / "environment.json")
        run("check", "--device", "cpu", "--output", RUNS / "checks.json")
        run("trials")
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
            "11,957,698,560 training targets plus one lookahead token. uint32 tokens occupy about 48 GB "
            "in decimal units, plus validation, tokenizer and caches. If preparation stops, rerun this cell. "
            "Tokenization starts over if needed. A partial Drive copy resumes from verified shards."
        ),
        code("""
        import shutil
        print("Free disk GB:", shutil.disk_usage(WORKSPACE).free / 1e9)
        if not (SAVED_DATA / "manifest.json").is_file():
            if not (DATA / "manifest.json").is_file():
                run("prepare-data", "--output", DATA)
            if SAVED_DATA != DATA:
                run("copy-data", "--source", DATA, "--destination", SAVED_DATA)
        print("Prepared corpus:", SAVED_DATA)
    """),
    ],
    gpu=False,
)

for family, number in (("standard", "02"), ("modified", "03")):
    notebook(
        f"{number}_{family}_proxy_search.ipynb",
        f"{family.title()} proxy: 32 shared trials",
        [
            code(f"""
            FAMILY_RUNS = RUNS / "proxy" / "{family}"

            def run_proxy(data, *options):
                run_training("sweep", "{family}_proxy", FAMILY_RUNS / "protocol.json",
                    "--data", data, "--output", RUNS / "proxy", *options)
        """),
            code("""
            #@title Smoke run · one proxy model, up to 10 optimizer updates
            if RUN_SMOKE:
                run_proxy(training_data(local_copy=False), "--start", 0, "--end", 1, "--stop-after-updates", 10)
                result = json.loads((FAMILY_RUNS / "trial_000" / "result.json").read_text())
                print(f"Trial 000 is {result['status']} at step {result['step']}.")
            else:
                print("Smoke run disabled.")
        """),
            code("""
            #@title Full run · all 32 trials
            if RUN_FULL:
                run_proxy(training_data())
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
        training=True,
        controls=code("""
            #@title Run controls
            RUN_SMOKE = True   #@param {type:"boolean"}
            RUN_FULL = False   #@param {type:"boolean"}
            # Smoke tests 10 updates of trial 000. Proxy trials save metrics only.
            # Full starts unfinished trials from their seed and skips completed trials.
            # Enabling both runs smoke first, then full. Disabling both skips training.
        """),
    )

for family, number, proxy_number in (("standard", "04", "02"), ("modified", "05", "03")):
    notebook(
        f"{number}_train_{family}_main.ipynb",
        f"Train the {family} main model from scratch",
        [
            markdown(
                "Trains once through this family's token budget and validates at 5% intervals. "
                "Each of the 20 validations saves the model weights and optimizer state to Drive. "
                "The 20th save includes the final optimizer update. Rerun this notebook to continue "
                "from the latest completed save. Allow about 144 GB on Drive per main model."
            ),
            code(f'''
            OUTPUT = RUNS / "main" / "{family}"
            WINNER = RUNS / "proxy" / "{family}" / "best_hparams.json"
            if not WINNER.is_file():
                raise FileNotFoundError("Enable RUN_FULL in notebook {proxy_number} "
                                        f"and finish the {family} proxy search to create {{WINNER}}.")
            options = ["--resume"] if (OUTPUT / "run.json").exists() else []
            run_training("train", "{family}_main", OUTPUT / "run.json", "--hyperparameters", WINNER,
                "--data", training_data(), "--output", OUTPUT, *options)
        '''),
        ],
        training=True,
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
