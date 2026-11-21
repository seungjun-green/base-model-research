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


SETTINGS = code("""
    from pathlib import Path
    import os

    # The checkout, Python environment and active training data use the runtime SSD.
    default_workspace = "/content/bmr" if Path("/content").is_dir() else "~/bmr-workspace"
    WORKSPACE = Path(os.environ.get("BMR_WORKSPACE", default_workspace)).expanduser().resolve()
    EXPERIMENT = "experiment_003"
    DATA = WORKSPACE / "data" / "fineweb_edu"
    MICROBATCH_SIZE = 4  # profile before launching full runs; must divide 128
""")

MOUNT_DRIVE = code("""
    #@title Google Drive storage
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

    def training_data():
        if SAVED_DATA != DATA:
            run("copy-data", "--source", SAVED_DATA, "--destination", DATA)
        return DATA

    def run_training(*arguments):
        command = [str(PYTHON), "-u", "-m", "base_model_research.cli", *map(str, arguments), "--progress-events"]
        run_with_progress(command, cwd=REPO)
"""


def notebook(name, title, cells, *, controls=None, training=False, benchmark=False):
    setup = (ROOT / "scripts" / "colab_setup.py").read_text()
    bootstrap = (
        "#@title Set up experiment runtime\n"
        + setup
        + f"\nREPO, PYTHON = setup_colab(WORKSPACE, RUNS, benchmark={benchmark})\n"
        + textwrap.dedent(BOOTSTRAP)
    )
    report = "environment.benchmark.json" if benchmark else "environment.json"
    bootstrap += f'\nrun("environment", "--output", RUNS / "{report}")\n'
    if training:
        bootstrap += textwrap.dedent(TRAINING_LAUNCHER)
    content = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": name},
        },
        "cells": [
            markdown(
                f"# {title}\n\nSelect your GPU in Colab, run the setup cells in order, "
                "and approve the Google Drive mount when prompted."
            ),
            *([controls] if controls is not None else []),
            SETTINGS,
            MOUNT_DRIVE,
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
            "Profile each shape separately. If needed reduce MICROBATCH_SIZE or add "
            "`--activation-checkpointing`. Profiling performs real optimizer updates on synthetic data."
        ),
        code("""
        MODEL_TO_PROFILE = "standard_proxy"  # repeat for modified_proxy, standard_main, modified_main
        run("profile", "--model", f"configs/models/{MODEL_TO_PROFILE}.yaml",
            "--microbatch-size", MICROBATCH_SIZE, "--output", RUNS / f"{MODEL_TO_PROFILE}_profile.json")
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
            "in decimal units, plus validation, tokenizer, caches, and checkpoints. An interrupted "
            "`.preparing` directory is incomplete; choose a fresh path or explicitly remove it before retrying."
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
)

for family, number in (("standard", "02"), ("modified", "03")):
    notebook(
        f"{number}_{family}_proxy_search.ipynb",
        f"{family.title()} proxy: 32 shared trials",
        [
            code(f"""
            import json
            FAMILY_RUNS = RUNS / "proxy" / "{family}"

            def run_proxy(data, *options):
                run_training("sweep", "--model", "configs/models/{family}_proxy.yaml", "--data", data,
                    "--output", RUNS / "proxy", "--microbatch-size", MICROBATCH_SIZE, *options)
        """),
            code("""
            #@title Smoke run · one proxy model, up to 10 optimizer updates
            if RUN_SMOKE:
                run_proxy(SAVED_DATA, "--start", 0, "--end", 1, "--stop-after-updates", 10)
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
                    print(winner.read_text())
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

for family, number in (("standard", "04"), ("modified", "05")):
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
            PROXY_EXPERIMENT = EXPERIMENT  # Set to "experiment_002" to reuse that experiment's winner.
            WINNER = RUNS.parent / PROXY_EXPERIMENT / "proxy" / "{family}" / "best_hparams.json"
            if not WINNER.exists():
                raise FileNotFoundError("Complete this family's proxy search first")
            options = ["--resume"] if (OUTPUT / "run.json").exists() else []
            run_training("train", "--model", "configs/models/{family}_main.yaml", "--hyperparameters", WINNER,
                "--data", training_data(), "--output", OUTPUT, "--microbatch-size", MICROBATCH_SIZE, *options)
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
        # Set LIMIT=10 for a clearly labelled debug evaluation, then use a NEW output directory
        # with LIMIT=None for the actual comparison.
        LIMIT = None
        OUTPUT = RUNS / ("benchmarks_debug" if LIMIT is not None else "benchmarks")
        options = ["--limit", LIMIT] if LIMIT is not None else []
        run("benchmark", "--standard", RUNS / "main" / "standard" / "checkpoints",
            "--modified", RUNS / "main" / "modified" / "checkpoints",
            "--tokenizer", SAVED_DATA / "tokenizer", "--output", OUTPUT, *options)
        print((OUTPUT / "comparison.csv").read_text())
    """),
    ],
    benchmark=True,
)
