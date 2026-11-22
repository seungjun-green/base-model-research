# Colab and RTX PRO 6000 workflow

## Runtime

In hosted Colab, select **CPU** for notebook 01 and **RTX PRO 6000** for the
other six notebooks. Run **Experiment storage** in each notebook and approve the Google Drive mount.
The cell prints the experiment folder shared by all notebooks.

Allow about 12.7 GB of runtime disk for corpus preparation or full main training,
plus room for download caches. FP32 model weights, gradients, and two Adam moments
occupy roughly 2.5 GB for a main model before activations, logits, and overhead.
Each main-model checkpoint uses about 1.9 GB on Drive for the FP32 weights and
two Adam moments. Budget about 38 GB per main model, or 76 GB for both, plus the corpus.

Notebook 00 reports the GPU, runs CPU correctness checks, records activation
magnitudes across widths, and profiles all four models in one cell. If a profile
runs out of memory, reduce `MICROBATCH_SIZE` in that cell and rerun it. New training
runs use the profile's activation checkpointing setting. Main runs also use the profiled
microbatch size. Each proxy notebook has its own microbatch control. The global batch size stays at 128.

## GitHub and environment

Open notebook 00 in Colab, mount Drive, and run the setup cell. All seven notebooks
use [seungjun-green/base-model-research](https://github.com/seungjun-green/base-model-research).
The first setup in a new experiment saves the current commit of the repository's
default branch in `source.json` on Drive. Later notebooks and fresh runtimes reuse
that commit automatically. Use a new `EXPERIMENT` name to pick up a newer revision.

Setup uses Colab's current Python and keeps its installed PyTorch build. It
installs the project in that runtime and reuses compatible dependencies. Notebook
06 adds the evaluation harness. Run setup again when you connect to a fresh VM.
Each training run records its package versions in `run.json`. To resume an
existing run, use the recorded PyTorch, NumPy, and μP versions. Older
virtual-environment snapshots remain on Drive for reference.

The default workspace is `/content/bmr` where `/content` exists, otherwise
`~/bmr-workspace`. On a local/remote runtime choose a persistent SSD directory.
The hosted runtime can be replaced between notebooks. The checkout, installed
packages and local dataset cache are disposable. Mounting Drive again
reopens the saved experiment. All notebooks launch the same Python package.

## Notebook order

1. `00_setup_and_checks.ipynb`: inspect environment and diagnostics; profile the four shapes.
2. `01_prepare_data.ipynb`: prepare the common dataset once.
3. `02_standard_proxy_search.ipynb`: standard family's 32 trials.
4. `03_modified_proxy_search.ipynb`: modified family's same 32 combinations.
5. `04_train_standard_main.ipynb`: transfer the standard winning settings.
6. `05_train_modified_main.ipynb`: transfer the modified winning settings.
7. `06_benchmarks.ipynb`: run the five-model comparison.

In notebooks 02 and 03, the first code cell mounts Google Drive. In the next cell,
set `MICROBATCH_SIZE`, `RUN_SMOKE`, and `RUN_FULL`. The microbatch defaults to 4 sequences
and must divide 128. For an existing sweep, use its recorded microbatch size.
To run a fresh sweep with a different size, choose a new `EXPERIMENT` in the storage cell
and use that name in subsequent notebooks for that experiment.
The default enables only `RUN_SMOKE`, which trains trial 000 on 20 tokens per
parameter (about 267 million tokens) and validates 20 times. For the full search,
set `RUN_SMOKE = False` and `RUN_FULL = True`. A completed trial 000 counts toward
the 32, leaving 31 trials. Enabling both runs trial 000 first and then the remaining trials.

Notebooks 02–05 display a tqdm bar for optimizer updates with training loss, tokens
per second, and estimated time remaining. A resumed main run starts the bar at its saved step.
The full proxy searches also show how many of the 32 trials have finished.

Use one training process per GPU. Every full trial or main run validates on the same
held-out split at 5%, 10%, …, 100% of its one-pass budget. Reopening a sweep skips
terminal trials. An unfinished proxy trial restarts from its seed. Rerunning
notebook 04 or 05 restores the latest main checkpoint and continues with the next
optimizer update. Work since that save is repeated. Before the first validation,
the main run restarts from its seed.

## Google Drive

Hosted notebooks use `drive.mount("/content/drive")`. Keep `USE_GOOGLE_DRIVE = True`
in the **Experiment storage** cell. Notebook 01 prepares the corpus on the runtime SSD, then copies the
tokenizer, manifest, and verified shards to `My Drive/bmr/data/fineweb_edu_tinyllama/`
(about 12.7 GB). Before either proxy mode starts, the notebook copies the training shards
covering its token budget and the validation split into `/content/bmr/data/proxy_shards/`
(about 1.3 GB combined). Run **Copy proxy data from Google Drive to /content** after setup.
Its tqdm bar shows how many shards are ready. All trials on that runtime read
those local files. A fresh runtime
needs a fresh copy. If the full corpus is already prepared on the runtime, training uses it directly.
Main training copies and verifies shards on first use. A resumed main run fetches the shards
it needs from its saved token position. An existing, larger TinyLlama corpus works as well:
each run reads only its assigned training prefix.
If tokenization itself was interrupted, rerunning starts it again in a fresh temporary directory.
Notebook 06 reads just the saved tokenizer.

Run outputs go directly to `My Drive/bmr/runs/standard_vs_gated_160m_tinyllama/`. To continue
an existing experiment, use its original folder name for `EXPERIMENT` in all notebooks.

| Notebook | Output in the experiment folder |
| --- | --- |
| 00 | Environment, diagnostic reports and GPU profiles |
| 02–03 | Each family's proxy metrics, trial results and winning `best_hparams.json` |
| 04–05 | All 20 validation checkpoints per main model, the latest checkpoint pointer, metrics and final result |
| 06 | Per-model/task results and `comparison.csv` in `benchmarks/` or `benchmarks_debug/` |

Each checkpoint contains the updated weights, optimizer state and position in the
training stream. The 20th includes the last gradient update. Notebook 06 loads
that final checkpoint from each completed run.

Notebooks 04 and 05 find the corresponding proxy winner in the current experiment.
To reuse one from a different experiment, set `WINNER` to its `best_hparams.json` path. They use the
shared corpus and verify each winner's training settings and dataset fingerprint.
Earlier weights-only files can still be evaluated with their original protocol.

For a self-hosted runtime with local storage, set `USE_GOOGLE_DRIVE = False` and choose a persistent
`BMR_WORKSPACE`. Command-line users with an authenticated `rclone` remote
can use `bmr train --remote`, `bmr sweep --remote`, or `bmr sync` for backups.

