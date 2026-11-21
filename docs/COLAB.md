# Colab and RTX PRO 6000 workflow

## Runtime

In hosted Colab, select **CPU** for notebook 01 and **RTX PRO 6000** for the
other six notebooks. Run **Experiment storage** in each notebook and approve the Google Drive mount.
The cell prints the experiment folder shared by all notebooks.

Allow about 48 GB of local disk space for prepared tokens, plus room for caches
and the Python environment. FP32 model weights, gradients, and two Adam moments
occupy roughly 9.6 GB for a 0.6B model before activations, logits, and overhead.
Each main-model checkpoint uses about 7.2 GB on Drive for the FP32 weights and
two Adam moments. Budget about 144 GB per main model, or 288 GB for both, plus the corpus.

Notebook 00 reports the GPU, runs CPU correctness checks, records activation
magnitudes across widths, and profiles all four models in one cell. If a profile
runs out of memory, reduce `MICROBATCH_SIZE` in that cell and rerun it. New training
runs read their model's memory settings from its saved profile. An existing run
keeps the settings it started with. The global batch size stays at 128.

## GitHub and environment

Open notebook 00 in Colab, mount Drive, and run the setup cell. All seven notebooks
use [seungjun-green/base-model-research](https://github.com/seungjun-green/base-model-research).
The first setup in a new experiment saves the current commit of the repository's
default branch in `source.json` on Drive. Later notebooks and fresh runtimes reuse
that commit automatically. Use a new `EXPERIMENT` name to pick up a newer revision.

Notebooks 00–05 share an isolated data and training environment. Notebook 06
installs the evaluation harness in its own environment. On a fresh runtime,
setup automatically installs the versions saved in `requirements.freeze.txt`
for training or `requirements.benchmark.freeze.txt` for evaluation. Existing
experiments with the older combined training snapshot restore that package set.

The default workspace is `/content/bmr` where `/content` exists, otherwise
`~/bmr-workspace`. On a local/remote runtime choose a persistent SSD directory.
The hosted runtime can be replaced between notebooks. The checkout, virtual
environment and local dataset cache are disposable. Mounting Drive again
reopens the saved experiment. All notebooks launch the same Python package.

## Notebook order

1. `00_setup_and_checks.ipynb`: inspect environment and diagnostics; profile the four shapes.
2. `01_prepare_data.ipynb`: prepare the common dataset once.
3. `02_standard_proxy_search.ipynb`: standard family's 32 trials.
4. `03_modified_proxy_search.ipynb`: modified family's same 32 combinations.
5. `04_train_standard_main.ipynb`: transfer the standard winning settings.
6. `05_train_modified_main.ipynb`: transfer the modified winning settings.
7. `06_benchmarks.ipynb`: run the five-model comparison.

In notebooks 02 and 03, choose the checkboxes in **Run controls**, then run all cells.
The default enables only `RUN_SMOKE`, which tests trial 000 for up to ten optimizer
updates. Smoke reads the needed token shards directly from the corpus saved by
notebook 01. For the full search, clear `RUN_SMOKE` and enable `RUN_FULL`. The full search
starts trial 000 again at step zero. Enabling both runs them in order.

Notebooks 02–05 display a tqdm bar for optimizer updates with training loss, tokens
per second, and estimated time remaining. Smoke runs count the updates requested
for that invocation. A resumed main run starts the bar at its saved step.
The full proxy searches also show how many of the 32 trials have finished.

Use one training process per GPU. Every full trial or main run validates on the same
held-out split at 5%, 10%, …, 100% of its one-pass budget. A ten-step production
smoke run ends before the first scheduled validation. Reopening a sweep skips
terminal trials. An unfinished proxy trial restarts from its seed. Rerunning
notebook 04 or 05 restores the latest main checkpoint and continues with the next
optimizer update. Work since that save is repeated. Before the first validation,
the main run restarts from its seed.

## Google Drive

Hosted notebooks use `drive.mount("/content/drive")`. Leave **USE_GOOGLE_DRIVE**
checked. Notebook 01 prepares the corpus on the runtime SSD, then copies the
tokenizer, manifest, and verified shards to `My Drive/bmr/data/fineweb_edu/`
(about 48 GB). On a fresh VM, the first full proxy or main run copies the corpus
back to the SSD. Further full runs reuse that copy. Rerun the cell
after an interrupted transfer to reuse the shards that passed verification.
If tokenization itself was interrupted, rerunning starts it again in a fresh temporary directory.
Notebook 06 reads just the saved tokenizer.

Run outputs go directly to `My Drive/bmr/runs/standard_vs_gated/`. To continue
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

For a self-hosted runtime, clear **USE_GOOGLE_DRIVE** and choose a persistent
`BMR_WORKSPACE`. Command-line users with an authenticated `rclone` remote
can use `bmr train --remote`, `bmr sweep --remote`, or `bmr sync` for backups.

