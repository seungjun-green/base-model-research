# Base model architecture research

PyTorch implementation of the experiment in [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md):
a standard Qwen3-style decoder and a decoder whose residual connections use the
previous layer's attention/FFN outputs through GELU gates. Both train from scratch
with μP. Colab notebooks launch a shared training package.

The default experiment has 64 proxy trials and two main runs. **No full training
has been performed. GPU correctness, memory, throughput, and μP activation
diagnostics must be checked on the target Blackwell runtime before full sweeps.**

## Models and protocol

| Configuration | Unique parameters | Training targets (20× parameters) |
|---|---:|---:|
| Standard proxy | 13,326,336 | 266,526,720 |
| Modified proxy | 13,362,624 | 267,252,480 |
| Standard main | 157,814,784 | 3,156,295,680 |
| Modified main | 158,395,392 | 3,167,907,840 |

All four models use the 32,000-token [TinyLlama tokenizer](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T).
Regenerate FineWeb-Edu with notebook 01 for this experiment. Earlier Qwen-tokenized shards
and checkpoints belong to the original experiment.
All models have 28 layers, tied embeddings, QK normalization, RoPE, and SwiGLU.
Attention projections use explicit head dimensions; query heads × head dimension
does not have to equal hidden width. The modified layer implements the equations
in the outline exactly, including the first-layer fallback and differentiable
previous-output branches.

Shared defaults: seed 42, context 2048, global batch 128, Adam betas (0.9, 0.95),
epsilon 1e-8, zero weight decay, global gradient clipping 1.0, BF16 autocast with
FP32 parameters/Adam state. The user selected **1% warmup and final LR 10% of peak**
on 2026-11-12. The scheduler spans each model's own complete budget and preserves
μP parameter-group LR ratios. A masked last update hits the target-token budget
exactly instead of overshooting it.

Implementation choices made explicit in the configs: proxy-width random weights
use standard deviation `0.02 × init_multiplier`; normalization scales start at
one. μP reduces hidden matrix standard deviation and Adam LR with width, while
the shared embedding/readout uses constant initialization variance and LR plus
the readout's `1 / width_multiplier` factor. Attention scale is the selected
multiplier divided by head dimension. No second manual readout division is added.

## Start in Colab

Open notebook 00, select the RTX PRO 6000 runtime, and run all cells. Approve the
Google Drive mount. Notebook 01 prepares the data on CPU. Use the GPU for notebooks
02–06. [Open notebook 00 in Colab](https://colab.research.google.com/github/seungjun-green/base-model-research/blob/main/notebooks/00_setup_and_checks.ipynb) · [Notebook order and run controls](docs/COLAB.md)

To prepare the data on your Mac, open [local notebook 01](notebooks/local/01_prepare_data.ipynb)
and select **Run All**. Drag the completed `~/Downloads/fineweb_edu_tinyllama` folder into
**My Drive → bmr → data** and continue with notebook 02 once the upload finishes.

To time one standard 158M run using your original `My Drive/bmr/data/fineweb_edu` upload, open the
[optional timing notebook in Colab](https://colab.research.google.com/github/seungjun-green/base-model-research/blob/main/notebooks/time_standard_main.ipynb).
It converts the Qwen tokens to TinyLlama tokens in Colab as training reads each group of documents.
It trains with one randomly selected hyperparameter set and shows a live ETA for its full token budget.

To test the three alternative gates on the settings that caused the original gated proxy to diverge,
open the [gate comparison notebook in Colab](https://colab.research.google.com/github/seungjun-green/base-model-research/blob/main/notebooks/compare_gate_variants.ipynb).
It runs trial 000 once per architecture and saves the failure step or final validation loss to Drive.

To run trial 000 on the modified proxy with a direct residual and a bounded gate, open the
[residual + tanh proxy notebook in Colab](https://colab.research.google.com/github/seungjun-green/base-model-research/blob/main/notebooks/trial_000_residual_tanh_proxy.ipynb).
It saves pre-clipping gradient RMS for all 112 gate matrices at every optimizer update and plots the 20 held-out validation losses.

The [previous-attention ReLU notebook in Colab](https://colab.research.google.com/github/seungjun-green/base-model-research/blob/main/notebooks/trial_000_previous_attention_relu_proxy.ipynb)
trains trial 000 once with `ReLU(x + current_attention + previous_attention)`. Its FFN width of 432
and 16 query / 8 KV heads match the standard 13,326,336-parameter proxy.

To let each layer learn how much previous attention to add, run the
[trial 000 learned-alpha notebook in Colab](https://colab.research.google.com/github/seungjun-green/base-model-research/blob/main/notebooks/trial_000_previous_attention_relu_with_alpha_proxy.ipynb).
It saves the 27 layer weights alongside each validation loss.

All Colab notebooks use the selected runtime's Python and PyTorch. Setup
installs the project and its dependencies into that runtime. Notebook 06 also
installs the evaluation harness. The Git commit is saved once per experiment.

## Local installation and first checks

Use Python 3.11+ and a dedicated environment. On the GPU machine install a
CUDA-enabled PyTorch build that supports the installed Blackwell GPU and driver.
`requirements.txt` pins direct dependencies; it is not a transitive lockfile.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e '.[data,eval,dev]'
bmr check --device cpu
python -m pytest -q
```

## Shared data and search

```bash
bmr prepare-data --output data/fineweb_edu_tinyllama
bmr sweep --model configs/models/standard_proxy.yaml --data data/fineweb_edu_tinyllama --output runs/standard_vs_gated_160m_tinyllama/proxy
bmr sweep --model configs/models/modified_proxy.yaml --data data/fineweb_edu_tinyllama --output runs/standard_vs_gated_160m_tinyllama/proxy
```

`experiments/trials_seed42.json` is the committed list of 32 combinations. Its
sampling RNG is independent of training RNGs. Both families read exactly this
file, but select winners independently by final held-out next-token cross-entropy.
An interrupted proxy trial restarts from its seed. The sweep skips completed and diverged trials.
`--start 0 --end 8` runs a subset. A finalized `best_hparams.json` is written only
after all 32 trials are complete or diverged. NaN/Inf trials are recorded explicitly.

Preparation resolves immutable dataset/tokenizer revisions, streams the
FineWeb-Edu `sample-100BT` configuration, holds out the first 2,048 whole documents,
then writes a deterministic training stream with EOS document separators.
Training and validation documents are separated before tokenization/packing.
All runs consume the same training prefix to their respective budgets. Validation
uses every prepared held-out target token. No training-data shuffle is performed.
The fixed split is reproducible, not a claim that the first documents represent
all of FineWeb-Edu; inspect it before committing to the full experiment.

The training corpus occupies approximately 12.67 GB in uint32, plus validation,
tokenizer files, caches, and checkpoints. Use the runtime's SSD for active reads.
Rerun notebook 01 after an interruption. It reuses a completed corpus, retries
tokenization from the beginning if necessary, and resumes a partial Drive copy.

## Main training

```bash
bmr train --model configs/models/standard_main.yaml \
  --hyperparameters runs/standard_vs_gated_160m_tinyllama/proxy/standard/best_hparams.json \
  --data data/fineweb_edu_tinyllama --output runs/standard_vs_gated_160m_tinyllama/main/standard
bmr train --model configs/models/modified_main.yaml \
  --hyperparameters runs/standard_vs_gated_160m_tinyllama/proxy/modified/best_hparams.json \
  --data data/fineweb_edu_tinyllama --output runs/standard_vs_gated_160m_tinyllama/main/modified
```

Use `--resume` to continue from the latest saved validation. Before the first save,
training restarts from the seed. `--stop-after-updates 10` runs up to ten more updates. Main runs validate
their winning family's configuration, data fingerprint, and shared training
settings. Existing runs refuse incompatible configuration, package, or commit
changes. Do not edit code in the runtime checkout during experiments.

`--microbatch-size` must divide 128. Profile first; the default is 4. Activation
checkpointing is available with `--activation-checkpointing`. The vocabulary
projection/CE is computed in checkpointed chunks to bound retained logit memory.
This recomputation adds overhead beyond the outline's coarse 6PT estimate.

## Persistence and outputs

Each full run makes one pass through its assigned 20P training targets. It evaluates the
entire held-out split 20 times, at the first optimizer boundary at or after each
5% of its updates, including the final update. Tiny runs with fewer than 20 updates
validate once per update.

Proxy trials save run settings, metrics, and results only. Main runs keep a separate
`checkpoints/validation-<step>.pt` after each validation. All 20 contain the
updated model weights, Adam optimizer state, schedule information, RNG state and
training-token position. `checkpoints/latest.json` identifies the latest completed
save by step and SHA-256 checksum. The final save includes the last optimizer update.

On hosted Colab, approve the Drive mount in **Experiment storage**. Checkpoints and
results go to `My Drive/bmr/runs/standard_vs_gated_160m_tinyllama/`. Notebook 01 saves the shared
corpus under `My Drive/bmr/data/` for use across runtimes. The proxy smoke run
trains trial 000 on its full 20-tokens-per-parameter budget. The full search
reuses the completed trial. Both proxy notebooks copy the required training and validation
shards (about 1.3 GB total) to the runtime SSD before training. Set `MICROBATCH_SIZE` near the top
of each proxy notebook. See [storage guidance](docs/COLAB.md) for
the notebook outputs and the optional self-hosted backup configuration.

## Evaluation

```bash
bmr benchmark --standard runs/standard_vs_gated_160m_tinyllama/main/standard/checkpoints \
  --modified runs/standard_vs_gated_160m_tinyllama/main/modified/checkpoints --tokenizer data/fineweb_edu_tinyllama/tokenizer \
  --output runs/standard_vs_gated_160m_tinyllama/benchmarks
```

After both main runs finish, this evaluates each run's final checkpoint and the three pretrained
references from the outline. The initial explicit evaluation config uses zero
shots for all tasks and the `lambada_openai` variant. Those are configurable
implementation defaults; choose the final protocol before the comparison and
apply it to every model. Each reference keeps its own tokenizer. All runs use
base completion formatting, a 2048 context cap, and evaluation seed 42.

Per-model/task JSON preserves raw harness metrics and provenance; `comparison.csv`
contains metric rows. Completed matching tasks are skipped. A changed checkpoint
or protocol requires a new output directory. `--limit 10` creates labelled debug
results; use a separate directory for the complete evaluation. The custom adapter
preserves our multipliers and μP output computation. It scores continuations with
shifted causal indexing and supports rolling likelihood and basic greedy generation.

## Repository map

- `configs/`: four architectures, shared training, search, and evaluation settings.
- `src/base_model_research/models.py`: model equations and memory-bounded loss.
- `parameterization.py`: μP base shapes, initialization, and optimizer.
- `data.py`: token preparation, manifests, shards, and exact data cursors.
- `training.py`, `storage.py`: training, validation, checkpointing, and Drive synchronization.
- `search.py`, `evaluation.py`, `diagnostics.py`: sweeps, harness adapter, and runtime checks.
- `notebooks/` contains the seven experiment launchers, the optional timing and gate comparison notebooks, and local data preparation in `local/`.
  Regenerate them with `python scripts/build_notebooks.py`.
- `tests/`: parameter counts, equations, causality, gradients, μP, token accounting,
  resumability, likelihood indexing, and trial reproducibility.

Architecture references: [Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py),
[official μP library](https://github.com/microsoft/mup), and
[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

