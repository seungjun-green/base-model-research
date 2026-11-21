# Local implementation verification

Verified on 2026-11-12 with Python 3.12.7, PyTorch 2.13.0 (CPU/macOS),
`mup` 1.0.0, NumPy 2.1.2, Transformers 5.15.0, Datasets 4.5.0, and
LM Evaluation Harness 0.4.13.

The automated suite checks:

- Exact unique parameter counts and derived 20× token budgets for all four full configurations,
  using meta-device construction without allocating the full models.
- Causality, tied weights, both gate equations, first-layer fallback, and gradients
  through previous attention/FFN outputs.
- Chunked/recomputed loss and parameter gradients against a full-logit reference,
  and activation-checkpointing gradients against the ordinary forward pass.
- μP hidden/gate learning rates, shared embedding learning rate, and width-scaled readout.
- Cross-shard reads, shifted targets, final-token masking, and exact budget completion.
- Bitwise CPU agreement between uninterrupted and paused/resumed synthetic training,
  including final validation loss and RNG state, for both model families.
- Refusal to resume a changed training protocol; cosine warmup/endpoints and group ratios.
- Shared trial reproducibility, bounds, manifest integrity, and separate sweep winners.
- Main-training CLI transfer of each family's selected settings to width 4, with
  the selection provenance included in the checkpoint.
- Offline data preparation with mocked downloads, immutable revisions, and whole-document splitting.
- Actual LM Evaluation Harness execution against an in-memory synthetic task.
- Continuation likelihood indexing and long-continuation accounting.
- All seven notebooks' Python cells and deterministic notebook regeneration.

Separately, a tiny standard decoder was compared against the Hugging Face Qwen3
implementation with identical weights, RoPE configuration, and attention scaling.
The maximum absolute logit difference was **5.96e-8**. For this parity check, our
attention multiplier was set to `sqrt(head_dim)` to match Qwen3's default
`1/sqrt(head_dim)` scaling. This changes only the verification setup; the actual
experiment retains the outline's selected multiplier divided by head dimension.
The verification process disabled this host's unrelated, incompatible optional
vision/audio imports; no installed packages or model equations were modified.

The package also built and installed into a temporary directory, and its `bmr`
entry point executed successfully. Ruff lint and format checks pass.

On 2026-11-20 the review suite passed 34 CPU tests. New cases cover interrupted
initial saves, checkpoint paths after reopening a run, sweep configuration and
result mismatches, and the checksum recorded when a checkpoint pointer changes
during evaluation. All seven notebooks passed nbformat validation. An rclone
transfer between local directories, interrupted during verification and retried,
published the checkpoint pointer and result files only after verification succeeded.

These checks do not establish CUDA/BF16 correctness, Blackwell kernel support,
GPU memory use, training throughput, full-dataset preparation, Drive credentials/
transfer behavior, μP transfer quality for the gated architecture, or benchmark
scores. Those require the target runtime and actual experiment artifacts. Run
notebook 00 and inspect its coordinate diagnostics before launching the sweeps.
