# Standard vs. gated model progress

Updated November 24, 2026. [Experiment design](PROJECT_OUTLINE.md)

## Experiment checklist

- [x] Prepare FineWeb-Edu with the TinyLlama 32k tokenizer and upload it to Drive. The shared corpus has 3,167,907,840 training targets and 2,609,879 validation tokens. [Notebook 01](notebooks/local/01_prepare_data.ipynb)
- [x] Complete the standard 13.3M proxy sweep. All **32 of 32** trials finished. **`trial_004`** had the lowest final validation loss, **4.4548**. [Notebook 02](notebooks/02_standard_proxy_search.ipynb)
- [ ] Complete the gated 13.3M proxy sweep on the same 32 hyperparameter combinations and record its winner and final validation loss. [Notebook 03](notebooks/03_modified_proxy_search.ipynb)
- [ ] Train the 157.8M standard main model using μP transfer from `trial_004`. [Notebook 04](notebooks/04_train_standard_main.ipynb)
- [ ] Train the 158.4M gated main model using μP transfer from its own proxy winner. [Notebook 05](notebooks/05_train_modified_main.ipynb)
- [ ] Evaluate both final checkpoints and the three pretrained references on all seven benchmarks. [Notebook 06](notebooks/06_benchmarks.ipynb)
- [ ] Compare the main models’ final validation losses and benchmark scores to report where gating helped or hurt.

## Standard proxy results

- [All 32 trial results](results/proxy/standard/results.csv)
- [Winning hyperparameters for `trial_004`](results/proxy/standard/best_hparams.json)

Notebook 04 reads the winner from the Google Drive copy at `My Drive/bmr/runs/standard_vs_gated_160m_tinyllama/proxy/standard/best_hparams.json`.
