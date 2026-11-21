# Base Model Architecture Research Project

Updated: 2026-11-12 (Asia/Seoul)

This document captures the agreed experiment design. Codebase implementation began on 2026-11-12; full experiment training has not begun.

## 1. Research objective

Compare two model families trained from scratch:

1. **Standard:** a Qwen3-style decoder with its ordinary residual connections, trained using μP.
2. **Modified:** a Qwen3-style decoder with gated connections using the previous layer's attention and FFN outputs, also trained using μP.

Tune each family independently on a small proxy, transfer its winning hyperparameters to its approximately 0.6B main model, and compare benchmark results. Include several existing pretrained base models as references.

## 2. Model configurations

| Setting | Standard proxy | Modified proxy | Standard main | Modified main |
|---|---:|---:|---:|---:|
| Decoder layers | 28 | 28 | 28 | 28 |
| Hidden width (`d_model`) | 256 | 256 | 1024 | 1024 |
| Query heads | 16 | 12 | 16 | 12 |
| KV heads | 8 | 6 | 8 | 6 |
| Head dimension | 32 | 32 | 128 | 128 |
| FFN intermediate dimension | 768 | 560 | 3072 | 2240 |
| Vocabulary size | 151,936 | 151,936 | 151,936 | 151,936 |
| Total unique parameters | 66,437,120 | 66,551,808 | 596,049,920 | 597,884,928 |
| Approximate size | 66.44M | 66.55M | 596.05M | 597.88M |

The standard main model follows the [Qwen3-0.6B-Base architecture configuration](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json), with μP applied to training and parameterization.

The modified family shrinks **both attention and FFN** to offset the gate parameters. The two main models differ in parameter count by approximately **0.31%**. Exact equality is unnecessary.

Retain Qwen3's causal grouped-query attention, pre-norm RMSNorm, QK normalization, RoPE, SwiGLU FFN, and tied input/output embeddings. The new gate linear layers have no biases.

Here, **width means the hidden dimension representing each token**, not total parameter count. Within each family, the proxy-to-main changes are:

- Hidden width: 256 → 1024 (**4×**).
- Head dimension: 32 → 128 (**4×**).
- FFN dimension: 768 → 3072 for standard; 560 → 2240 for modified (**4×**).
- Layer count and head counts stay fixed within each family.

## 3. Modified architecture

### 3.1 Gate inputs: the important distinction

At decoder layer `l`, define:

| Symbol | Meaning |
|---|---|
| `x_l` | Current layer's residual-stream input **before the RMSNorm preceding attention** |
| `a_l` | Current attention block's output **after its output projection, before residual addition** |
| `h_l` | Current layer's residual-stream input **before the RMSNorm preceding the FFN** |
| `f_l` | Current FFN block's output **after its down projection, before residual addition** |

Each gate combines a **current pre-norm input** with a **previous layer's block output**. The previous-layer branch uses the block output, not the previous pre-norm input.

### 3.2 Gate equations and decoder updates

For the attention connection:

$$
g_l^{\mathrm{attn}}
=\operatorname{GELU}(W_{l,1}^{\mathrm{attn}}x_l)
\odot(W_{l,2}^{\mathrm{attn}}a_{l-1})
$$

For the FFN connection:

$$
g_l^{\mathrm{ffn}}
=\operatorname{GELU}(W_{l,1}^{\mathrm{ffn}}h_l)
\odot(W_{l,2}^{\mathrm{ffn}}f_{l-1})
$$

The gated term replaces the ordinary identity residual term at each addition:

$$
a_l=\operatorname{Attn}(\operatorname{RMSNorm}(x_l))
$$

$$
h_l=a_l+g_l^{\mathrm{attn}}
$$

$$
f_l=\operatorname{FFN}(\operatorname{RMSNorm}(h_l))
$$

$$
x_{l+1}=f_l+g_l^{\mathrm{ffn}}
$$

The multiplication between gate branches is elementwise. The gate activation is **GELU**, inspired by GEGLU. The FFN itself retains Qwen3's **SwiGLU** activation structure.

### 3.3 First decoder layer

There is no previous decoder layer. Use the corresponding current pre-norm input on both branches:

- Attention gate: use `x_1` in place of `a_0`.
- FFN gate: use `h_1` in place of `f_0`.

The two branches still have separate weight matrices.

### 3.4 Shapes and added parameters

There are **four independent new matrices per decoder layer**, two for attention and two for FFN. They are independent across layers and have no biases.

| Model | Shape of each new matrix | Added gate parameters across 28 layers |
|---|---|---:|
| Modified proxy | 256 × 256 | 7,340,032 |
| Modified main | 1024 × 1024 | 117,440,512 |

The attention block's final output has shape `(batch, sequence_length, d_model)` after the output projection, even with the modified head counts. At the main model's training context length, that is `(batch, 2048, 1024)`. FFN outputs and gate outputs have the same shape.

## 4. Dataset and shared training settings

| Setting | Agreed choice |
|---|---|
| Dataset | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| Tokenizer for all models we train | Qwen3 tokenizer |
| Training objective | Next-token prediction |
| Context length | **2048 tokens** |
| Global batch size | **128 sequences per optimizer update** |
| Tokens per full optimizer update | **262,144** |
| Random seed | **42 for every run** |
| Input/output embedding weights | Tied; count the shared matrix once |

Use microbatches and gradient accumulation as needed to reach the global batch size. Use a shared held-out validation set for both model families.

Each trial or main run makes one pass through its assigned 20P training-token budget. Evaluate validation loss at 20 evenly spaced optimizer boundaries, including the final update. Proxy trials save metrics only. Each main run saves its model weights and optimizer state after every validation and keeps all 20 checkpoints. Resume training from the latest saved validation. Benchmark the final checkpoint from each completed main run.

## 5. Token budgets: retain the 20× rule

For every proxy trial and every main-model run:

$$
\text{training tokens}=20\times\text{actual total unique model parameters}
$$

Use each model's actual parameter count, including the shared embedding matrix once.

| Model | Training tokens per run | Training runs |
|---|---:|---:|
| Standard proxy | 1,328,742,400 (~1.329B) | 32 |
| Modified proxy | 1,331,036,160 (~1.331B) | 32 |
| Standard main | 11,920,998,400 (~11.921B) | 1 |
| Modified main | 11,957,698,560 (~11.958B) | 1 |

Total planned training runs: **64 proxy trials + 2 main runs = 66 runs**. Use the single fixed seed, 42.

## 6. Hyperparameters to tune

Search the following five hyperparameters:

| Hyperparameter | Meaning | Search range |
|---|---|---:|
| Base peak learning rate | Base Adam learning-rate setting, before μP's parameter-group scaling | 0.0001–0.1 |
| Initialization multiplier | Scales initial random weights | 0.1–10 |
| Embedding multiplier | Multiplies token vectors after embedding lookup | 0.1–10 |
| Output multiplier | Multiplies final next-token scores before softmax | 0.25–4 |
| Attention multiplier | Multiplies attention scores before attention softmax | 0.25–4 |

These are fixed scalar settings chosen through proxy trials, rather than extra scalar parameters learned during a run.

### 6.1 What the embedding and output multipliers do

- **Embedding multiplier:** after looking up a token's vector, multiply every entry by the chosen number before it enters the decoder.
- **Output multiplier:** after computing the vocabulary scores, multiply every score by the chosen number before softmax.

For example, an embedding multiplier of 2 changes `[0.2, -0.4]` into `[0.4, -0.8]`. An output multiplier of 2 changes scores `[1, 2, 3]` into `[2, 4, 6]`.

The two multipliers act on different computed values even though the underlying input/output weight matrix is shared. The shared matrix's initialization and learning rate are controlled by the global initialization multiplier and base learning-rate setting; the input and output uses do not receive separate matrix initializations or learning rates.

These two tunable multipliers are additions to the standard Qwen3 implementation for our μP experiment. Apply them to **both** model families.

### 6.2 How to generate 32 combinations

Use **log-uniform random sampling**, following the μP paper's GPT-3 proxy-search method:

| Hyperparameter | Sampling rule |
|---|---|
| Base peak learning rate | Draw an exponent uniformly from −4 to −1; calculate 10 to that power |
| Initialization multiplier | Draw an exponent uniformly from −1 to 1; calculate 10 to that power |
| Embedding multiplier | Draw an exponent uniformly from −1 to 1; calculate 10 to that power |
| Output multiplier | Draw an exponent uniformly from −1 to 1; calculate 4 to that power |
| Attention multiplier | Draw an exponent uniformly from −1 to 1; calculate 4 to that power |

One combination contains one sampled value for each of the five hyperparameters. Repeat 32 times using **seed 42**. For example, a learning-rate exponent of −2.5 produces approximately 0.00316.

Test the **same 32 sampled combinations on both proxy families**. The standard and modified proxies therefore each receive 32 trials. The number 32 is our agreed search budget.

The original paper's GPT-3 search calls the output/attention knobs **temperatures**, which are reciprocals of the multipliers used here. Reciprocating those log-uniform distributions leaves the same 0.25–4 bounds and sampling distribution. The paper also searched a relative-position embedding multiplier; our agreed five-knob search uses Qwen3's RoPE and excludes that extra knob. See [μP Appendix F.4](https://arxiv.org/html/2203.03466#A6.SS4).

## 7. Proxy evaluation and model selection

Evaluate proxies by **held-out next-token cross-entropy**. Monitor training loss and stability as well.

- Select the lowest-validation-loss combination independently for the standard family.
- Select the lowest-validation-loss combination independently for the modified family.

Proxy benchmark evaluation is unnecessary. The purpose of these models is to select hyperparameters for their corresponding main models.

## 8. Optimizer settings

| Setting | Agreed value |
|---|---|
| Optimizer | Adam with μP scaling, using `mup.MuAdam` |
| Adam betas | (0.9, 0.95) |
| Adam epsilon | 1e-8 |
| Weight decay | 0 |
| Global gradient-norm clipping | 1.0 |

Keep these settings fixed across trials and families. The base peak learning rate is one of the five searched hyperparameters.

## 9. μP transfer from proxy to main model

Transfer each proxy family's winning hyperparameters to its corresponding main model. Initialize fresh main-model weights; the transferred objects are the selected hyperparameter settings.

The width multiplier is **1024 / 256 = 4**, rather than the ratio of total parameter counts.

### 9.1 Hidden weights

Treat the new gate matrices as hidden weights, alongside the attention and FFN projection matrices. As their fan-in grows 4×:

| Quantity | Main-model setting relative to its proxy |
|---|---|
| Hidden-weight initialization variance | Divide by 4 |
| Hidden-weight initialization standard deviation | Divide by 2 |
| Hidden-weight Adam learning rate | Divide by 4 |

Keep the chosen base hyperparameter constants; μP produces the corresponding parameter-group scaling. Input-like and normalization parameters follow their appropriate μP rules. Use the [official μP library](https://github.com/microsoft/mup), including base-shape registration, μP initialization, `MuAdam`, and a shared readout for tied embeddings.

### 9.2 Shared embedding/output matrix: both uses

Qwen3 uses one shared matrix for input embeddings and the output vocabulary projection. Use μP's equivalent **Table 8 formulation**, which supports this weight sharing.

| Item | Proxy: width 256 | Main: width 1024 |
|---|---|---|
| Width multiplier | 1 | 4 |
| Shared matrix initialization variance | Selected proxy value | Same variance |
| Shared matrix Adam learning rate | Selected proxy value | Same learning rate |
| Input computation | Embedding lookup × selected embedding multiplier | Same operation and same selected multiplier |
| Output computation | Raw vocabulary scores × selected output multiplier | Raw vocabulary scores × the same selected output multiplier, then **divide by 4** |

The output division occurs **after** the shared matrix is used to compute scores. It changes the output computation, while the input lookup uses the shared matrix without that division. Both uses contribute gradients to the same matrix.

The additional **1/4 is calculated from the width ratio; it is not a searched hyperparameter**. The separate-output initialization/LR rows in the summary's table are replaced by the shared-weight formulation for this matrix. See [μP Table 8 and Appendix B.1](https://arxiv.org/html/2203.03466).

### 9.3 Attention scaling

Use μP's **1 / head_dimension** attention-score factor with the selected attention multiplier. Head dimension grows from 32 to 128, while the chosen attention multiplier stays fixed.

## 10. Cosine learning-rate schedule

Use cosine learning-rate decay. **Each run's schedule must span that run's complete training duration**, computed from its own 20× parameter token budget and the global batch size.

| Model | Approximate optimizer updates, rounding up |
|---|---:|
| Standard proxy | 5,069 |
| Modified proxy | 5,078 |
| Standard main | 45,475 |
| Modified main | 45,615 |

Recompute schedule duration for each model size. Preserve μP's parameter-group learning-rate ratios when applying the scheduler.

During implementation on 2026-11-12, the user selected **1% warmup** and cosine decay to **10% of the peak learning rate**. Apply these settings to both families and all model sizes. Warmup spans 1% of each run's own total optimizer updates (rounding up).

### Qwen3's published learning rate

The **Qwen3 Technical Report does not disclose an exact numerical pretraining learning rate for Qwen3-0.6B**. Section 3.2 describes selecting learning-rate and batch-size strategies through scaling-law experiments without publishing the numerical LR for this model. Our learning rate therefore comes from the agreed proxy search. See [Qwen3 Section 3.2](https://arxiv.org/html/2505.09388#S3.SS2).

## 11. Main-model benchmark evaluation

Evaluate both trained main models on the agreed seven benchmarks:

1. HellaSwag
2. PIQA
3. ARC-Easy
4. WinoGrande
5. OpenBookQA
6. MMLU
7. LAMBADA

Use the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with consistent benchmark settings across compared models. ARC-Challenge was removed from the agreed benchmark list.

Compare these five models:

| Model | Role |
|---|---|
| Our standard main model (~596.05M) | Trained baseline |
| Our modified main model (~597.88M) | Trained architecture experiment |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Pretrained base reference |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | Pretrained base reference |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | Pretrained base reference |

Use the base checkpoints listed above. Pythia-410M was mentioned as an optional additional reference, but is not part of the main five-model comparison.

## 12. Updated training-compute estimate

Use the rough estimate `training FLOPs ≈ 6 × parameters × training tokens`.

| Group | Runs | Approximate total training FLOPs |
|---|---:|---:|
| Standard proxy | 32 | 16.95 EFLOPs |
| Modified proxy | 32 | 17.01 EFLOPs |
| Standard main | 1 | 42.63 EFLOPs |
| Modified main | 1 | 42.90 EFLOPs |
| **Total** | **66** | **119.49 EFLOPs** |

Total: approximately **1.195 × 10^20 FLOPs**. This is a coarse estimate and excludes validation, additional attention costs, recomputation, and other overhead.

For one **NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96GB**, the earlier planning assumption was BF16 training and **100 TFLOP/s sustained training throughput**. Under that illustrative assumption, the updated compute corresponds to approximately **332 hours (~14 days)** before additional overhead. This is not a measured throughput result; actual runtime must be estimated from measured training throughput.

These counts use the latest modified configuration with **12 query heads, 6 KV heads, and FFN dimension 2240**. They supersede earlier calculations for the larger ~713.49M modified model.

## 13. Source material and references

### Local PDF notes already read during planning

- [Qwen3 attention notes](</Users/seungjunlee/Desktop/base model paper/Qwen3_Attention_Box_Notes.pdf>)
- [Qwen3-0.6B architecture notes](</Users/seungjunlee/Desktop/base model paper/Qwen3-0.6B_Architecture_Notes.pdf>)
- [Chinchilla summary](</Users/seungjunlee/Desktop/base model paper/Chinchilla_Paper_Summary.pdf>)
- [FineWeb summary](</Users/seungjunlee/Desktop/base model paper/FineWeb_Paper_Summary.pdf>)
- [GLU variants summary](</Users/seungjunlee/Desktop/base model paper/GLU_Variants_Improve_Transformer_Summary.pdf>)
- [μP / Tensor Programs V summary](</Users/seungjunlee/Desktop/base model paper/muP_TensorProgramsV_Summary.pdf>)

### Primary references

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3-0.6B-Base configuration](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)
- [Qwen3 model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [Official μP library](https://github.com/microsoft/mup)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Training Compute-Optimal Large Language Models / Chinchilla](https://arxiv.org/abs/2203.15556)
- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

