"""Qwen3-style blocks implemented in PyTorch; no pretrained weights are loaded.

Architecture reference: Hugging Face Transformers Qwen3 (Apache-2.0).
Our attention uses the experiment's muP 1/head_dim scaling.
"""

import torch
from mup import MuSharedReadout
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import GATE_VARIANTS, Hyperparameters, ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def apply_rope(x, positions, theta):
    # Qwen's split-half rotary convention, with frequencies computed in FP32.
    inv_freq = theta ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32) / x.shape[-1])
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None, None]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config, multiplier):
        super().__init__()
        self.config = config
        self.scale = multiplier / config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, x):
        b, length, _ = x.shape
        c = self.config
        q = self.q_proj(x).view(b, length, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, length, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, length, c.n_kv_heads, c.head_dim).transpose(1, 2)
        positions = torch.arange(length, device=x.device)
        q = apply_rope(self.q_norm(q), positions, c.rope_theta)
        k = apply_rope(self.k_norm(k), positions, c.rope_theta)
        # Native GQA avoids explicit repeated KV tensors on supported CUDA kernels.
        if x.is_cuda:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0, scale=self.scale, enable_gqa=True
            )
        else:
            repeats = c.n_heads // c.n_kv_heads
            out = F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(repeats, dim=1),
                v.repeat_interleave(repeats, dim=1),
                is_causal=True,
                dropout_p=0.0,
                scale=self.scale,
            )
        return self.o_proj(out.transpose(1, 2).reshape(b, length, c.n_heads * c.head_dim))


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ConnectionGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.current = nn.Linear(width, width, bias=False)
        self.previous = nn.Linear(width, width, bias=False)

    def forward(self, current_input, previous_output):
        return F.gelu(self.current(current_input), approximate="none") * self.previous(previous_output)


class ResidualTanhGate(ConnectionGate):
    def forward(self, current_input, previous_output):
        product = super().forward(current_input, previous_output)
        return current_input + product.tanh()


class DecoderLayer(nn.Module):
    def __init__(self, config, hp, *, layer_index=0):
        super().__init__()
        self.modified = config.family in ("modified", "residual_tanh")
        self.previous_attention_relu = config.family in (
            "previous_attention_relu",
            "previous_attention_relu_with_alpha",
        )
        self.previous_attention_alpha = None
        if config.family == "previous_attention_relu_with_alpha" and layer_index > 0:
            self.previous_attention_alpha = nn.Parameter(torch.zeros(()))
        self.input_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = Attention(config, hp.attention_multiplier)
        self.ffn = FFN(config)
        if self.modified:
            gate = ResidualTanhGate if config.family == "residual_tanh" else ConnectionGate
            self.attention_gate = gate(config.d_model)
            self.ffn_gate = gate(config.d_model)

    def forward(self, x, previous_attention=None, previous_ffn=None):
        attention = self.attention(self.input_norm(x))
        if self.modified:
            h = attention + self.attention_gate(x, x if previous_attention is None else previous_attention)
        else:
            h = x + attention
            if self.previous_attention_relu:
                if previous_attention is not None:
                    if self.previous_attention_alpha is not None:
                        previous_attention = self.previous_attention_alpha * previous_attention
                    h = h + previous_attention
                h = F.relu(h)
        ffn = self.ffn(self.post_attention_norm(h))
        if self.modified:
            result = ffn + self.ffn_gate(h, h if previous_ffn is None else previous_ffn)
        else:
            result = h + ffn
        # Preserve the autograd graph: the previous branch must not be detached.
        return result, attention, ffn


class CurrentProductWithSkip(nn.Module):
    def __init__(self, width, *, learned_scale):
        super().__init__()
        self.input_proj = nn.Linear(width, width, bias=False)
        self.output_proj = nn.Linear(width, width, bias=False)
        self.scale = nn.Parameter(torch.ones(1)) if learned_scale else None

    def forward(self, x, current, previous):
        skip = previous if self.scale is None else self.scale.to(previous.dtype) * previous
        return F.gelu(self.input_proj(x)) * self.output_proj(current) + skip


class ResidualTimesPrevious(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.previous_proj = nn.Linear(width, width, bias=False)

    def forward(self, x, current, previous):
        return F.gelu(F.relu(x + current)) * self.previous_proj(previous)


class GateVariantLayer(DecoderLayer):
    def __init__(self, config, hp):
        super().__init__(config, hp)
        if config.family == "multiply_previous":
            self.attention_gate = ResidualTimesPrevious(config.d_model)
            self.ffn_gate = ResidualTimesPrevious(config.d_model)
        else:
            learned_scale = config.family == "scale_previous"
            self.attention_gate = CurrentProductWithSkip(config.d_model, learned_scale=learned_scale)
            self.ffn_gate = CurrentProductWithSkip(config.d_model, learned_scale=learned_scale)

    def forward(self, x, previous_attention, previous_ffn):
        attention = self.attention(self.input_norm(x))
        h = self.attention_gate(x, attention, previous_attention)
        ffn = self.ffn(self.post_attention_norm(h))
        return self.ffn_gate(h, ffn, previous_ffn), attention, ffn


class ResearchLM(nn.Module):
    def __init__(self, config: ModelConfig, hp: Hyperparameters):
        super().__init__()
        self.config = config
        self.hp = hp
        self.activation_checkpointing = False
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # The three comparison variants use a standard residual in the first layer,
        # where no previous attention or FFN output exists.
        self.layers = nn.ModuleList(
            GateVariantLayer(config, hp)
            if index > 0 and config.family in GATE_VARIANTS
            else DecoderLayer(config, hp, layer_index=index)
            for index in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.readout = MuSharedReadout(self.embedding.weight, bias=False, output_mult=hp.output_multiplier)

    def hidden_states(self, input_ids):
        x = self.embedding(input_ids) * self.hp.embedding_multiplier
        previous_attention, previous_ffn = None, None
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                x, previous_attention, previous_ffn = checkpoint(
                    layer, x, previous_attention, previous_ffn, use_reentrant=False
                )
            else:
                x, previous_attention, previous_ffn = layer(x, previous_attention, previous_ffn)
        return self.final_norm(x)

    def forward(self, input_ids):
        return self.readout(self.hidden_states(input_ids))

    def loss_sum(self, input_ids, targets, chunk_size=128):
        """Summed FP32 CE over unmasked next-token targets.

        Checkpoint each readout/loss chunk to avoid retaining B*S*vocab_size logits
        for backward. Recomputing the chunks trades compute for bounded memory.
        """
        hidden = self.hidden_states(input_ids).reshape(-1, self.config.d_model)
        targets = targets.reshape(-1)

        def chunk_loss(h, y):
            return F.cross_entropy(self.readout(h).float(), y, ignore_index=-100, reduction="sum")

        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(targets), chunk_size):
            h, y = hidden[start : start + chunk_size], targets[start : start + chunk_size]
            if self.training and torch.is_grad_enabled():
                total = total + checkpoint(chunk_loss, h, y, use_reentrant=False)
            else:
                total = total + chunk_loss(h, y)
        return total


def parameter_count(model):
    # parameters() deduplicates the tied embedding/readout Parameter.
    return sum(parameter.numel() for parameter in model.parameters())
