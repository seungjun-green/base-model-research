import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

DEFAULT_TOKENIZER = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
TINYLLAMA_VOCAB_SIZE = 32_000
DEFAULT_TRAINING_TARGETS = 3_167_907_840  # 20 targets per parameter in the largest main model.
GATE_VARIANTS = ("add_previous", "scale_previous", "multiply_previous")

SHARED_TRAINING_FIELDS = (
    "context_length",
    "global_batch_size",
    "tokens_per_parameter",
    "seed",
    "warmup_fraction",
    "min_lr_ratio",
    "betas",
    "eps",
    "weight_decay",
    "grad_clip",
    "precision",
)


def load_mapping(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


@dataclass(frozen=True)
class ModelConfig:
    family: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int = TINYLLAMA_VOCAB_SIZE
    base_d_model: int = 256
    base_head_dim: int = 32
    base_d_ff: int = 768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    initializer_std: float = 0.02

    def __post_init__(self):
        if self.family not in ("standard", "modified", *GATE_VARIANTS):
            raise ValueError(f"Unknown model family: {self.family!r}")
        for name in (
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "d_ff",
            "vocab_size",
            "base_d_model",
            "base_head_dim",
            "base_d_ff",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_heads % self.n_kv_heads or self.head_dim % 2:
            raise ValueError("Query heads must divide by KV heads; head_dim must be even")
        if not (
            self.d_model / self.base_d_model
            == self.head_dim / self.base_head_dim
            == self.d_ff / self.base_d_ff
        ):
            raise ValueError("Hidden, head, and FFN widths must share the same width multiplier")

    @property
    def width_mult(self):
        return self.d_model / self.base_d_model

    def at_width(self, multiple):
        dimensions = [self.base_d_model * multiple, self.base_head_dim * multiple, self.base_d_ff * multiple]
        if any(int(x) != x for x in dimensions):
            raise ValueError("Width multiplier must produce integer dimensions")
        return replace(self, d_model=int(dimensions[0]), head_dim=int(dimensions[1]), d_ff=int(dimensions[2]))

    @classmethod
    def load(cls, path):
        return cls(**load_mapping(path))


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    init_multiplier: float
    embedding_multiplier: float
    output_multiplier: float
    attention_multiplier: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError("All hyperparameters must be finite and positive")


@dataclass(frozen=True)
class TrainConfig:
    context_length: int = 2048
    global_batch_size: int = 128
    microbatch_size: int = 4
    tokens_per_parameter: int = 20
    seed: int = 42
    warmup_fraction: float | None = 0.01
    min_lr_ratio: float | None = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    precision: str = "bf16"
    validations_per_run: int = 20
    log_interval: int = 10
    loss_chunk_size: int = 128
    activation_checkpointing: bool = False

    def __post_init__(self):
        object.__setattr__(self, "betas", tuple(self.betas))
        for key in (
            "context_length",
            "global_batch_size",
            "microbatch_size",
            "tokens_per_parameter",
            "validations_per_run",
            "log_interval",
            "loss_chunk_size",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.global_batch_size % self.microbatch_size:
            raise ValueError("microbatch_size must divide global_batch_size")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if self.warmup_fraction is not None and not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.min_lr_ratio is not None and not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if tuple(self.betas) != (0.9, 0.95) or self.eps != 1e-8 or self.weight_decay != 0:
            raise ValueError("The protocol fixes Adam betas=(0.9,0.95), eps=1e-8, weight_decay=0")
        if self.grad_clip != 1.0:
            raise ValueError("The protocol fixes global gradient clipping at 1.0")

    def require_schedule(self):
        if self.warmup_fraction is None or self.min_lr_ratio is None:
            raise ValueError(
                "warmup_fraction and min_lr_ratio must be numeric. The experiment defaults are 0.01 and 0.1."
            )

    @classmethod
    def from_record(cls, values):
        # Older checkpoints and proxy winners recorded fixed step intervals.
        return cls(
            **{k: v for k, v in values.items() if k not in ("validation_interval", "checkpoint_interval")}
        )

    @classmethod
    def load(cls, path, **overrides):
        values = load_mapping(path)
        values.update({k: v for k, v in overrides.items() if v is not None})
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        return cls(**values)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
