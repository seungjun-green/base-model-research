import numpy as np
import pytest
import torch

from base_model_research.config import Hyperparameters, ModelConfig, TrainConfig
from base_model_research.data import TokenWriter, write_manifest


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def hp():
    return Hyperparameters(0.003, 1.0, 1.0, 1.0, 1.0)


@pytest.fixture
def tiny_config():
    return ModelConfig(
        "standard", 4, 1, 2, 1, 2, 4, vocab_size=8, base_d_model=4, base_head_dim=2, base_d_ff=4
    )


@pytest.fixture
def training_config():
    # Tiny synthetic verification budget; production configs retain 20 tokens/parameter.
    return TrainConfig(
        context_length=8,
        global_batch_size=4,
        microbatch_size=2,
        tokens_per_parameter=1,
        precision="fp32",
        warmup_fraction=0.01,
        min_lr_ratio=0.1,
        log_interval=1,
        loss_chunk_size=8,
    )


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "tokens"
    splits = {}
    # The 208-parameter gated fixture needs 20 targets per parameter plus one lookahead token.
    for split, n in (("train", 20 * 208 + 1), ("validation", 35)):
        writer = TokenWriter(root, split, shard_tokens=31)
        writer.write(np.arange(n) % 8)
        splits[split] = writer.finish()
    write_manifest(root, splits, tokenizer_max_id=7, tokenizer="synthetic")
    return root
