"""Immutable uint32 token shards, document-disjoint validation, exact token cursors."""

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .config import dump_json, json_fingerprint


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenWriter:
    def __init__(self, root, split, shard_tokens=64_000_000):
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        self.root = Path(root)
        self.split = split
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.entries = []
        self.stream = None
        self.count = 0
        self.total = 0

    def _close(self):
        if self.stream is not None:
            self.stream.close()
            path = self.root / f"{self.split}-{len(self.entries):05d}.bin"
            self.entries.append({"path": path.name, "tokens": self.count, "sha256": file_sha256(path)})
            self.stream = None
            self.count = 0

    def write(self, tokens):
        tokens = np.asarray(tokens, dtype="<u4")
        offset = 0
        while offset < len(tokens):
            if self.stream is None:
                # Lifetime spans writes; finish() closes the current shard.
                self.stream = open(self.root / f"{self.split}-{len(self.entries):05d}.bin", "wb")  # noqa: SIM115
            take = min(len(tokens) - offset, self.shard_tokens - self.count)
            self.stream.write(tokens[offset : offset + take].tobytes())
            offset += take
            self.count += take
            self.total += take
            if self.count == self.shard_tokens:
                self._close()

    def finish(self):
        self._close()
        return {"tokens": self.total, "shards": self.entries}


def write_manifest(root, splits, **metadata):
    manifest = {"format_version": 1, "dtype": "<u4", "splits": splits, **metadata}
    manifest["fingerprint"] = json_fingerprint(manifest)
    dump_json(Path(root) / "manifest.json", manifest)
    return manifest


class TokenCorpus:
    def __init__(self, root, split, verify=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        raw = {k: v for k, v in self.manifest.items() if k != "fingerprint"}
        expected = json_fingerprint(raw)
        if self.manifest.get("fingerprint") != expected:
            raise ValueError("Dataset manifest fingerprint mismatch")
        if self.manifest["format_version"] != 1 or self.manifest["dtype"] != "<u4":
            raise ValueError("Unsupported dataset format")
        self.entries = self.manifest["splits"][split]["shards"]
        self.total_tokens = self.manifest["splits"][split]["tokens"]
        self.starts = np.cumsum([0] + [e["tokens"] for e in self.entries]).tolist()
        if self.starts[-1] != self.total_tokens or self.total_tokens < 2:
            raise ValueError("Invalid or empty token corpus")
        self._maps = OrderedDict()
        for entry in self.entries:
            path = self.root / entry["path"]
            if path.stat().st_size != entry["tokens"] * 4:
                raise ValueError(f"Shard size mismatch: {path}")
            if verify and file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {path}")

    @property
    def fingerprint(self):
        return self.manifest["fingerprint"]

    def read(self, start, count):
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise ValueError("Requested tokens outside the prepared corpus")
        tokens = np.empty(count, dtype=np.int64)
        written = 0
        while written < count:
            idx = bisect_right(self.starts, start) - 1
            if idx not in self._maps:
                self._maps[idx] = np.memmap(self.root / self.entries[idx]["path"], dtype="<u4", mode="r")
            self._maps.move_to_end(idx)
            while len(self._maps) > 2:
                self._maps.popitem(last=False)
            take = min(count - written, self.starts[idx + 1] - start)
            offset = start - self.starts[idx]
            tokens[written : written + take] = self._maps[idx][offset : offset + take]
            start += take
            written += take
        return tokens

    def batch(self, start, target_count, context_length):
        """Every input position has one shifted target, including the block boundary.

        The final partial block is padded and masked; only target_count positions
        contribute to the objective or the training budget.
        """
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        tokens = self.read(start, target_count + 1)
        padding = -target_count % context_length
        if padding:
            tokens = np.pad(tokens, (0, padding))
        x = torch.from_numpy(tokens[:-1].reshape(-1, context_length))
        y = tokens[1:].copy()
        y[target_count:] = -100
        return x, torch.from_numpy(y.reshape(-1, context_length))


def prepare_data(
    output,
    max_training_tokens=11_957_698_560,
    validation_documents=2048,
    dataset="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-100BT",
    tokenizer_name="Qwen/Qwen3-0.6B-Base",
    dataset_revision=None,
    tokenizer_revision=None,
    shard_tokens=64_000_000,
):
    """Download/tokenize once. Validation uses the first N complete documents.

    Preparation is deliberately staged: an interrupted directory is never mistaken
    for a complete corpus. Re-run with a new output or remove the incomplete staging
    directory explicitly. Training itself resumes by absolute target-token offset.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    output = Path(output)
    staging = output.with_name(output.name + ".preparing")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Output or staging directory already exists: {output}, {staging}")
    if max_training_tokens <= 0 or validation_documents <= 0:
        raise ValueError("Training tokens and validation document count must be positive")
    api = HfApi()
    dataset_revision = api.dataset_info(dataset, revision=dataset_revision or "main").sha
    tokenizer_revision = api.model_info(tokenizer_name, revision=tokenizer_revision or "main").sha
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token for document boundaries")
    documents = iter(
        load_dataset(dataset, name=dataset_config, split="train", streaming=True, revision=dataset_revision)
    )
    staging.mkdir(parents=True)
    tokenizer.save_pretrained(staging / "tokenizer")
    train = TokenWriter(staging, "train", shard_tokens)
    validation = TokenWriter(staging, "validation", shard_tokens)
    identities = hashlib.sha256()
    seen = 0
    required = max_training_tokens + 1  # next-token lookahead
    try:
        while train.total < required:
            batch = []
            for _ in range(128):
                try:
                    batch.append(next(documents))
                except StopIteration:
                    break
            if not batch:
                raise ValueError(
                    "Dataset exhausted before the requested budget; choose a larger dataset config"
                )
            encoded = tokenizer([row["text"] for row in batch], add_special_tokens=False)["input_ids"]
            for row, tokens in zip(batch, encoded):
                tokens = tokens + [tokenizer.eos_token_id]
                identity = str(row.get("id", row.get("url", seen)))
                identities.update(f"{seen}:{identity}\n".encode())
                if seen < validation_documents:
                    validation.write(tokens)
                else:
                    train.write(tokens[: required - train.total])
                seen += 1
                if train.total == required:
                    break
            print(f"Prepared {train.total:,}/{required:,} training tokens; {seen:,} documents", flush=True)
    finally:
        train_info, validation_info = train.finish(), validation.finish()
    manifest = write_manifest(
        staging,
        {"train": train_info, "validation": validation_info},
        dataset=dataset,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        tokenizer=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        tokenizer_max_id=max(tokenizer.get_vocab().values()),
        validation_documents=validation_documents,
        documents_seen=seen,
        document_order_sha256=identities.hexdigest(),
        packing="continuous_document_stream_with_eos; shifted_targets; reset_positions_per_sequence",
    )
    staging.replace(output)
    return manifest
