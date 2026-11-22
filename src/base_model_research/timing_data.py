"""Read the original FineWeb-Edu upload for the TinyLlama timing notebook."""

import json
from bisect import bisect_right

import numpy as np

from .config import DEFAULT_TOKENIZER, TINYLLAMA_VOCAB_SIZE, dump_json, json_fingerprint
from .data import TokenStream, file_sha256, load_corpus_tokenizer

QWEN_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
TINYLLAMA_REVISION = "59f6f375b26bde864a6ca194a9a3044570490064"


def _convert_block(source, start, decoder, encoder, block_size):
    """Read through a document boundary, even when one document spans several shards."""
    parts = []
    cursor = start
    while cursor < source.total_tokens:
        ids = source.read(cursor, min(block_size, source.total_tokens - cursor))
        ends = np.flatnonzero(ids == decoder.eos_token_id)
        if ends.size and cursor + len(ids) < source.total_tokens:
            ids = ids[: ends[-1] + 1]
        parts.append(ids)
        cursor += len(ids)
        if ends.size:
            break
    raw = np.concatenate(parts)
    docs = [
        doc.tolist() for doc in np.split(raw, np.flatnonzero(raw == decoder.eos_token_id) + 1) if len(doc)
    ]
    terminated = [doc[-1] == decoder.eos_token_id for doc in docs]
    for doc, ended in zip(docs, terminated):
        if ended:
            doc.pop()
    texts = decoder.backend_tokenizer.decode_batch(docs, skip_special_tokens=False)
    encoded = encoder(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    tokens = []
    for doc, ended in zip(encoded, terminated):
        tokens.extend(doc)
        if ended:
            tokens.append(encoder.eos_token_id)
    return np.asarray(tokens, dtype=np.int64), cursor


class RetokenizedSplit(TokenStream):
    """Index whole-document blocks by target-token offset for lookahead and checkpoint seeks."""

    def __init__(
        self, source, decoder, encoder, manifest, *, limit=None, index_file=None, block_size=131_072
    ):
        self.source = source
        self.decoder = decoder
        self.encoder = encoder
        self.manifest = manifest
        self.fingerprint = manifest["fingerprint"]
        self.limit = limit
        self.index_file = index_file
        self.block_size = block_size
        self.positions = [(0, 0)]  # (TinyLlama token offset, Qwen token offset)
        if index_file and index_file.is_file():
            self.positions = [tuple(pair) for pair in json.loads(index_file.read_text())]
        self.blocks = {}

    def _block(self, i):
        if i not in self.blocks:
            start, source_start = self.positions[i]
            if source_start >= self.source.total_tokens:
                raise ValueError("The FineWeb-Edu source ended before the requested TinyLlama token budget.")
            tokens, source_end = _convert_block(
                self.source, source_start, self.decoder, self.encoder, self.block_size
            )
            end = (start + len(tokens), source_end)
            if i + 1 == len(self.positions):
                self.positions.append(end)
                # On a fresh VM a resumed run can seek close to its checkpoint without replaying the corpus.
                if self.index_file and (i + 1) % 64 == 0:
                    dump_json(self.index_file, self.positions)
            elif self.positions[i + 1] != end:
                raise ValueError("Retokenized data disagrees with the saved token offsets.")
            if self.limit is not None:
                self.blocks.clear()  # Training is sequential. The small validation split is reused 20 times.
            self.blocks[i] = tokens
        return self.blocks[i]

    @property
    def total_tokens(self):
        if self.limit is not None:
            return self.limit
        while self.positions[-1][1] < self.source.total_tokens:
            self._block(len(self.positions) - 1)
        return self.positions[-1][0]

    def read(self, start, count):
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise ValueError("Requested tokens outside the retokenized split")
        result = np.empty(count, dtype=np.int64)
        written = 0
        i = bisect_right(self.positions, (start, self.source.total_tokens)) - 1
        while written < count:
            block = self._block(i)
            offset = start + written - self.positions[i][0]
            part = block[offset : offset + count - written]
            result[written : written + len(part)] = part
            written += len(part)
            i += 1
        return result


def timing_corpora(train, validation, required_tokens, *, run_dir):
    source_name = train.manifest.get("tokenizer")
    if source_name == DEFAULT_TOKENIZER:
        return train, validation
    if source_name != QWEN_TOKENIZER:
        raise ValueError(f"Expected FineWeb-Edu prepared with Qwen or TinyLlama. Found {source_name!r}.")

    original = train.root / "tokenizer"
    decoder = load_corpus_tokenizer(original, local_files_only=True)
    encoder = load_corpus_tokenizer(DEFAULT_TOKENIZER, revision=TINYLLAMA_REVISION)
    if decoder.eos_token_id is None or encoder.eos_token_id is None:
        raise ValueError("Both tokenizers need an EOS token to preserve document boundaries.")
    if len(encoder) != TINYLLAMA_VOCAB_SIZE or max(encoder.get_vocab().values()) != TINYLLAMA_VOCAB_SIZE - 1:
        raise ValueError("The timing model requires the 32,000-token TinyLlama vocabulary.")
    fingerprint = json_fingerprint(
        {
            "conversion": "qwen-documents-to-tinyllama-v1",
            "source": train.fingerprint,
            "decoder": file_sha256(original / "tokenizer.json"),
            "source_eos": decoder.eos_token_id,
            "target": TINYLLAMA_REVISION,
        }
    )
    manifest = {
        "fingerprint": fingerprint,
        "tokenizer": DEFAULT_TOKENIZER,
        "tokenizer_max_id": len(encoder) - 1,
    }
    index_file = run_dir / f"token-offsets-{fingerprint}.json"
    return (
        RetokenizedSplit(train, decoder, encoder, manifest, limit=required_tokens, index_file=index_file),
        RetokenizedSplit(validation, decoder, encoder, manifest),
    )
