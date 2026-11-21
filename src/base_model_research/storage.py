import json
import os
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from .config import dump_json
from .data import TokenCorpus, file_sha256


@contextmanager
def run_lock(directory, *, blocking=False):
    """Advisory process lock; OS releases it after a crash (Linux/macOS runtimes)."""
    import fcntl

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / ".lock", "a+") as stream:
        try:
            flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
            fcntl.flock(stream, flags)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def copy_prepared_data(source, destination):
    """Copy an immutable corpus between Drive and the runtime SSD, resuming verified shards."""
    source, destination = Path(source), Path(destination)
    manifest = (source / "manifest.json").read_bytes()
    with run_lock(destination.parent, blocking=True):
        if destination.exists():
            if (destination / "manifest.json").read_bytes() != manifest:
                raise ValueError(f"Different dataset already exists at {destination}")
            return
        staging = destination.with_name(destination.name + ".copying")
        staging.mkdir(exist_ok=True)
        staged_manifest = staging / "manifest.json"
        if staged_manifest.exists():
            if staged_manifest.read_bytes() != manifest:
                raise ValueError(f"Partial copy belongs to a different dataset: {staging}")
        else:
            temporary = staging / "manifest.json.tmp"
            temporary.write_bytes(manifest)
            temporary.replace(staged_manifest)
        for split in ("train", "validation"):
            for shard in TokenCorpus(source, split).entries:
                name = shard["path"]
                if Path(name).name != name:
                    raise ValueError(f"Expected a shard filename: {name}")
                target = staging / name
                if (
                    target.is_file()
                    and target.stat().st_size == shard["tokens"] * 4
                    and file_sha256(target) == shard["sha256"]
                ):
                    continue
                shutil.copyfile(source / name, target)
                if file_sha256(target) != shard["sha256"]:
                    raise OSError(f"Copied shard failed verification: {target}")
        shutil.copytree(
            source / "tokenizer", staging / "tokenizer", dirs_exist_ok=True, copy_function=shutil.copyfile
        )
        staging.replace(destination)


def _write_checkpoint(directory, payload, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    temporary = directory / (name + ".tmp")
    with open(temporary, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {"file": name, "sha256": file_sha256(path), "step": payload["step"]}


def save_checkpoint(directory, payload, keep=2):
    """Write the legacy latest-checkpoint format."""
    if keep < 1:
        raise ValueError("Keep at least the current checkpoint")
    directory = Path(directory)
    name = f"step-{payload['step']:08d}-{uuid.uuid4().hex[:8]}.pt"
    pointer = _write_checkpoint(directory, payload, name)
    path = directory / name
    dump_json(directory / "latest.json", pointer)
    # Switch the pointer only after the complete file exists. Keep rollback copies.
    backups = (p for p in directory.glob("step-*.pt") if p != path)
    backups = sorted(backups, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    for old in backups[keep - 1 :]:
        old.unlink()
    return path


def ranked_checkpoints(directory):
    index = directory / "best.json"
    return json.loads(index.read_text()) if index.is_file() else []


def validation_checkpoint_name(step):
    return f"validation-{step:08d}.pt"


def save_validation_checkpoint(directory, payload):
    """Publish the new resume point after its complete file is on disk."""
    name = validation_checkpoint_name(payload["step"])
    pointer = _write_checkpoint(directory, payload, name)
    dump_json(directory / "latest.json", pointer)
    return directory / name


def load_checkpoint(directory_or_file, *, return_checksum=False):
    path = Path(directory_or_file)
    if path.is_dir():
        latest = path / "latest.json"
        if latest.is_file():
            pointer = json.loads(latest.read_text())
        else:
            ranking = ranked_checkpoints(path)  # Archived weights-only runs.
            if not ranking:
                raise FileNotFoundError(f"No published checkpoint in {path}")
            pointer = ranking[0]
        path = path / pointer["file"]
        checksum = file_sha256(path)
        if checksum != pointer["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
    elif return_checksum:
        checksum = file_sha256(path)
    # Checkpoints include RNG/optimizer state. Only load files from trusted runs.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (payload, checksum) if return_checksum else payload


def sync_to_remote(source, destination):
    """Use a user-configured rclone remote, including Google Drive.

    Verify files before publishing their indices and results. For ranked runs,
    prune only weight files belonging to that run after publishing its index.
    """
    source = str(Path(source).resolve())
    # rclone basename patterns apply at every directory depth, including the root.
    # A restored sweep treats result.json as terminal, so publish it only after
    # both its checkpoint and latest.json have reached the destination.
    markers = ("best.json", "result.json", "best_hparams.json")
    excluded = [arg for name in (".lock", "*.tmp", "latest.json", *markers) for arg in ("--exclude", name)]
    subprocess.run(["rclone", "copy", source, destination, *excluded], check=True)
    subprocess.run(["rclone", "check", source, destination, "--one-way", "--download", *excluded], check=True)
    subprocess.run(["rclone", "copy", source, destination, "--include", "latest.json"], check=True)
    for name in markers:
        subprocess.run(["rclone", "copy", source, destination, "--include", name], check=True)
    # Only prune weight files owned by this ranked run, after its index is published.
    for index in Path(source).rglob("best.json"):
        if index.parent.name != "checkpoints":
            continue
        run = json.loads((index.parent.parent / "run.json").read_text())
        fingerprint = run["fingerprint"]
        if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("Invalid ranked checkpoint fingerprint")
        remote_dir = f"{str(destination).rstrip('/')}/{index.parent.relative_to(source).as_posix()}"
        subprocess.run(
            ["rclone", "sync", index.parent, remote_dir, "--include", f"/ranked-{fingerprint}-*.pt"],
            check=True,
        )
