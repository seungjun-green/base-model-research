"""Standard-library setup embedded in the notebooks so it can run before the checkout exists."""

import fcntl
import json
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

GITHUB_REPO_URL = "https://github.com/seungjun-green/base-model-research"


def _checkout(workspace, runs):
    source_file = runs / "source.json"
    if source_file.exists():
        source = json.loads(source_file.read_text())
        if source["url"].rstrip("/").removesuffix(".git").lower() != GITHUB_REPO_URL.lower():
            raise ValueError(
                f"{runs} is already pinned to a different repository. Use a new EXPERIMENT name."
            )
        commit = source["commit"]
    else:
        head = subprocess.check_output(
            ["git", "ls-remote", "--exit-code", GITHUB_REPO_URL, "HEAD"], text=True
        )
        commit = head.split()[0]
        source = {"url": GITHUB_REPO_URL, "commit": commit}
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Invalid Git commit for this experiment: {commit!r}")

    repo = workspace / "repos" / commit
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--", GITHUB_REPO_URL, repo], check=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError(f"The runtime checkout has local edits: {repo}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != commit:
        subprocess.run(["git", "fetch", "origin", commit], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--detach", commit], cwd=repo, check=True)
    if not source_file.exists():
        temporary = source_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(source, indent=2) + "\n")
        temporary.replace(source_file)
    return repo


def setup_colab(workspace, runs, *, benchmark):
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        repo = _checkout(workspace, runs)
        try:
            torch_version = version("torch")
        except PackageNotFoundError as error:
            raise RuntimeError("Select a Colab runtime with PyTorch installed and rerun setup.") from error

        # Keep the runtime's PyTorch build, including its CUDA version suffix.
        extras = "data,eval" if benchmark else "data"
        print(f"Using {sys.executable} with PyTorch {torch_version}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                f"torch=={torch_version}",
                "-e",
                f"{repo}[{extras}]",
            ],
            check=True,
        )
    return repo
