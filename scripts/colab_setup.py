"""Standard-library setup embedded in the notebooks so it can run before the checkout exists."""

import fcntl
import json
import re
import subprocess
import sys
from venv import EnvBuilder

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


def _snapshot_pins(text):
    # Earlier notebooks included the editable checkout. Reinstall it from the saved commit.
    pins = []
    comment = ""
    for line in text.splitlines():
        if line.startswith("#"):
            comment = line
            continue
        if line.startswith("-e ") and re.search(r"(?i)base[-_]model[-_]research\b", line + " " + comment):
            comment = ""
            continue
        if line.strip():
            pins.append(line)
        comment = ""
    return "\n".join(sorted(pins)) + "\n"


def _installed_pins(pip):
    frozen = subprocess.check_output([*pip, "freeze", "--exclude", "base-model-research"], text=True)
    return _snapshot_pins(frozen)


def _uses_checkout(pip, repo):
    packages = subprocess.check_output(
        [*pip, "list", "--editable", "--format=json", "--disable-pip-version-check"], text=True
    )
    return any(
        package["name"] == "base-model-research" and package.get("editable_project_location") == str(repo)
        for package in json.loads(packages)
    )


def _prepare_venv(directory):
    # Colab's system Python can lack ensurepip. Finish partial environments in place.
    EnvBuilder(symlinks=True, with_pip=False).create(directory)
    python = directory / "bin" / "python"
    return python, [sys.executable, "-m", "pip", "--python", str(python)]


def setup_colab(workspace, runs, *, benchmark):
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        repo = _checkout(workspace, runs)
        role = "benchmark" if benchmark else "training"
        venv = workspace / "envs" / runs.name / repo.name / role
        python, pip = _prepare_venv(venv)
        snapshot = runs / ("requirements.benchmark.freeze.txt" if benchmark else "requirements.freeze.txt")
        if snapshot.exists():
            pins = _snapshot_pins(snapshot.read_text())
            if _uses_checkout(pip, repo) and _installed_pins(pip) == pins:
                return repo, python
            requirements = venv / "recorded-requirements.txt"
            requirements.write_text(pins)
        else:
            pins = None
            requirements = repo / ("requirements.txt" if benchmark else "requirements-train.txt")
            if not requirements.exists():  # Older pinned revisions used a single requirements file.
                requirements = repo / "requirements.txt"
        subprocess.run([*pip, "install", "pip", "-r", requirements, "-e", repo], check=True)
        frozen = _installed_pins(pip)
        if pins is None:
            temporary = snapshot.with_suffix(".tmp")
            temporary.write_text(frozen)
            temporary.replace(snapshot)
        elif frozen != pins:
            raise RuntimeError(f"The installed {role} packages differ from {snapshot}.")
    return repo, python
