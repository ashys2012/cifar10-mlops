"""Small helpers for keeping MLflow runs reproducible and well-documented.

If you already have a tracking.py that train_step.py / evaluate_step.py
import from, ignore this file — it's included so the example is runnable
end-to-end and so the expected function signatures are clear.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import mlflow


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def log_git_info() -> None:
    """Tag the active MLflow run with the current commit, branch, and dirty state."""
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = _git("status", "--porcelain")

    if commit:
        mlflow.set_tag("git.commit", commit)
    if branch:
        mlflow.set_tag("git.branch", branch)
    mlflow.set_tag("git.dirty", bool(dirty))


def get_logbook() -> str:
    """Return a short markdown note describing this run, logged as an artefact."""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"# Run logbook\n\n"
        f"- Timestamp: {timestamp}\n"
        f"- Git branch: `{branch}`\n"
        f"- Git commit: `{commit}`\n"
    )


def log_model_architecture(model) -> None:
    """Log a text dump of the model's architecture as an MLflow artefact."""
    mlflow.log_text(str(model), "model_architecture.txt")
    num_params = sum(p.numel() for p in model.parameters())
    mlflow.log_param("num_parameters", num_params)


def generate_reproduce_run_command(run_id: str, experiment_id: str) -> str:
    """Return a shell command that reproduces this exact run."""
    commit = _git("rev-parse", "HEAD") or "<unknown-commit>"
    return f"git checkout {commit} && mlflow runs describe --run-id {run_id} && python run.py"
