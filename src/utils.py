from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LOGGER_NAME = "eurosat"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    short_commit: str | None
    is_dirty: bool
    branch: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "short_commit": self.short_commit,
            "is_dirty": self.is_dirty,
            "branch": self.branch,
        }


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def get_git_info() -> GitInfo:
    try:
        commit = _run_git(["rev-parse", "HEAD"]) or None
        short_commit = _run_git(["rev-parse", "--short", "HEAD"]) or None
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or None
        dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
        return GitInfo(commit=commit, short_commit=short_commit, is_dirty=dirty, branch=branch)
    except Exception:
        return GitInfo(commit=None, short_commit=None, is_dirty=False, branch=None)


def make_run_id(git_info: GitInfo) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = git_info.short_commit or "nogit"
    return f"{timestamp}_{suffix}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
