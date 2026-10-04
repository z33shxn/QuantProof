"""Reproducibility manifest.

Every audit produces a manifest recording what was audited (strategy and data
identities and hashes), how (configuration, validation methodology, execution
assumptions, seed), and where (Python, platform, package versions, Git commit).

``content_hash`` covers every field *except* the run timestamp, so two runs with
the same code, data, configuration, environment and seed produce the same
``content_hash``.
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from quantproof._utils import to_jsonable
from quantproof._version import __version__
from quantproof.experiments.hashing import hash_config

TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "pydantic", "pyarrow", "scikit-learn")


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions of key dependencies (None if not installed)."""
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_info(path: str | Path | None) -> dict[str, Any]:
    """Commit hash and dirty flag of the Git repository containing ``path`` (if any)."""
    if path is None:
        return {"commit": None, "dirty": None}
    cwd = Path(path).resolve()
    cwd = cwd if cwd.is_dir() else cwd.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        return {"commit": commit or None, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def build_manifest(
    *,
    strategy: dict[str, Any],
    data: dict[str, Any],
    config: dict[str, Any],
    validation: dict[str, Any],
    execution: dict[str, Any],
    seed: int,
    parameters: dict[str, Any] | None = None,
    lineage: list[dict[str, Any]] | None = None,
    git_path: str | Path | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dictionary (JSON/YAML serializable)."""
    timestamp = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    body: dict[str, Any] = {
        "quantproof_version": __version__,
        "strategy": strategy,
        "data": data,
        "parameters": parameters or {},
        "validation": validation,
        "execution": execution,
        "configuration": {"sha256": hash_config(config), "values": config},
        "experiment": {"random_seed": seed, "git": git_info(git_path)},
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(terse=True),
            "packages": package_versions(),
        },
        "lineage": lineage or [],
    }
    body = to_jsonable(body)
    content_hash = hash_config(body)
    body["content_hash"] = content_hash
    body["created_at"] = timestamp.isoformat()
    body["timezone"] = "UTC"
    return body


def manifest_to_yaml(manifest: dict[str, Any]) -> str:
    """YAML rendering of a manifest."""
    return yaml.safe_dump(to_jsonable(manifest), sort_keys=False, allow_unicode=True)
