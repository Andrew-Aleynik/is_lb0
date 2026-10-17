"""Универсальные утилиты запусков курса («Правила», п. 4.2–4.4)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

try:
    import mlflow
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Нужен пакет mlflow. Установите зависимости: pip install -r requirements.txt"
    ) from exc


# --- JSON / YAML / JSONL -----------------------------------------------------


def to_plain(obj: Any) -> Any:
    """Датакласс → dict/list для сериализации; остальное без изменений."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_plain(obj), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(obj), f, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_plain(dict(row)), ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- Общие хелперы запуска ---------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(f"{stamp}-{time.time_ns()}".encode()).hexdigest()[:8]
    return f"{stamp}_{suffix}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def library_versions(packages: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Версии Python и переданных модулей (ключ → модуль с __version__)."""
    versions = {"python": platform.python_version()}
    if packages:
        for name, mod in packages.items():
            versions[name] = getattr(mod, "__version__", "unknown")
    return versions


def git_commit_and_dirty(allow_dirty: bool, *, cwd: Path | None = None) -> str:
    """Вернуть полный хеш HEAD; при грязном дереве — ошибка, если не allow_dirty."""
    root = cwd or Path.cwd()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode != 0:
        raise SystemExit("Не удалось выполнить git status (нужен git-репозиторий).")
    porcelain = dirty.stdout.strip()
    if porcelain and not allow_dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен "
            "(«Правила», п. 4.3). Для отладки: --allow-dirty.\n" + porcelain
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        raise SystemExit("Не удалось получить git rev-parse HEAD.")
    return commit.stdout.strip()


# --- MLflow ------------------------------------------------------------------


def setup_mlflow(
    experiment: str,
    *,
    root: Path | None = None,
    db_name: str = "mlflow.db",
) -> Path:
    db_path = ((root or Path.cwd()) / db_name).resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(str(experiment))
    return db_path


@contextmanager
def mlflow_run(
    run_name: str,
    *,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})
        yield run


def log_metrics(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            if step is None:
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_metric(key, float(value), step=step)


def log_artifacts(path: Path) -> None:
    mlflow.log_artifacts(str(path))
