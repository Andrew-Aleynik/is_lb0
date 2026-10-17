import argparse
import csv
import random
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from dacite import from_dict
from pydantic import Field, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import optuna

from run_utils import (
    git_commit_and_dirty,
    library_versions,
    log_artifacts,
    log_metrics,
    make_run_id,
    mlflow_run,
    read_yaml,
    setup_mlflow,
    sha256_file,
    utc_now_iso,
    write_json,
    write_yaml,
)

ROOT = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    n_layers: Annotated[int, Field(gt=0)]
    hidden: Annotated[int, Field(gt=0)]
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)]


@dataclass
class TrainConfig:
    lr: Annotated[float, Field(gt=0.0)]
    weight_decay: Annotated[float, Field(ge=0.0)]
    batch_size: Annotated[int, Field(gt=0)]
    epochs: Annotated[int, Field(gt=0)]


@dataclass
class RuntimeConfig:
    device: Annotated[str, Field(...)]
    num_threads: Annotated[int, Field(gt=0)]


@dataclass
class Config:
    experiment: Annotated[str, Field(...)]
    split_seed: Annotated[int, Field(...)]
    seed: Annotated[int, Field(...)]
    model: Annotated[ModelConfig, Field(...)]
    train: Annotated[TrainConfig, Field(...)]
    runtime: Annotated[RuntimeConfig, Field(...)]


@dataclass
class SplitConfig:
    split_seed: Annotated[int, Field(...)]
    train: Annotated[list[int], Field(...)]
    val: Annotated[list[int], Field(...)]
    test: Annotated[list[int], Field(...)]


@dataclass
class RunInfo:
    run_id: Annotated[str, Field(...)]
    lab: Annotated[str, Field(...)]
    series: Annotated[str, Field(...)]
    status: Annotated[str, Field(...)]
    started_at: Annotated[str, Field(...)]
    ended_at: Annotated[str | None, Field(...)]
    commit: Annotated[str, Field(...)]
    versions: Annotated[dict[str, str], Field(...)]
    device: Annotated[str, Field(...)]
    num_threads: Annotated[int, Field(...)]
    seed: Annotated[int, Field(...)]
    data_sha256: Annotated[dict[str, str], Field(...)]
    model: Annotated[ModelConfig, Field(...)]
    error: Annotated[dict[str, str] | None, Field(...)]


class MLP(nn.Module):
    def __init__(self, hidden: int, n_layers: int, dropout: float):
        if n_layers <= 0 or hidden <= 0 or dropout < 0:
            raise ValueError("Заданы неправильные параметры нейросети")
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 64
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seeds(seed: int, num_threads: int = 1) -> torch.Generator:
    torch.set_num_threads(num_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def prepare_data(split_seed: int = 0):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    idx = np.arange(len(y))
    rest, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=split_seed
    )
    train_idx, val_idx = train_test_split(
        rest, test_size=0.25, stratify=y[rest], random_state=split_seed
    )
    assert (len(train_idx), len(val_idx), len(test_idx)) == (1077, 360, 360)
    assert len(train_idx) + len(val_idx) + len(test_idx) == 1797
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    for part in (train_idx, val_idx, test_idx):
        assert set(y[part]) == set(range(10))
    return X, y, train_idx, val_idx, test_idx


def make_loader(
    X,
    y,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader:
    xt = torch.tensor(X[indices], dtype=torch.float32)
    yt = torch.tensor(y[indices], dtype=torch.long)
    return DataLoader(
        TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(yb)
        pred = logits.argmax(dim=1)
        y_true_all.extend(yb.cpu().tolist())
        y_pred_all.extend(pred.cpu().tolist())
    n = len(y_true_all)
    acc = sum(int(a == b) for a, b in zip(y_true_all, y_pred_all)) / n
    return total_loss / n, acc, y_true_all, y_pred_all


def train_epochs(
    params: dict[str, Any],
    seed: int = 0,
    epochs: int = 30,
    *,
    split_seed: int = 0,
    device: str = "cpu",
    num_threads: int = 1,
) -> Iterator[tuple[int, float]]:
    """Обучение одной конфигурации; после каждой эпохи — (epoch, val_acc). Для Optuna."""
    g = set_seeds(seed, num_threads)
    dev = torch.device(device)
    X, y, train_idx, val_idx, _ = prepare_data(split_seed)
    model = MLP(
        hidden=int(params["hidden"]),
        n_layers=int(params["n_layers"]),
        dropout=float(params["dropout"]),
    ).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    train_loader = make_loader(
        X, y, train_idx, int(params["batch_size"]), True, g
    )
    val_loader = make_loader(
        X, y, val_idx, int(params["batch_size"]), False, None
    )

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        _, val_acc, _, _ = evaluate(model, val_loader, criterion, dev)
        yield epoch, float(val_acc)


def load_config(config_path: Path, seed_override: int | None) -> Config:
    raw_data = read_yaml(config_path)
    if not isinstance(raw_data, dict):
        raise SystemExit(f"Конфигурация должна быть mapping: {config_path}")
    if seed_override is not None:
        raw_data["seed"] = seed_override
    try:
        TypeAdapter(Config).validate_python(raw_data)
    except ValidationError as e:
        raise SystemExit(f"Неправильный конфиг: {e}") from e
    return from_dict(data_class=Config, data=raw_data)


def fmt_float(x: float) -> str:
    """Стабильная запись float для побайтового совпадения CSV."""
    return format(float(x), ".17g")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ЛР0: один запуск обучения MLP")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "base.yaml",
        help="путь к YAML-конфигурации",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed обучения")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="оценить на test после обучения",
    )
    parser.add_argument("--series", type=str, default="debug", help="имя серии запусков")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="отладочный запуск при незакоммиченных изменениях",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.is_file():
        raise SystemExit(f"Нет конфигурации: {config_path}")

    commit = git_commit_and_dirty(args.allow_dirty, cwd=ROOT)
    config = load_config(config_path, args.seed)
    eval_test = bool(args.eval_test)
    series = args.series

    g = set_seeds(config.seed, config.runtime.num_threads)
    device = torch.device(config.runtime.device)

    X, y, train_idx, val_idx, test_idx = prepare_data(config.split_seed)
    split = SplitConfig(
        split_seed=config.split_seed,
        train=train_idx.tolist(),
        val=val_idx.tolist(),
        test=test_idx.tolist(),
    )
    split_path = ROOT / "data" / "split.json"
    write_json(split_path, split)
    split_sha = sha256_file(split_path)
    write_json(
        ROOT / "data" / "manifest.json",
        {
            "datasets": {
                "load_digits": {
                    "source": "sklearn.datasets.load_digits",
                    "version": sklearn.__version__,
                    "selection": {
                        "rule": (
                            "stratified split: test=20%, then val=25% of remainder; "
                            f"split_seed={config.split_seed}"
                        ),
                        "available": int(len(y)),
                        "taken": int(len(y)),
                        "excluded": 0,
                        "excluded_reason": None,
                        "sizes": {
                            "train": len(train_idx),
                            "val": len(val_idx),
                            "test": len(test_idx),
                        },
                    },
                    "files": {
                        "data/split.json": {
                            "sha256": split_sha,
                            "n_records": int(len(y)),
                        }
                    },
                }
            }
        },
    )

    model = MLP(
        hidden=config.model.hidden,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    train_loader = make_loader(
        X, y, train_idx, config.train.batch_size, True, g
    )
    val_loader = make_loader(
        X, y, val_idx, config.train.batch_size, False, None
    )
    test_loader = make_loader(
        X, y, test_idx, config.train.batch_size, False, None
    )

    run_id = make_run_id()
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    final_config = {
        "experiment": config.experiment,
        "split_seed": config.split_seed,
        "seed": config.seed,
        "model": {
            "n_layers": config.model.n_layers,
            "hidden": config.model.hidden,
            "dropout": config.model.dropout,
        },
        "train": {
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
            "batch_size": config.train.batch_size,
            "epochs": config.train.epochs,
        },
        "runtime": {
            "device": config.runtime.device,
            "num_threads": config.runtime.num_threads,
        },
    }
    write_yaml(run_dir / "config.yaml", final_config)

    run_info = RunInfo(
        run_id=run_id,
        lab=config.experiment,
        series=series,
        status="running",
        started_at=utc_now_iso(),
        ended_at=None,
        commit=commit,
        versions=library_versions(
            {
                "torch": torch,
                "sklearn": sklearn,
                "optuna": optuna,
                "mlflow": mlflow,
            }
        ),
        device=config.runtime.device,
        num_threads=config.runtime.num_threads,
        seed=config.seed,
        data_sha256={"data/split.json": split_sha},
        model=config.model,
        error=None,
    )
    write_json(run_dir / "metadata.json", run_info)

    history_path = run_dir / "history.csv"
    predictions_path = run_dir / "predictions.csv"
    with history_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_acc", "epoch_time_s"]
        )

    setup_mlflow(config.experiment, root=ROOT)
    config_params = {
        "n_layers": config.model.n_layers,
        "hidden": config.model.hidden,
        "dropout": config.model.dropout,
        "lr": config.train.lr,
        "weight_decay": config.train.weight_decay,
        "batch_size": config.train.batch_size,
        "epochs": config.train.epochs,
    }

    try:
        with mlflow_run(
            run_id,
            params=config_params,
            tags={
                "commit": commit,
                "series": series,
                "seed": config.seed,
            },
        ):
            for epoch in range(1, config.train.epochs + 1):
                epoch_t0 = time.perf_counter()
                model.train()
                running_loss = 0.0
                n_seen = 0
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * len(yb)
                    n_seen += len(yb)
                train_loss = running_loss / n_seen
                val_loss, val_acc, _, _ = evaluate(
                    model, val_loader, criterion, device
                )
                epoch_time_s = time.perf_counter() - epoch_t0

                with history_path.open("a", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            epoch,
                            fmt_float(train_loss),
                            fmt_float(val_loss),
                            fmt_float(val_acc),
                            fmt_float(epoch_time_s),
                        ]
                    )
                log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch,
                )
                print(
                    f"epoch={epoch:02d} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

            _, _, y_true_val, y_pred_val = evaluate(
                model, val_loader, criterion, device
            )
            with predictions_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["split", "index", "y_true", "y_pred"])
                for i, yt, yp in zip(val_idx.tolist(), y_true_val, y_pred_val):
                    w.writerow(["val", i, yt, yp])
                if eval_test:
                    _, _, y_true_test, y_pred_test = evaluate(
                        model, test_loader, criterion, device
                    )
                    for i, yt, yp in zip(
                        test_idx.tolist(), y_true_test, y_pred_test
                    ):
                        w.writerow(["test", i, yt, yp])

            hist_rows = list(csv.DictReader(history_path.open(encoding="utf-8")))
            pred_rows = list(
                csv.DictReader(predictions_path.open(encoding="utf-8"))
            )
            val_rows = [r for r in pred_rows if r["split"] == "val"]
            metrics = {
                "val_acc": sum(int(r["y_true"] == r["y_pred"]) for r in val_rows)
                / len(val_rows),
                "val_loss": float(hist_rows[-1]["val_loss"]),
                "train_time_s": sum(float(r["epoch_time_s"]) for r in hist_rows),
            }
            if eval_test:
                test_rows = [r for r in pred_rows if r["split"] == "test"]
                metrics["test_acc"] = sum(
                    int(r["y_true"] == r["y_pred"]) for r in test_rows
                ) / len(test_rows)

            write_json(run_dir / "metrics.json", metrics)
            log_metrics(
                {
                    k: v
                    for k, v in metrics.items()
                    if k in {"val_acc", "val_loss", "train_time_s", "test_acc"}
                }
            )
            run_info.status = "completed"
            run_info.ended_at = utc_now_iso()
            write_json(run_dir / "metadata.json", run_info)
            log_artifacts(run_dir)
            print(f"run_dir={run_dir}")
            print(f"metrics={metrics}")
    except Exception as exc:
        run_info.status = "failed"
        run_info.ended_at = utc_now_iso()
        run_info.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "metadata.json", run_info)
        try:
            setup_mlflow(config.experiment, root=ROOT)
            log_artifacts(run_dir)
        except Exception:
            pass
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
