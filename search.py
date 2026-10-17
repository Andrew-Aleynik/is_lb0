from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import mlflow
import optuna
from optuna.study import MaxTrialsCallback

from run_utils import git_commit_and_dirty, read_yaml, setup_mlflow, write_json, write_yaml
from train import ROOT, train_epochs

STUDY_NAME = "lr0_search"
N_TRIALS = 40
TRAIN_SEED = 0
EPOCHS = 30
SPLIT_SEED = 0


def suggest_from_space(trial: optuna.Trial, space: dict) -> dict:
    params: dict = {}
    for name, spec in space.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif spec.get("type") == "int" or name in {"n_layers", "hidden", "batch_size"}:
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        else:
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ЛР0: поиск гиперпараметров (Optuna)")
    parser.add_argument(
        "--space",
        type=Path,
        default=ROOT / "configs" / "search_space.yaml",
        help="пространство поиска",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help="целевое число trial в любом состоянии",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="отладочный запуск при незакоммиченных изменениях",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
    )
    args = parser.parse_args(argv)

    space_path = args.space if args.space.is_absolute() else ROOT / args.space
    if not space_path.is_file():
        raise SystemExit(f"Нет пространства поиска: {space_path}")

    git_commit_and_dirty(args.allow_dirty, cwd=ROOT)
    space = read_yaml(space_path)
    if not isinstance(space, dict):
        raise SystemExit(f"search space должен быть mapping: {space_path}")

    setup_mlflow("lr0", root=ROOT)
    storage = f"sqlite:///{(ROOT / 'optuna.db').resolve()}"

    def objective(trial: optuna.Trial) -> float:
        # Как в приложении ЛР0: params trial, val_acc по эпохам, тег state
        params = suggest_from_space(trial, space)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            val_acc = float("nan")
            for epoch, val_acc in train_epochs(
                params,
                seed=TRAIN_SEED,
                epochs=EPOCHS,
                split_seed=SPLIT_SEED,
                device=args.device,
                num_threads=args.num_threads,
            ):
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                trial.report(val_acc, step=epoch)
                if trial.should_prune():
                    mlflow.set_tag("state", "PRUNED")
                    raise optuna.TrialPruned()
            mlflow.set_tag("state", "FAIL" if math.isnan(val_acc) else "COMPLETE")
            return val_acc

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    with mlflow.start_run(run_name=STUDY_NAME):
        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[MaxTrialsCallback(args.n_trials, states=None)],
        )

    # сводка для отчёта
    states = Counter(t.state.name for t in study.trials)
    epochs_budget = args.n_trials * EPOCHS
    epochs_done = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            epochs_done += EPOCHS
        elif t.state == optuna.trial.TrialState.PRUNED:
            # последний reported step
            if t.intermediate_values:
                epochs_done += max(t.intermediate_values.keys())
    epochs_saved = epochs_budget - epochs_done

    best = study.best_trial
    best_params = dict(best.params)
    best_config = {
        "experiment": "lr0",
        "split_seed": SPLIT_SEED,
        "seed": TRAIN_SEED,
        "model": {
            "n_layers": int(best_params["n_layers"]),
            "hidden": int(best_params["hidden"]),
            "dropout": float(best_params["dropout"]),
        },
        "train": {
            "lr": float(best_params["lr"]),
            "weight_decay": float(best_params["weight_decay"]),
            "batch_size": int(best_params["batch_size"]),
            "epochs": EPOCHS,
        },
        "runtime": {"device": args.device, "num_threads": args.num_threads},
    }
    best_path = ROOT / "configs" / "best.yaml"
    write_yaml(best_path, best_config)

    summary = {
        "study_name": STUDY_NAME,
        "n_trials": len(study.trials),
        "states": dict(states),
        "epochs_budget": epochs_budget,
        "epochs_done": epochs_done,
        "epochs_saved_by_pruner": epochs_saved,
        "best_trial": best.number,
        "best_val_acc": best.value,
        "best_params": best_params,
        "best_config_path": str(best_path.relative_to(ROOT)),
        "importances": optuna.importance.get_param_importances(study),
    }
    summary_path = ROOT / "artifacts" / "search_summary.json"
    write_json(summary_path, summary)

    print(f"trials={dict(states)}")
    print(f"epochs_done={epochs_done}/{epochs_budget} saved={epochs_saved}")
    print(f"best_trial={best.number} best_val_acc={best.value}")
    print(f"best_config={best_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
