# ЛР0. Воспроизводимый эксперимент

## Структура

| Путь | Назначение |
|------|------------|
| `train.py` | один запуск обучения |
| `search.py` | поиск гиперпараметров (Optuna) |
| `run_utils.py` | общие утилиты запусков |
| `configs/base.yaml` | базовая конфигурация |
| `configs/search_space.yaml` | пространство поиска Optuna |
| `configs/best.yaml` | лучшая конфигурация (после поиска) |
| `data/split.json` | индексы train/val/test |
| `data/manifest.json` | манифест данных |
| `forecast.md` | прогноз до экспериментов |
| `report.md` | отчёт |
| `runs/`, `mlflow.db`, `optuna.db` | артефакты (в `.gitignore`) |

## Окружение

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Запуск

```bash
python train.py --config configs/base.yaml
python train.py --config configs/base.yaml --seed 1
python train.py --config configs/base.yaml --seed 101 --eval-test
```

CLI переопределяет значения из файла; в `runs/<run_id>/config.yaml` сохраняется итог.  
Для отладки при незакоммиченных изменениях: `--allow-dirty`.

## Поиск гиперпараметров

```bash
python search.py --n-trials 40 --allow-dirty
```

Study: `lr0_search` в `optuna.db`, родительский run MLflow `lr0_search`, trial — nested.  
Лучшая конфигурация → `configs/best.yaml`, сводка → `artifacts/search_summary.json`.
