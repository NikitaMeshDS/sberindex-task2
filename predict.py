"""Run configurable forecasts without restoring the research archive."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sberindex.forecasting.predict import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"Ошибка прогноза: {error}") from None
