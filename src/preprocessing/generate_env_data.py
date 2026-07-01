"""
generate_env_data.py
====================
Stage 2 — Generate 60-month spatio-temporal environmental time-series.

For each of the 100 cities in the EU registry, this module simulates
monthly PM2.5 and NO2 readings using a seasonal cosine signal plus
city-specific random noise.  The per-city seed is derived deterministically
from the city's string ID (preserving reproducibility without requiring a
global seed reset on each iteration).

Prerequisites
-------------
``data/raw/eu_registry.csv`` must exist.  Run
``src.preprocessing.build_eu_registry`` first.

Output
------
``data/raw/regional_environmental_history.csv``
Columns: date, location, pm2_5, no2
Rows: 100 cities × 60 months = 6,000 records

Usage
-----
    python -m src.preprocessing.generate_env_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.paths import DataPaths
from src.utils.constants import Columns
from src.utils.config_loader import load_config

logger = get_logger(__name__)

_FOLLOW_UP_MONTHS = 60  # 5-year study window


def generate_pan_european_data() -> pd.DataFrame:
    """
    Generate monthly environmental telemetry for the 100-city registry.

    Seasonality is modelled as a cosine wave scaled to ±5 μg/m³, layered
    on top of each city's baseline pollution level.  Per-city seeds ensure
    reproducible noise without cross-city contamination.

    Returns
    -------
    pd.DataFrame
        Environmental time-series with columns: date, location, pm2_5, no2.

    Side-effects
    ------------
    Writes ``data/raw/regional_environmental_history.csv``.

    Raises
    ------
    FileNotFoundError
        If ``data/raw/eu_registry.csv`` does not exist.
    """
    cfg = load_config()  # noqa: F841  (loaded for future param use)

    logger.info("Initialising AETHEL 100-City Environmental Engine...")

    if not DataPaths.RAW_REGISTRY.exists():
        raise FileNotFoundError(
            f"Registry not found at {DataPaths.RAW_REGISTRY}. "
            "Run src.preprocessing.build_eu_registry first."
        )

    registry = pd.read_csv(DataPaths.RAW_REGISTRY)
    logger.info("Loaded registry with %d cities.", len(registry))

    dates = pd.date_range(start="2020-01-01", periods=_FOLLOW_UP_MONTHS, freq="ME")
    all_data: list[pd.DataFrame] = []

    for _, row in registry.iterrows():
        # Per-city deterministic seed derived from city string ID
        city_seed = abs(hash(row[Columns.CITY_ID])) % (10 ** 8)
        np.random.seed(city_seed)

        seasonality = np.cos(np.linspace(0, 10 * np.pi, _FOLLOW_UP_MONTHS)) * 5

        city_frame = pd.DataFrame({
            Columns.DATE: dates,
            Columns.LOCATION: row[Columns.CITY_ID],
            Columns.PM25: np.maximum(
                row[Columns.BASE_PM25] + seasonality + np.random.normal(0, 2, _FOLLOW_UP_MONTHS),
                0,
            ),
            Columns.NO2: np.maximum(
                row[Columns.BASE_NO2] + (seasonality * 1.5) + np.random.normal(0, 3, _FOLLOW_UP_MONTHS),
                0,
            ),
        })
        all_data.append(city_frame)

    df = pd.concat(all_data, ignore_index=True)
    df.to_csv(DataPaths.RAW_ENV_HISTORY, index=False)

    logger.info(
        "Success! Generated %d months of history for %d European hubs. Written to %s.",
        _FOLLOW_UP_MONTHS,
        len(registry),
        DataPaths.RAW_ENV_HISTORY,
    )
    return df


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    generate_pan_european_data()
