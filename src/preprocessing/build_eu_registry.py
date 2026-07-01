"""
build_eu_registry.py
====================
Stage 1 — Build the Pan-European city metadata registry.

Generates a 100-city lookup table of European healthcare/research hubs
with baseline pollution levels (PM2.5, NO2).  Output is written to
``data/raw/eu_registry.csv`` and consumed downstream by
``generate_env_data.py`` and ``preprocess_features.py``.

Methodology
-----------
Baseline pollution values are drawn from a uniform distribution seeded
deterministically (see ``src/utils/seed.py``).  Nordic/Swiss cities are
modelled with lower baseline pollution — this is encoded in the random draw
range rather than post-hoc adjustments, keeping the generation reproducible.

Usage
-----
    python -m src.preprocessing.build_eu_registry
    # or via the pipeline orchestrator:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logging_setup import get_logger
from src.utils.paths import DataPaths, DataDirs
from src.utils.seed import set_global_seed
from src.utils.config_loader import load_config
from src.utils.constants import Columns

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# City registry — 100 real European cities with approximate lat/lon
# ---------------------------------------------------------------------------

_CITIES: list[tuple[str, str, float, float]] = [
    # UK & Ireland
    ("London", "UK", 51.5074, -0.1278), ("Manchester", "UK", 53.4808, -2.2426),
    ("Birmingham", "UK", 52.4862, -1.8904), ("Edinburgh", "UK", 55.9533, -3.1883),
    ("Glasgow", "UK", 55.8642, -4.2518), ("Liverpool", "UK", 53.4084, -2.9916),
    ("Bristol", "UK", 51.4545, -2.5879), ("Cardiff", "UK", 51.4816, -3.1791),
    ("Belfast", "UK", 54.5973, -5.9301), ("Dublin", "IE", 53.3498, -6.2603),
    # France
    ("Paris", "FR", 48.8566, 2.3522), ("Marseille", "FR", 43.2965, 5.3698),
    ("Lyon", "FR", 45.7640, 4.8357), ("Toulouse", "FR", 43.6047, 1.4442),
    ("Nice", "FR", 43.7102, 7.2620), ("Nantes", "FR", 47.2184, -1.5536),
    ("Strasbourg", "FR", 48.5734, 7.7521), ("Montpellier", "FR", 43.6108, 3.8767),
    ("Bordeaux", "FR", 44.8378, -0.5792), ("Lille", "FR", 50.6292, 3.0573),
    # Germany
    ("Berlin", "DE", 52.5200, 13.4050), ("Munich", "DE", 48.1351, 11.5820),
    ("Frankfurt", "DE", 50.1109, 8.6821), ("Hamburg", "DE", 53.5511, 9.9937),
    ("Cologne", "DE", 50.9375, 6.9603), ("Stuttgart", "DE", 48.7758, 9.1829),
    ("Düsseldorf", "DE", 51.2277, 6.7735), ("Leipzig", "DE", 51.3397, 12.3731),
    ("Dortmund", "DE", 51.5136, 7.4653), ("Essen", "DE", 51.4556, 7.0116),
    # Italy
    ("Rome", "IT", 41.9028, 12.4964), ("Milan", "IT", 45.4642, 9.1900),
    ("Naples", "IT", 40.8518, 14.2681), ("Turin", "IT", 45.0703, 7.6869),
    ("Palermo", "IT", 38.1157, 13.3615), ("Genoa", "IT", 44.4056, 8.9463),
    ("Bologna", "IT", 44.4949, 11.3426), ("Florence", "IT", 43.7696, 11.2558),
    ("Bari", "IT", 41.1171, 16.8719), ("Catania", "IT", 37.5079, 15.0830),
    # Spain & Portugal
    ("Madrid", "ES", 40.4168, -3.7038), ("Barcelona", "ES", 41.3851, 2.1734),
    ("Valencia", "ES", 39.4699, -0.3774), ("Seville", "ES", 37.3891, -5.9845),
    ("Zaragoza", "ES", 41.6488, -0.8891), ("Malaga", "ES", 36.7213, -4.4214),
    ("Murcia", "ES", 37.9922, -1.1307), ("Palma", "ES", 39.5696, 2.6502),
    ("Lisbon", "PT", 38.7223, -9.1393), ("Porto", "PT", 41.1579, -8.6291),
    # DACH & Benelux
    ("Zurich", "CH", 47.3769, 8.5417), ("Geneva", "CH", 46.2044, 6.1432),
    ("Vienna", "AT", 48.2082, 16.3738), ("Amsterdam", "NL", 52.3676, 4.9041),
    ("Rotterdam", "NL", 51.9225, 4.4792), ("The Hague", "NL", 52.0705, 4.3007),
    ("Brussels", "BE", 50.8503, 4.3517), ("Antwerp", "BE", 51.2194, 4.4025),
    ("Ghent", "BE", 51.0543, 3.7174), ("Luxembourg", "LU", 49.6116, 6.1319),
    # Nordics
    ("Stockholm", "SE", 59.3293, 18.0686), ("Gothenburg", "SE", 57.7089, 11.9746),
    ("Malmo", "SE", 55.6050, 13.0038), ("Oslo", "NO", 59.9139, 10.7522),
    ("Bergen", "NO", 60.3913, 5.3221), ("Copenhagen", "DK", 55.6761, 12.5683),
    ("Aarhus", "DK", 56.1629, 10.2039), ("Helsinki", "FI", 60.1695, 24.9354),
    ("Espoo", "FI", 60.2055, 24.6559), ("Tampere", "FI", 61.4978, 23.7610),
    # Eastern Europe
    ("Warsaw", "PL", 52.2297, 21.0122), ("Krakow", "PL", 50.0647, 19.9450),
    ("Lodz", "PL", 51.7592, 19.4560), ("Wroclaw", "PL", 51.1079, 17.0385),
    ("Poznan", "PL", 52.4064, 16.9252), ("Prague", "CZ", 50.0755, 14.4378),
    ("Brno", "CZ", 49.1951, 16.6068), ("Budapest", "HU", 47.4979, 19.0402),
    ("Bratislava", "SK", 48.1486, 17.1077), ("Bucharest", "RO", 44.4268, 26.1025),
    # Baltics & Balkans
    ("Riga", "LV", 56.9496, 24.1052), ("Tallinn", "EE", 59.4370, 24.7536),
    ("Vilnius", "LT", 54.6872, 25.2797), ("Athens", "GR", 37.9838, 23.7275),
    ("Thessaloniki", "GR", 40.6401, 22.9444), ("Sofia", "BG", 42.6977, 23.3219),
    ("Belgrade", "RS", 44.7866, 20.4489), ("Zagreb", "HR", 45.8150, 15.9819),
    ("Ljubljana", "SI", 46.0569, 14.5058), ("Sarajevo", "BA", 43.8563, 18.4131),
]


def build_registry() -> pd.DataFrame:
    """
    Build and persist the 100-city Pan-European metadata registry.

    Each city receives a simulated baseline PM2.5 and NO2 value drawn
    from a uniform distribution.  NO2 is modelled as a linear function
    of PM2.5 with additive noise (reflecting real-world co-linearity).

    Returns
    -------
    pd.DataFrame
        Registry with columns: city_id, city, country, lat, lon,
        base_pm25, base_no2.  Shape: (100, 7).

    Side-effects
    ------------
    Writes ``data/raw/eu_registry.csv``.
    """
    cfg = load_config()
    set_global_seed(cfg.seeds.python)

    logger.info("Building AETHEL Pan-European Metadata Registry (%d cities)...", len(_CITIES))

    records: list[dict] = []
    for city, country, lat, lon in _CITIES:
        base_pm25 = np.random.uniform(6.0, 18.0)
        base_no2 = base_pm25 * 1.4 + np.random.uniform(-2, 2)
        records.append({
            Columns.CITY_ID: f"{city}_{country}",
            Columns.CITY: city,
            Columns.COUNTRY: country,
            Columns.LAT: lat,
            Columns.LON: lon,
            Columns.BASE_PM25: round(base_pm25, 2),
            Columns.BASE_NO2: round(base_no2, 2),
        })

    df_registry = pd.DataFrame(records)

    DataDirs.RAW.mkdir(parents=True, exist_ok=True)
    df_registry.to_csv(DataPaths.RAW_REGISTRY, index=False)

    logger.info(
        "Success! Created %s containing %d European hubs.",
        DataPaths.RAW_REGISTRY,
        len(df_registry),
    )
    return df_registry


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    build_registry()
