"""
logging_setup.py
================
Centralised logging configuration for the AETHEL pipeline.

Replaces ad-hoc ``print()`` calls throughout the codebase with a
structured ``logging.Logger``.  Every module acquires a logger via::

    logger = get_logger(__name__)

A single ``configure_logging()`` call at pipeline entry sets up:

- Console handler  — INFO level, coloured where supported
- File handler     — DEBUG level, written to ``outputs/logs/aethel.log``

Log format
----------
``2026-01-15 14:23:01 | INFO     | src.preprocessing.build_eu_registry | Building AETHEL registry...``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False  # guard against duplicate handler registration


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    log_dir: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure the root logger for the AETHEL pipeline.

    Call this **once** at the start of any entry-point script
    (e.g. ``scripts/run_pipeline.py`` or individual stage scripts).

    Parameters
    ----------
    log_dir:
        Directory where ``aethel.log`` will be written.
        Defaults to ``outputs/logs/`` resolved via ``paths.OutputDirs``.
    console_level:
        Minimum severity shown on stdout (default: INFO).
    file_level:
        Minimum severity written to the log file (default: DEBUG).
    """
    global _CONFIGURED  # noqa: PLW0603

    if _CONFIGURED:
        return

    # Resolve log directory
    if log_dir is None:
        from src.utils.paths import OutputDirs  # lazy import to avoid circulars
        log_dir = OutputDirs.LOGS

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "aethel.log"

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # --- File handler ---
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # --- Root logger ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # handlers filter, root is permissive
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True

    root_logger.info("Logging initialised. Log file: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for the calling module.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger

    Example
    -------
        logger = get_logger(__name__)
        logger.info("Processing %d records", n)
    """
    return logging.getLogger(name)
