"""
seed.py
=======
Deterministic seed management for the AETHEL pipeline.

A single ``set_global_seed()`` call initialises all Python-level
pseudo-random number generators.  The matching R seed is kept in
``configs/default.yaml`` under ``seeds.r`` and consumed by R scripts
via the ``yaml`` package.

Rationale
---------
The original codebase had ``np.random.seed(42)`` scattered across three
Python files and ``set.seed(123)`` in one R file — inconsistent seeds
that make full cross-language reproducibility impossible to audit.

This module establishes a single, config-driven contract:

    seeds:
      python: 42   # used here
      r: 123       # used in src/models/survival_model.R

Usage
-----
    from src.utils.seed import set_global_seed

    set_global_seed(42)                   # direct call
    # or, from config:
    set_global_seed(cfg.seeds.python)
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Initialise all Python-level RNG states to a deterministic seed.

    Affects
    -------
    - ``random`` standard library module
    - ``numpy`` random generator

    Parameters
    ----------
    seed:
        Integer seed value.  Should be sourced from
        ``configs/default.yaml → seeds.python``.

    Notes
    -----
    This function does **not** affect R's RNG.  The R seed is set via
    ``set.seed()`` inside R scripts, reading the value from
    ``configs/default.yaml → seeds.r``.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.debug("Global Python random seed set to %d", seed)
