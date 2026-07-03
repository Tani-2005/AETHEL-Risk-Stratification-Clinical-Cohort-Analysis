"""
structured_logger.py
===================
Sets up structured logging, creating file and console handlers, separating levels,
and routing logs to experiment-specific logs directory.
"""
from __future__ import annotations
import logging
from pathlib import Path

def configure_structured_logging(log_dir: Path | None = None) -> logging.Logger:
    """
    Configures logging with a console handler and a file handler.
    If log_dir is provided, writes logs to log_dir/experiment.log.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 1. Console Handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. File Handler (DEBUG and above)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def get_logger(name: str) -> logging.Logger:
    """Wrapper around logging.getLogger to return a named logger."""
    return logging.getLogger(name)
