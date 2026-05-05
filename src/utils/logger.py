"""
logger.py
=========
Centralised logging for the leaching ML pipeline.

Every module imports this instead of calling logging.basicConfig()
directly, so all log output (file + console) is consistent across
the entire project.

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Data loaded successfully")
"""

import os
import logging
from datetime import datetime


# ── Paths ─────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "logs"
)
os.makedirs(_LOG_DIR, exist_ok=True)

_LOG_FILE = os.path.join(
    _LOG_DIR,
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

# ── Root handler setup (runs once at import time) ─────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_FILE),   # write to file
        logging.StreamHandler(),          # also print to console
    ],
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that inherits the root configuration.

    Parameters
    ----------
    name : str
        Typically ``__name__`` from the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
