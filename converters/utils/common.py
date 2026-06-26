from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

__all__ = ["configure_logging"]


def _make_log_path(
        env_var: str = "VBT_JOB_LOGS",
        default_dir: str | os.PathLike = ".",
        log_file_name: str = "default_name.log",
) -> Path:
    """
    Resolve the directory in which the log file will be created.

    Parameters
    ----------
    env_var : str, optional
        Name of the environment variable that may contain a custom log directory.
    default_dir : str | Path, optional
        Directory to fall back to when the env‑var is not set.
    log_file_name : str, optional
        Name of the log file (placed inside the resolved directory).

    Returns
    -------
    Path
        Fully‑qualified path to the log file.
    """
    log_dir = Path(os.getenv(env_var, default_dir)).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / log_file_name


def configure_logging(
        module_name: str,
        *,
        level: int = logging.INFO,
        env_var: str = "VBT_JOB_LOGS",
        default_dir: str | os.PathLike = ".",
        log_file_name: str = "convert_direc_to_hdf5.log",
        fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
        stream: Optional[object] = sys.stderr,
) -> logging.Logger:
    """
    Set up a consistent logging configuration and return a module‑specific logger.

    The function creates (once per interpreter session) a ``FileHandler`` that
    writes to ``<log_dir>/<log_file_name>`` and a ``StreamHandler`` that writes
    to ``stderr`` (or another stream you provide).  Subsequent calls with the
    same arguments will **not** add duplicate handlers.

    Parameters
    ----------
    module_name : str
        ``__name__`` of the calling module – the returned logger will be a child
        of the root logger with this name.
    level : int, optional
        Minimum severity level to handle (defaults to :data:`logging.INFO`).
    env_var : str, optional
        Environment variable that may point at a custom log directory.
    default_dir : str | Path, optional
        Fallback directory when ``env_var`` is not set.
    log_file_name : str, optional
        Name of the log file that will live inside the resolved directory.
    fmt : str, optional
        Logging format string (passed to :class:`logging.Formatter`).
    stream : file‑like, optional
        Stream for the console handler; defaults to ``sys.stderr``.

    Returns
    -------
    logging.Logger
        The logger bound to ``module_name`` ready for use.

    Example
    -------
    >> from utils import configure_logging
    >> logger = configure_logging(__name__)      # in any module
    >> logger.debug("debug message")             # will appear in file + stderr
    """
    # Build the (singleton) root handlers only once.
    root_logger = logging.getLogger()
    if not root_logger.handlers:  # first call only
        # Formatter shared by both handlers
        formatter = logging.Formatter(fmt)

        #  File handler
        log_path = _make_log_path(env_var, default_dir, log_file_name)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Stream (stderr) handler
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        # Global configuration
        root_logger.setLevel(level)

    # Return a child logger for the calling module.
    logger = logging.getLogger(module_name)
    # Ensure the module logger respects the level we passed (it may be
    # ``WARNING`` by default if the user over‑rode it elsewhere).
    logger.setLevel(level)
    return logger
