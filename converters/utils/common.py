# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

from __future__ import annotations

import json
import logging
import os
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Tuple, Any

__all__ = ["configure_logging"]

import h5py

from converters.utils.compound_attribute_helper import read_compound_attr
from converters.utils.units import Unit


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


def read_compound_attr_parsed(
        parent: h5py.Group,
        name: str,
) -> Tuple[Any, Optional[Unit]]:
    """
    Read a compound attribute and **automatically JSON‑decode** its ``value`` field
    when the stored data is a valid JSON document.

    Parameters
    ----------
    parent:
        The HDF5 group (or file) that owns the attribute.
    name:
        The attribute name.

    Returns
    -------
    (value, unit):
        * ``value`` – either the native scalar stored in the attribute **or** the
          Python object obtained from ``json.loads`` when the string can be parsed.
        * ``unit`` – a :class:`~converters.utils.units.Unit` instance or ``None``.

    Notes
    -----
    * No exception propagates from the JSON decoder – any problem simply leaves
      ``value`` untouched.
    * The function does **not** perform any further type conversion; the caller
      receives exactly what was stored (or its JSON representation).
    """
    # First, use the original implementation to obtain the raw data.
    raw_value, unit = read_compound_attr(parent, name)

    # Is the value a string‑like object that could contain JSON?
    if isinstance(raw_value, (str, bytes, bytearray)):
        # Ensure we are dealing with a Python ``str`` – ``read_compound_attr`` may
        # already have decoded bytes, but we guard against the unlikely case it
        # didn't (some h5py versions on Windows can return ``bytes``).
        if isinstance(raw_value, (bytes, bytearray)):
            raw_value = raw_value.decode("utf-8")

        # Try to decode JSON – any failure is silently ignored.
        try:
            parsed = json.loads(raw_value)
        except JSONDecodeError:
            # Not a JSON document → keep the original string.
            parsed = raw_value
    else:
        # The value is already a native scalar (int, float, bool, …) – nothing to do.
        parsed = raw_value
    return parsed, unit
