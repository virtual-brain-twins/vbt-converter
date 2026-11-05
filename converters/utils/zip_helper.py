# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

"""Utility for creating ZIP archives of directory trees.

This module selects an appropriate compression method at import time:
* If the running interpreter supports Zstandard (Python ≥ 3.14) **and**
  the optional ``zstandard`` library is importable, ZSTD compression is used
  with a default level of 15.
* Otherwise DEFLATE compression (the classic ZIP format) is used with a
  default level of 6.

The primary public entry point is :func:`zip_directory`, which recursively
adds all files under a source directory to a destination ``.zip`` file.
"""
from __future__ import annotations

import os
import pathlib
import zipfile
from typing import Final


def _zstd_available() -> bool:
    """
    Return ``True`` if the optional ``compression.zstd`` module can be imported.

    The Zstandard compression support in the standard library lives under
    ``zipfile.compression`` as a private import.  Importing that submodule
    will raise an :class:`ImportError` (or another exception) if the third‑party
    ``zstandard`` wheel or compiled extension is missing.  This helper
    encapsulates that check so the rest of the module can decide whether
    ZSTD is usable at import time.

    Returns
    -------
    bool
        ``True`` if the ``zstandard`` runtime is available, ``False`` otherwise.
    """
    try:
        # The module lives under ``zipfile`` as a private import.  Importing it
        # directly will raise ImportError if the third‑party ``zstandard`` wheel
        # (or compiled extension) is missing.
        from zipfile import compression  # noqa: F401  (import for side‑effect)
        return True
    except Exception:  # pragma: no cover – any error means “no ZSTD”
        return False


# Public constants – decide defaults once at import time.
if hasattr(zipfile, "ZIP_ZSTANDARD") and _zstd_available():
    # zstandard is both present as a flag and the runtime module is importable.
    DEFAULT_COMPRESSION_METHOD: Final[int] = zipfile.ZIP_ZSTANDARD
    DEFAULT_COMPRESSION_LEVEL: Final[int] = 15  # ZSTD level range: 1‑22
else:
    # Either the flag does not exist (Python < 3.14) or the optional module is missing.
    DEFAULT_COMPRESSION_METHOD: Final[int] = zipfile.ZIP_DEFLATED
    DEFAULT_COMPRESSION_LEVEL: Final[int] = 6  # DEFLATE default (1‑9)


def zip_directory(
        src_root: pathlib.Path,
        dst_zip: pathlib.Path,
        *,
        compress_level: int = DEFAULT_COMPRESSION_LEVEL,
        compression: int = DEFAULT_COMPRESSION_METHOD
):
    """
    Recursively zip the contents of ``src_root`` into ``dst_zip`` using the
    selected compression method and level.

    Parameters
    ----------
    src_root: pathlib.Path
        The root directory whose *files* (not directories) will be added to the
        archive.  All files under this path are included, preserving their
        relative paths.
    dst_zip: pathlib.Path
        Destination path for the created ``.zip`` file.  If the file already
        exists it will be overwritten.
    compress_level: int, optional
        Compression level passed to :class:`zipfile.ZipFile`.  The meaning of
        the value depends on the chosen compression method (e.g., 1‑9 for
        DEFLATE, 1‑22 for ZSTD).  Defaults to the module‑level
        ``DEFAULT_COMPRESSION_LEVEL``.
    compression: int, optional
        The compression method flag (e.g., ``zipfile.ZIP_DEFLATED`` or
        ``zipfile.ZIP_ZSTANDARD``).  Defaults to the module‑level
        ``DEFAULT_COMPRESSION_METHOD``.

    Notes
    -----
    * Only regular files are added; empty directories are not stored explicitly.
    * The function uses ``allowZip64=True`` to support archives larger than
      4 GiB.
    * The ``arcname`` for each entry is the file’s path relative to ``src_root``,
      ensuring the directory structure is reproduced when the archive is
      extracted.

    Example
    -------
    >>> from pathlib import Path
    >>> zip_directory(Path("my_folder"), Path("my_folder.zip"))
    """
    files = [p for p in src_root.rglob("*") if p.is_file()]
    os.makedirs(dst_zip.parent, exist_ok=True)
    with zipfile.ZipFile(
            dst_zip,
            mode="w",
            compression=compression,
            compresslevel=compress_level,
            allowZip64=True,
    ) as zf:
        for path in files:
            zf.write(
                filename=path,
                arcname=path.relative_to(src_root)
            )
