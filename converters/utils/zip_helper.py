from __future__ import annotations

import pathlib
import zipfile
from typing import Final

# Public constants
def _zstd_available() -> bool:
    """
    Return ``True`` if the optional ``compression.zstd`` module can be imported.
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
    DEFAULT_COMPRESSION_LEVEL: Final[int] = 15          # ZSTD level range: 1‑22
else:
    # Either the flag does not exist (Python < 3.14) or the optional module is missing.
    DEFAULT_COMPRESSION_METHOD: Final[int] = zipfile.ZIP_DEFLATED
    DEFAULT_COMPRESSION_LEVEL: Final[int] = 6           # DEFLATE default (1‑9)


def zip_directory(
        src_root: pathlib.Path,
        dst_zip: pathlib.Path,
        *,
        compress_level: int = DEFAULT_COMPRESSION_LEVEL,
        compression: int = DEFAULT_COMPRESSION_METHOD
):
    files = [p for p in src_root.rglob("*") if p.is_file()]
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
