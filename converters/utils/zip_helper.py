from __future__ import annotations

import pathlib
import zipfile
from typing import Final

# Public constants
DEFAULT_COMPRESSION_METHOD: Final[int] = getattr(zipfile, 'ZIP_ZSTANDARD', zipfile.ZIP_DEFLATED)
DEFAULT_COMPRESSION_LEVEL: Final[int] = 15 if DEFAULT_COMPRESSION_METHOD == getattr(zipfile, 'ZIP_ZSTANDARD',
                                                                                    None) else 6


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
