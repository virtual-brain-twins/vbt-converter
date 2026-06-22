import pathlib
import zipfile


def zip_directory_stream(
        src_root: pathlib.Path,
        dst_zip: pathlib.Path,
        *,
        compression: int = zipfile.ZIP_DEFLATED,
        chunk_size: int = 2 * 1024 * 1024,  # 2 MiB
) -> None:
    """
    Create ``dst_zip`` from ``src_root`` without loading full files into RAM.
    Each file is read and written in ``chunk_size`` blocks.
    """
    with zipfile.ZipFile(dst_zip, mode="w", compression=compression) as zf:
        for file_path in src_root.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(src_root)
                # ``ZipInfo`` allows us to set the external attributes (permissions)
                zipinfo = zipfile.ZipInfo(str(rel))
                zipinfo.external_attr = (file_path.stat().st_mode & 0o777) << 16
                with file_path.open("rb") as src_f:
                    # zipfile.write() would read the whole file; we stream instead.
                    with zf.open(zipinfo, "w") as dest_f:
                        while True:
                            chunk = src_f.read(chunk_size)
                            if not chunk:
                                break
                            dest_f.write(chunk)
