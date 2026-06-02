#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a single HDF5 “master” file that references all per‑simulation files
as external links.  The script copies every external file into a directory
tree (`bundle/`) that lives next to the master file, preserving the original
layout (network_state → files).

Requirements
------------
* Python ≥ 3.8
* h5py  ≥ 3.0
* numpy
* tqdm   (optional, for a pretty progress bar)

Run
----
    python make_master_h5.py <metadata‑csv> <output‑bundle‑folder> [options]

Result
------
bundle/
    master.h5                 # the only file you open with h5py / HDFView
    d2oe_0.00/ …             # copies of the original *.h5, *.json, *.txt files
    d2oe_0.25/ …
…

The master file can be opened anywhere; all external links are relative to the
master file, so moving the whole *bundle* directory keeps the whole dataset
intact.

Logging
-------
All information, warnings and errors are written to a log file located in the
directory given by the environment variable ``VBT_JOB_LOGS``.  If the variable
is not defined, the log file is created in the current working directory.
"""

import argparse
import csv
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Iterable, Mapping

import h5py
import numpy as np

# Logging configuration
log_dir = Path(os.getenv("VBT_JOB_LOGS", ".")).expanduser().resolve()
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "convert_direc_to_hdf5.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

# Helper utilities


def _ensure_dir(path: Path) -> None:
    """Make sure a directory exists (like `mkdir -p`)."""
    path.mkdir(parents=True, exist_ok=True)


def _copy_to_bundle(src: Path, bundle_root: Path) -> Path | None:
    """
    Copy *src* (file) into the *bundle_root*, keeping the relative directory
    hierarchy intact.  Returns the new absolute path inside the bundle.
    """
    if not src.is_file():
        return None

    rel = src.relative_to(src.anchor)  # keep everything after the leading '/'
    dest = bundle_root / rel
    _ensure_dir(dest.parent)
    shutil.copy2(src, dest)  # preserve timestamps / permissions
    logger.debug("Copied %s → %s", src, dest)
    return dest.resolve()


def _parse_float_array(s: str) -> np.ndarray:
    """
    Convert a string like "[5e-10, 5e-10, …]" → 1‑D float64 array.
    """
    numbers = [float(tok) for tok in s.strip("[]").split(",")]
    return np.asarray(numbers, dtype=np.float64)


def _parse_optional_float_array(s: str) -> np.ndarray:
    """
    Convert a string like "[1.25, 1.25, None, None]" → float array where
    ``None`` becomes ``np.nan``.
    """
    tokens = s.strip("[]").split(",")
    parsed = [
        float(tok) if tok.strip().lower() != "none" else np.nan
        for tok in tokens
    ]
    return np.asarray(parsed, dtype=np.float64)


def _make_external_link(
    parent: h5py.Group,
    name: str,
    target_path: Path,
    bundle_root: Path,
) -> None:
    """
    Insert an ExternalLink *name* into *parent* that points to the root group ``/``
    of the HDF5 file *target_path*.

    The link (and the saved attribute) use a relative path that is
    expressed with respect to *bundle_root* – this keeps the bundle portable.
    """
    rel_path = relative_path(bundle_root, target_path)

    # Create the ExternalLink.  h5py accepts a **string** filename, so we
    # convert the Path to str.
    parent[name] = h5py.ExternalLink(str(rel_path), "/")

    # Store the relative path as an attribute for easy lookup later.
    parent[name].attrs["relative_path"] = str(rel_path)
    logger.debug("Created external link '%s' → %s", name, rel_path)


def relative_path(bundle_root: Path, target_path: Path) -> Path:
    """Compute a relative path from *bundle_root* to *target_path*."""
    try:
        rel_path = target_path.relative_to(bundle_root)
    except ValueError:
        rel_path = Path(
            os.path.relpath(str(target_path), start=str(bundle_root))
        )
    return rel_path


# Copy the auxiliary text / json files and store them in the HDF5.
MAX_BLOB_SIZE = 8 * 1024 * 1024  # 8MiB – files larger than this become a blob
COMPRESS_TEXT = True


def _store_auxiliary_file(
    parent: h5py.Group,
    dset_name: str,
    src_path: Path,
    bundle_root: Path,
) -> None:
    """
    Copy *src_path* into the bundle and store its contents inside *parent*
    as a compressed dataset (v‑len string or binary blob depending on size).
    """
    dst_path = _copy_to_bundle(src_path, bundle_root)
    if dst_path is None:
        raise FileNotFoundError(f"Failed to copy {src_path}")

    file_bytes = dst_path.read_bytes()
    size = len(file_bytes)

    if size <= MAX_BLOB_SIZE:
        text = file_bytes.decode("utf-8")
        _store_text_as_vlen_dataset(
            parent,
            dset_name,
            text,
            compress=COMPRESS_TEXT,
        )
    else:
        _store_blob_dataset(
            parent,
            dset_name,
            file_bytes,
            compress=COMPRESS_TEXT,
        )

    parent[dset_name].attrs["original_path"] = str(
        relative_path(bundle_root, dst_path)
    )
    logger.debug(
        "Stored auxiliary file '%s' (size=%d bytes) as dataset %s",
        src_path,
        size,
        dset_name,
    )


def _store_blob_dataset(
    parent: h5py.Group,
    name: str,
    data_bytes: bytes,
    compress: bool = True,
    compression: str = "gzip",
    compression_opts: int = 9,
    chunk_target_mb: float = 1.0,
) -> None:
    """Store *data_bytes* in a chunked uint8 dataset with optional compression."""
    arr = np.frombuffer(data_bytes, dtype=np.uint8)

    if compress:
        chunk_bytes = int(chunk_target_mb * 1024 * 1024)
        chunk_len = min(chunk_bytes, arr.size)
        chunks = (chunk_len,)
        kw = dict(
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks,
        )
    else:
        kw = {}

    dset = parent.create_dataset(name, data=arr, **kw)
    dset.attrs["is_blob"] = True
    dset.attrs["encoding"] = "utf-8"


def _store_text_as_vlen_dataset(
    parent: h5py.Group,
    name: str,
    text: str,
    compress: bool = True,
    compression: str = "gzip",
    compression_opts: int = 9,
) -> None:
    """Store *text* as a 1‑element variable‑length string dataset."""
    vlen_dtype = h5py.string_dtype(encoding="utf-8")
    arr = np.empty((1,), dtype=vlen_dtype)
    arr[0] = text

    kw = {}
    if compress:
        kw.update(compression=compression, compression_opts=compression_opts)

    dset = parent.create_dataset(name, data=arr, **kw)
    dset.attrs["is_vlen"] = True


# ----------------------------------------------------------------------
# Path handling helpers
# ----------------------------------------------------------------------
PATH_COLUMNS = {
    "network_path",
    "sim_info_file",
    "network_file",
    "input_config",
    "input_file",
    "output_file",
    "virtual_input_file",
    "virtual_input_mapping",
}

ANCHOR_FORCE_DEFAULT = {
    "input_config",
    "virtual_input_file",
    "virtual_input_mapping",
}


def _relative_path(
    abs_path: str,
    base_dir: Path,
    fallback_anchor: str | None = None,
) -> str:
    """
    Return a path that is *relative* to ``base_dir`` if possible.
    If not, keep the part of the path that starts with ``fallback_anchor``.
    """
    p = Path(abs_path)

    try:
        return str(p.relative_to(base_dir))
    except ValueError:
        pass

    if fallback_anchor and fallback_anchor in abs_path:
        _, _, tail = abs_path.partition(fallback_anchor)
        return f"{fallback_anchor}{tail}"
    return abs_path


def relative_row(
    row: MutableMapping[str, str],
    base_dir: Path,
    anchor_col: str = "network_state",
    default_anchor: str | None = None,
) -> None:
    """
    Mutate *row* in‑place: for every column listed in ``PATH_COLUMNS`` replace
    its value with a path that is relative to ``base_dir``.
    """
    row_anchor = row.get(anchor_col)
    for col in PATH_COLUMNS & row.keys():
        fallback = default_anchor if col in ANCHOR_FORCE_DEFAULT else row_anchor
        row[col] = _relative_path(row[col], base_dir, fallback_anchor=fallback)


def stream_csv_as_dicts(
    csv_path: Path,
    delimiter: str = ",",
    quotechar: str = '"',
    encoding: str = "utf-8",
    default_state: str | None = None,
) -> Iterable[Dict[str, str]]:
    """
    Yield each CSV row as a ``dict``. Path columns are relativised as described
    in ``relative_row``.
    """
    base_dir = csv_path.parent
    with csv_path.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
        if reader.fieldnames is None:
            raise ValueError(
                f"The file '{csv_path}' appears to be empty or missing a header line."
            )
        for row in reader:
            relative_row(
                row,
                base_dir,
                anchor_col="network_state",
                default_anchor=default_state,
            )
            yield row


def group_by_memory(
    rows_: Iterable[Mapping[str, str]], key: str
) -> Dict[str, List[Mapping[str, str]]]:
    """In‑memory grouping using ``defaultdict(list)``."""
    groups_: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows_:
        try:
            groups_[row[key]].append(row)
        except KeyError:
            raise KeyError(f"The column '{key}' does not exist in the CSV header.")
    return groups_


# Core routine
def build_master_h5(
    metadata_json: dict[str, list[Mapping[str, str]]], bundle_root: Path
) -> Path:
    """
    Parse ``metadata_json`` (the giant dictionary you posted), copy all external
    files into ``bundle_root`` and create the master HDF5 file.

    Parameters
    ----------
    metadata_json : dict
        Mapping of ``network_state`` → list of simulation dictionaries.
    bundle_root : Path
        Destination folder that will contain the master HDF5 file **and** a copy
        of every external file in the same relative hierarchy.

    Returns
    -------
    Path
        The absolute path of the created master file (``bundle_root/master.h5``).
    """
    _ensure_dir(bundle_root)
    master_path = bundle_root / "master.h5"
    logger.info("Creating master HDF5 file at %s", master_path)

    with h5py.File(master_path, "w") as master_h5:
        comp_kwargs = dict(compression="gzip", compression_opts=9)

        for network_state, sim_list in metadata_json.items():
            logger.info("Processing network_state '%s' (%d simulations)", network_state, len(sim_list))
            state_grp = master_h5.create_group(network_state)

            for sim in sim_list:
                sim_id = Path(sim["sim_info_file"]).stem
                logger.debug("  → handling simulation %s", sim_id)
                sim_grp = state_grp.create_group(sim_id)

                # scalar metadata as attributes
                for attr_key in ("network_path", "snudda_data", "duration", "network_state", "da_level"):
                    sim_grp.attrs[attr_key] = sim[attr_key]

                # numeric vectors
                ic = _parse_float_array(sim["input_conductance"])
                sim_grp.create_dataset("input_conductance", data=ic, **comp_kwargs)

                isf = _parse_optional_float_array(sim["input_scale_factor"])
                sim_grp.create_dataset("input_scale_factor", data=isf, **comp_kwargs)

                # external HDF5 files
                hdf5_files = {
                    "network_file": sim["network_file"],
                    "input_file": sim["input_file"],
                    "output_file": sim["output_file"],
                }

                # auxiliary text / JSON files
                text_files = {
                    "input_config": sim["input_config"],
                    "virtual_input_file": sim["virtual_input_file"],
                    "virtual_input_mapping": sim["virtual_input_mapping"],
                }

                for link_name, src_path_str in hdf5_files.items():
                    src_path = Path(src_path_str)
                    dst_path = _copy_to_bundle(src_path, bundle_root)
                    if dst_path is None:
                        logger.error(
                            "File not found: %s for item: %s in simulation %s – skipping link",
                            src_path,
                            link_name,
                            sim_id,
                        )
                        continue
                    _make_external_link(sim_grp, link_name, dst_path, bundle_root)

                for txt_name, src_path_str in text_files.items():
                    src_path = Path(src_path_str)
                    try:
                        _store_auxiliary_file(sim_grp, txt_name, src_path, bundle_root)
                    except Exception as exc:
                        logger.error(
                            "Failed to store auxiliary file %s for simulation %s: %s",
                            src_path,
                            sim_id,
                            exc,
                        )
    logger.info("Master HDF5 file creation completed: %s", master_path)
    return master_path


# Command line interface
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a bundled master HDF5 file that external‑links to all per‑simulation "
            "files referenced in a CSV metadata file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="Path to the CSV file containing simulation metadata.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        help="Directory where the bundle (master.h5 + copied files) will be created.",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="CSV delimiter (default: tab).",
    )
    parser.add_argument(
        "--quotechar",
        default='"',
        help="CSV quote character (default: \").",
    )
    parser.add_argument(
        "--default-state",
        default="d2oe_0.00",
        help="Default network_state used when path relativization needs a fall‑back anchor.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of console logging.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Adjust console log level according to CLI argument
    console_handler = next(
        (h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None
    )
    if console_handler:
        console_handler.setLevel(getattr(logging, args.log_level))

    csv_path: Path = args.metadata_csv.resolve()
    bundle_dir: Path = args.bundle_root.resolve()

    if not csv_path.is_file():
        logger.error("Metadata CSV file not found: %s", csv_path)
        sys.exit(1)

    logger.info("Starting bundling process")
    logger.info("  Metadata CSV : %s", csv_path)
    logger.info("  Bundle root   : %s", bundle_dir)

    try:
        rows = stream_csv_as_dicts(
            csv_path,
            delimiter=args.delimiter,
            quotechar=args.quotechar,
            encoding="utf-8",
            default_state=args.default_state,
        )
    except csv.Error as e:
        logger.error("CSV parsing error: %s", e)
        sys.exit(1)
    except UnicodeDecodeError as e:
        logger.error("Encoding error (utf-8): %s", e)
        sys.exit(1)

    try:
        groups = group_by_memory(rows, key="network_state")
    except KeyError as e:
        logger.error("%s", e)
        sys.exit(1)

    # Work relative to the CSV directory – mirrors original script behaviour
    os.chdir(csv_path.parent)
    logger.debug("Changed working directory to CSV parent: %s", os.getcwd())

    try:
        master_file = build_master_h5(groups, bundle_dir)
    except Exception as exc:
        logger.exception("Failed to build master HDF5 file: %s", exc)
        sys.exit(1)

    logger.info("Done – master file created at: %s", master_file)


if __name__ == "__main__":
    main()