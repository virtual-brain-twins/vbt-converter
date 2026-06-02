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
    python make_master_h5.py <json‑metadata‑file> <output‑bundle‑folder>

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
"""
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Iterable, Mapping

import h5py
import numpy as np


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
    return dest.resolve()


def _parse_float_array(s: str) -> np.ndarray:
    """
    Convert a string like "[5e-10, 5e-10, …]" → 1‑D float64 array.
    """
    # strip the outer brackets, split on commas, and cast
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
    Insert an ExternalLink name into parent that points to the root group ``/``
    of the HDF5 file target_path.

    The link (and the saved attribute) use a relative path that is
    expressed with respect to bundle_root – this keeps the bundle portable.
    """
    rel_path = relative_path(bundle_root, target_path)

    # Create the ExternalLink.  h5py accepts a **string** filename, so we
    # convert the Path to str.
    parent[name] = h5py.ExternalLink(str(rel_path), "/")

    # Store the relative path  as an attribute for easy lookup later.
    # (If you still need the absolute path somewhere else, you can store it
    #  under a different attribute name, e.g. "absolute_path".)
    parent[name].attrs["relative_path"] = str(rel_path)


def relative_path(bundle_root: Path, target_path: Path) -> Path:
    # Compute a relative path from the bundle root to the target file.
    # `Path.relative_to` only works when `target_path` is inside the tree,
    # otherwise it raises ValueError.  Falling back to ``os.path.relpath`` gives
    # a sensible result for any location on the same filesystem.
    try:
        rel_path = target_path.relative_to(bundle_root)
    except ValueError:
        # Different trees – fall back to a generic relative path.
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
    Copy src_path into the bundle and store its contents inside *parent*
    as a compressed dataset (v‑len string or binary blob depending on size).
    """
    dst_path = _copy_to_bundle(src_path, bundle_root)
    if dst_path is None:
        raise FileNotFoundError(f"Failed to copy {src_path}")

    # read the whole file (they are modestly sized, otherwise we would stream)
    file_bytes = dst_path.read_bytes()
    size = len(file_bytes)

    if size <= MAX_BLOB_SIZE:
        # Store as a variable‑length UTF‑8 string (most convenient)
        text = file_bytes.decode("utf-8")
        _store_text_as_vlen_dataset(
            parent,
            dset_name,
            text,
            compress=COMPRESS_TEXT,
        )
    else:
        # Very large → store the raw bytes as a blob (still gzip‑compressed)
        _store_blob_dataset(
            parent,
            dset_name,
            file_bytes,
            compress=COMPRESS_TEXT,
        )

    # keep a tiny trace of where the file originally lived (helps debugging)
    parent[dset_name].attrs["original_path"] = str(relative_path(bundle_root, dst_path))


# Helper to store a JSON object as a compressed string dataset
def _store_blob_dataset(
        parent: h5py.Group,
        name: str,
        data_bytes: bytes,
        compress: bool = True,
        compression: str = "gzip",
        compression_opts: int = 9,
        chunk_target_mb: float = 1.0,
) -> None:
    """
    Store data_bytes in a chunked uint8 dataset with optional compression.
    The chunk size is chosen so that each chunk is roughly chunk_target_mb.
    """
    # Turn the raw bytes into a NumPy uint8 array
    arr = np.frombuffer(data_bytes, dtype=np.uint8)

    # Heuristic chunk size: aim for ~1 MiB chunks (adjustable)
    if compress:
        # HDF5 needs a *tuple* for chunk shape; we use a 1‑D chunk.
        chunk_bytes = int(chunk_target_mb * 1024 * 1024)
        # Ensure the chunk is not larger than the whole dataset
        chunk_len = min(chunk_bytes, arr.size)
        chunks = (chunk_len,)
        kw = dict(compression=compression, compression_opts=compression_opts,
                  chunks=chunks)
    else:
        kw = {}

    dset = parent.create_dataset(name, data=arr, **kw)
    dset.attrs["is_blob"] = True
    dset.attrs["encoding"] = "utf-8"  # we stored UTF‑8 text as bytes


def _store_text_as_vlen_dataset(
        parent: h5py.Group,
        name: str,
        text: str,
        compress: bool = True,
        compression: str = "gzip",
        compression_opts: int = 9,
) -> None:
    """
    Store text as a 1‑element variable‑length string dataset.

    The dataset is created with shape=(1,) so that HDF5 filters can be used.
    The function also writes a small helper attribute ``is_vlen`` which can be
    used downstream to know how to read the data back.
    """
    # Choose a variable‑length UTF‑8 dtype
    vlen_dtype = h5py.string_dtype(encoding="utf-8")  # <-- this is a vlen type

    # Convert the Python string to a NumPy array of length‑1, dtype=vlen
    arr = np.empty((1,), dtype=vlen_dtype)
    arr[0] = text

    # Build the keyword arguments for compression (only if requested)
    kw = {}
    if compress:
        kw.update(compression=compression, compression_opts=compression_opts)

    dset = parent.create_dataset(name, data=arr, **kw)
    # a flag that tells us later that this dataset holds a single string
    dset.attrs["is_vlen"] = True

# CSV Columns that always contain paths that we want to make relative.
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

# The three columns that must always use the default network state as their fallback anchor.
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
    Return a string that is *relative* to ``base_dir`` if ``abs_path`` is a
    descendant of ``base_dir``. If not, try to keep the part of the path that
    starts with ``fallback_anchor`` (if supplied).  If both strategies fail,
    the original string is returned unchanged.
    """
    p = Path(abs_path)

    # Fast path – the file *is* inside the CSV directory tree.
    try:
        return str(p.relative_to(base_dir))
    except ValueError:
        pass  # not a sub‑path of ``base_dir``

    # Fallback: keep the part of the path that starts with the anchor.
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
    Mutate row in‑place: for every column listed in ``PATH_COLUMNS`` replace
    its value with a path that is relative to ``base_dir``.

    * For columns in ``ANCHOR_FORCE_DEFAULT`` the fallback anchor is
      ``default_anchor`` (the command‑line supplied default network state).
    * For all other columns the fallback anchor is the value taken from
      ``row[anchor_col]`` (if present).
    """
    row_anchor = row.get(anchor_col)  # e.g. "d2oe_1.23"
    for col in PATH_COLUMNS & row.keys():
        # Decide which fallback anchor we ought to use for this column.
        fallback = (
            default_anchor
            if col in ANCHOR_FORCE_DEFAULT
            else row_anchor
        )
        row[col] = _relative_path(row[col], base_dir, fallback_anchor=fallback)


# CSV streaming helper (unchanged apart from passing the new arg)
def stream_csv_as_dicts(
        csv_path: Path,
        delimiter: str = ",",
        quotechar: str = '"',
        encoding: str = "utf-8",
        default_state: str | None = None,
) -> Iterable[Dict[str, str]]:
    """
    Yield each CSV row as a ``dict``.  Path columns are relativized as
    described in ``relative_row``.
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


# Group‑by helper
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
def build_master_h5(metadata_json: dict[str, list[Mapping[str, str]]], bundle_root: Path) -> Path:
    """
    Parse ``metadata_json`` (the giant dictionary you posted), copy all external
    files into ``bundle_root`` and create the master HDF5 file.

    Parameters
    ----------
    metadata_json : Path
        JSON file that contains the mapping you posted.
    bundle_root : Path
        Destination folder that will contain the master HDF5 file **and** a copy
        of every external file in the same relative hierarchy.

    Returns
    -------
    Path
        The absolute path of the created master file (``bundle_root/master.h5``).
    """
    # Prepare the bundle folder.
    _ensure_dir(bundle_root)

    master_path = bundle_root / "master.h5"

    # Open the master file and start filling it.
    with h5py.File(master_path, "w") as master_h5:
        # Use a relatively high gzip compression level for the tiny numeric arrays.
        comp_kwargs = dict(compression="gzip", compression_opts=9)

        for network_state, sim_list in metadata_json.items():
            # One top‑level group per network_state (the key of the outer dict)
            state_grp = master_h5.create_group(network_state)

            for sim in sim_list:
                # Create a group that uniquely identifies a simulation.
                # The basename of the *sim_info_file* is a good unique ID.
                sim_id = Path(sim["sim_info_file"]).stem  # e.g. "sim-info-ef5a0245e2f04..."
                sim_grp = state_grp.create_group(sim_id)

                # Store the scalar metadata as attributes.
                for attr_key in ("network_path", "snudda_data",
                                 "duration", "network_state", "da_level"):
                    sim_grp.attrs[attr_key] = sim[attr_key]

                # Store the two numeric vectors as compressed datasets.
                ic = _parse_float_array(sim["input_conductance"])
                sim_grp.create_dataset("input_conductance",
                                       data=ic, **comp_kwargs)

                isf = _parse_optional_float_array(sim["input_scale_factor"])
                sim_grp.create_dataset("input_scale_factor",
                                       data=isf, **comp_kwargs)

                # Copy every external file into the bundle (preserving hierarchy)
                # Files that are HDF5 and therefore can be linked:
                hdf5_files = {
                    "network_file": sim["network_file"],
                    "input_file": sim["input_file"],
                    "output_file": sim["output_file"],
                }

                # Text / JSON files we also copy (they are small, we keep a copy
                # inside the bundle but we do **not** link them as VDS – a simple
                # file copy is enough).
                text_files = {
                    "input_config": sim["input_config"],
                    "virtual_input_file": sim["virtual_input_file"],
                    "virtual_input_mapping": sim["virtual_input_mapping"],
                }

                # Create external links for the heavy HDF5 files.
                for link_name, src_path_str in hdf5_files.items():
                    src_path = Path(src_path_str)
                    dst_path = _copy_to_bundle(src_path, bundle_root)
                    if dst_path is None:
                        print(
                            f"File not found: {src_path} for item: {link_name} in simulation {sim_id} – skipping link")
                        continue
                    # ExternalLink points to the *root* of the source HDF5 file.
                    _make_external_link(sim_grp, link_name, dst_path, bundle_root)

                # Copy the auxiliary text / json files.  They are stored as
                #     *attributes* with the same name – tiny enough to keep in RAM.
                for txt_name, src_path_str in text_files.items():
                    src_path = Path(src_path_str)
                    _store_auxiliary_file(sim_grp, txt_name, src_path, bundle_root)

    return master_path


# Command line interface (optional, for easy testing)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} <metadata‑csv> <bundle‑folder>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1]).resolve()
    bundle_dir = Path(sys.argv[2]).resolve()
    if not csv_path.is_file():
        sys.exit(f"File not found: {csv_path}")

    try:
        rows = stream_csv_as_dicts(
            csv_path,
            delimiter="\t",
            quotechar='"',
            encoding="utf-8",
            default_state="d2oe_0.00",
        )
    except csv.Error as e:
        sys.exit(f"CSV parsing error: {e}")
    except UnicodeDecodeError as e:
        sys.exit(f"Encoding error ({"utf-8"}): {e}")
    try:
        groups = group_by_memory(rows, key='network_state')
    except KeyError as e:
        sys.exit(f"{e}")
    os.chdir(os.path.dirname(csv_path))
    print(f"Current working directory: {os.getcwd()}")
    print(f" Building bundle in: {bundle_dir}")
    master_file = build_master_h5(groups, bundle_dir)
    print(f"Done – master file created at: {master_file}")
