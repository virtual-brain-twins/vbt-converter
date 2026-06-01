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

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import h5py

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    """Make sure a directory exists (like `mkdir -p`)."""
    path.mkdir(parents=True, exist_ok=True)


def _copy_to_bundle(src: Path, bundle_root: Path) -> Path:
    """
    Copy *src* (file) into the *bundle_root*, keeping the relative directory
    hierarchy intact.  Returns the new absolute path inside the bundle.
    """
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")

    rel = src.relative_to(src.anchor)          # keep everything after the leading '/'
    dest = bundle_root / rel
    _ensure_dir(dest.parent)
    shutil.copy2(src, dest)                    # preserve timestamps / permissions
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


def _make_external_link(parent: h5py.Group, name: str, target_path: Path) -> None:
    """
    Insert an *ExternalLink* named *name* into *parent* that points to the
    root group ``/`` of the HDF5 file *target_path*.
    """
    # ``target_path`` must be a string (h5py does not accept pathlib)
    parent[name] = h5py.ExternalLink(str(target_path), "/")
    # store the original (absolute) path as an attribute for reference
    parent[name].attrs["original_path"] = str(target_path)


# ----------------------------------------------------------------------
# Core routine
# ----------------------------------------------------------------------


def build_master_h5(metadata_json: Path, bundle_root: Path) -> Path:
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
    # ------------------------------------------------------------------
    # 0. Load the JSON – it is only a few MB, safe to hold in RAM.
    # ------------------------------------------------------------------
    with metadata_json.open("rt", encoding="utf-8") as fh:
        data: Dict[str, List[Dict[str, Any]]] = json.load(fh)

    # ------------------------------------------------------------------
    # 1. Prepare the bundle folder.
    # ------------------------------------------------------------------
    _ensure_dir(bundle_root)

    master_path = bundle_root / "master.h5"

    # ------------------------------------------------------------------
    # 2. Open the master file and start filling it.
    # ------------------------------------------------------------------
    with h5py.File(master_path, "w") as master_h5:
        # Use a relatively high gzip compression level for the tiny numeric arrays.
        comp_kwargs = dict(compression="gzip", compression_opts=9)

        for network_state, sim_list in data.items():
            # One top‑level group per network_state (the key of the outer dict)
            state_grp = master_h5.create_group(network_state)

            for sim in sim_list:
                # ------------------------------------------------------------------
                # 2a. Create a group that uniquely identifies a simulation.
                #     The basename of the *sim_info_file* is a good unique ID.
                # ------------------------------------------------------------------
                sim_id = Path(sim["sim_info_file"]).stem  # e.g. "sim-info-ef5a0245e2f04..."
                sim_grp = state_grp.create_group(sim_id)

                # ------------------------------------------------------------------
                # 2b. Store the *scalar* metadata as attributes.
                # ------------------------------------------------------------------
                for attr_key in ("network_path", "snudda_data",
                                 "duration", "network_state", "da_level"):
                    sim_grp.attrs[attr_key] = sim[attr_key]

                # ------------------------------------------------------------------
                # 2c. Store the two numeric vectors as compressed datasets.
                # ------------------------------------------------------------------
                ic = _parse_float_array(sim["input_conductance"])
                sim_grp.create_dataset("input_conductance",
                                       data=ic, **comp_kwargs)

                isf = _parse_optional_float_array(sim["input_scale_factor"])
                sim_grp.create_dataset("input_scale_factor",
                                       data=isf, **comp_kwargs)

                # ------------------------------------------------------------------
                # 2d. Copy every external file into the bundle (preserving hierarchy)
                # ------------------------------------------------------------------
                # Files that are HDF5 and therefore can be linked:
                hdf5_files = {
                    "network_file": sim["network_file"],
                    "input_file":   sim["input_file"],
                    "output_file":  sim["output_file"],
                }

                # Text / JSON files we also copy (they are small, we keep a copy
                # inside the bundle but we do **not** link them as VDS – a simple
                # file copy is enough).
                text_files = {
                    "input_config":           sim["input_config"],
                    "virtual_input_file":     sim["virtual_input_file"],
                    "virtual_input_mapping":  sim["virtual_input_mapping"],
                }

                # ------------------------------------------------------------------
                # 2e. Create external links for the heavy HDF5 files.
                # ------------------------------------------------------------------
                for link_name, src_path_str in hdf5_files.items():
                    src_path = Path(src_path_str)
                    dst_path = _copy_to_bundle(src_path, bundle_root)
                    # ExternalLink points to the *root* of the source HDF5 file.
                    _make_external_link(sim_grp, link_name, dst_path)

                # ------------------------------------------------------------------
                # 2f. Copy the auxiliary text / json files.  They are stored as
                #     *attributes* with the same name – tiny enough to keep in RAM.
                # ------------------------------------------------------------------
                for txt_name, src_path_str in text_files.items():
                    src_path = Path(src_path_str)
                    dst_path = _copy_to_bundle(src_path, bundle_root)

                    # Load the whole file (they are a few KB at most)
                    with dst_path.open("rt", encoding="utf-8") as fh:
                        txt = fh.read()
                    # Store as a *scalar* string dataset (compressed as well)
                    sim_grp.create_dataset(txt_name,
                                           data=np.string_(txt),
                                           **comp_kwargs)
                    # also keep the absolute path as an attribute for bookkeeping
                    sim_grp[txt_name].attrs["original_path"] = str(dst_path)

    return master_path


# ----------------------------------------------------------------------
# Command line interface (optional, for easy testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} <metadata‑json> <bundle‑folder>", file=sys.stderr)
        sys.exit(1)

    meta_path = Path(sys.argv[1]).resolve()
    bundle_dir = Path(sys.argv[2]).resolve()

    os.chdir(os.path.dirname(meta_path))
    print(f"Current working directory: {os.getcwd()}")

    if not meta_path.is_file():
        print(f"❌ JSON file not found: {meta_path}", file=sys.stderr)
        sys.exit(2)

    print(f"🔧 Building bundle in: {bundle_dir}")
    master_file = build_master_h5(meta_path, bundle_dir)
    print(f"✅ Done – master file created at: {master_file}")