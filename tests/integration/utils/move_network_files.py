# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""
move_network_files.py

Copy only the *files* that belong to a particular network_state from a
source tree (identified by a user‑provided *prefix* directory) into an
output directory that mirrors the original layout:

    <output_dir>/<network_state>/<relative_path_inside_state>

The script reads a CSV (tab‑ or comma‑separated) that contains many absolute
paths.  For every entry that

* is an absolute path,
* contains the value of the column ``network_state`` somewhere inside the
  path,

the part of the path that appears *before* the first occurrence of the
network_state folder is replaced with the user supplied ``prefix_path``.
If the resulting path points to an existing **file**, the file is copied
to the output location; if it points to a directory the entry is ignored.

Usage
-----
    python move_network_files.py \
        --csv      /path/to/table.csv \
        --prefix   /home/jmurugan/work/Artefacts/VT-236/vbt-converter/tests/test_data/KTH_Dardel_run_networks_2026-06-12 \
        --out      /home/jmurugan/work/Artefacts/VT-236/vbt-converter/tests/collected_files \
        [--dry-run]               # only show what would be copied

Generate a reduced CSV file with a fixed number of rows, e.g. 10 rows alone
head -n 10 run_list.csv > run_list_reduced.csv
Use the reduced CSV file to create a reduced set of files in the output directory.

"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def sniff_dialect(csv_path: Path) -> csv.Dialect:
    """
    Detect whether the file is comma‑ or tab‑separated.
    """
    with csv_path.open(newline="") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        # fallback to tab
        dialect = csv.excel_tab
    return dialect


def read_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Return a list of rows (as dictionaries) and the header list.
    """
    dialect = sniff_dialect(csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows = [row for row in reader]
        header = reader.fieldnames if reader.fieldnames else []
    return rows, header


def build_new_path(original: str,
                  network_state: str,
                  prefix_path: Path) -> Path:
    """
    Replace everything *before* the first occurrence of ``network_state``
    with ``prefix_path``.
    Example:
        original = "/cfs/.../networks/d2oe_0.25/sim-info.json"
        network_state = "d2oe_0.25"
        prefix_path = "/home/user/data"
        → "/home/user/data/d2oe_0.25/sim-info.json"
    """
    idx = original.find(network_state)
    if idx == -1:
        raise ValueError(f"network_state '{network_state}' not found in path '{original}'")
    # keep the part *starting* with the state (including the slash before it)
    # e.g. "/networks/d2oe_0.25/..." → "d2oe_0.25/..."
    # The slice from idx may start with the state or with a preceding '/'
    remainder = original[idx:].lstrip("/")  # remove possible leading slash
    new_path = prefix_path / remainder
    return new_path.resolve()


def copy_file(src: Path, dst: Path, dry_run: bool = False) -> None:
    """
    Copy ``src`` → ``dst`` creating parent directories as needed.
    """
    if dry_run:
        print(f"[DRY‑RUN] Would copy: {src} → {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied: {src} → {dst}")


def process_row(row: Dict[str, str],
                prefix_path: Path,
                output_path: Path,
                dry_run: bool = False) -> int:
    """
    Process a single CSV row. Returns the number of files successfully copied.
    """
    copied = 0
    network_state = row.get("network_state")
    if not network_state:
        sys.stderr.write("Warning: row without a network_state – skipping\n")
        return copied

    for col, value in row.items():
        # Skip the column that holds the network_state itself and any non‑string data
        if col == "network_state" or not isinstance(value, str) or not value:
            continue

        # Only consider absolute paths that contain the state name
        if not os.path.isabs(value) or network_state not in value:
            continue

        try:
            src_path = build_new_path(value, network_state, prefix_path)
        except ValueError as e:
            sys.stderr.write(f"Warning: {e}\n")
            continue

        if not src_path.exists():
            sys.stderr.write(f"Warning: source does not exist → {src_path}\n")
            continue

        if src_path.is_dir():
            # The spec says “skip the path” when it is a directory
            continue

        # Destination:  <output>/<network_state>/<relative path inside the state>
        rel_inside_state = src_path.relative_to(prefix_path / network_state)
        dst_path = output_path / network_state / rel_inside_state

        copy_file(src_path, dst_path, dry_run=dry_run)
        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy files listed in a CSV after re‑rooting them under a "
                    "user‑provided prefix, preserving per‑network_state folders."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to the CSV (or TSV) file containing the table.")
    parser.add_argument("--prefix", required=True,
                        help="Root directory that should replace the original prefix "
                             "(the part before the network_state folder).")
    parser.add_argument("--out", required=True,
                        help="Output directory that will receive the copied files. "
                             "A sub‑folder per network_state is created automatically.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without performing any file "
                             "operations.")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    prefix_path = Path(args.prefix).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    dry_run = args.dry_run

    if not csv_path.is_file():
        sys.exit(f"Error: CSV file not found: {csv_path}")

    if not prefix_path.is_dir():
        sys.exit(f"Error: Prefix directory does not exist: {prefix_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    rows, header = read_csv(csv_path)

    total_copied = 0
    for i, row in enumerate(rows, start=1):
        copied = process_row(row, prefix_path, output_path, dry_run=dry_run)
        total_copied += copied
        if i % 50 == 0:
            print(f"... processed {i} rows, total files copied so far: {total_copied}")

    print(f"\nFinished processing {len(rows)} rows.")
    print(f"Total files copied: {total_copied}")
    dst_csv = output_path / "run_list.csv"
    if dry_run:
        print(" (dry‑run – no files were actually written)")
        print(f"[DRY‑RUN] Would copy CSV: {csv_path} → {dst_csv}")
    else:
        shutil.copy2(csv_path, dst_csv)  # keep metadata (timestamps, mode)
        print(f"Copied CSV: {csv_path} → {dst_csv}")
        shutil.copy2(csv_path.parent / "protocol_info.pdf", output_path / "protocol_info.pdf")
        print(f"Copied protocol_info.pdf to {output_path / 'protocol_info.pdf'}")

if __name__ == "__main__":
    main()