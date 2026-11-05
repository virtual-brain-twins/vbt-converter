# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

"""
Utilities for reading, normalising and grouping CSV data used in network‑state
simulations.

The module provides helpers to:

* Convert absolute file paths stored in CSV rows to paths relative to a given
  base directory, optionally preserving a configurable “anchor” segment.
* Stream a CSV file as dictionaries while performing lightweight type conversion
  on a few known columns.
* Group an iterable of row mappings by an arbitrary column value using an in‑
  memory ``defaultdict``.

Typical usage::

    from pathlib import Path
    from mymodule import stream_csv_as_dicts, group_by_memory

    csv_path = Path("results/simulation.csv")
    rows = list(stream_csv_as_dicts(csv_path, default_state="network_state"))
    groups = group_by_memory(rows, key="network_state")
"""
import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

# Path handling helpers -------------------------------------------------------

# Columns that contain file system paths and therefore need to be relativised.
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

# Columns for which the ``fallback_anchor`` argument should always be used,
# regardless of the value in the CSV row.
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
    """Return a path that is relative to ``base_dir`` when possible.

    The function attempts to make ``abs_path`` relative to ``base_dir`` using
    :pymeth:`Path.relative_to`.  If ``abs_path`` is not located under ``base_dir``,
    the function falls back to returning a path that starts with ``fallback_anchor``
    (if supplied) and the remainder of the original path.  When ``fallback_anchor``
    is ``None`` or not present in ``abs_path``, the original absolute path is
    returned unchanged.

    Args:
        abs_path: The absolute path to be processed.
        base_dir: Directory to which the path should be made relative.
        fallback_anchor: Optional string that, when present in ``abs_path``, is
            retained as the leading segment of the returned path.

    Returns:
        A string representing either a relative path or a path that begins with
        ``fallback_anchor`` (if applicable).

    Raises:
        None.  Errors from :pymeth:`Path.relative_to` are intercepted and handled
        internally.
    """
    p = Path(abs_path)

    try:
        return str(p.relative_to(base_dir))
    except ValueError:
        # ``abs_path`` is not under ``base_dir`` – continue with fallback handling.
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
    """Mutate *row* in‑place, relativising any path columns.

    For each column listed in :data:`PATH_COLUMNS` that also exists in the row,
    replace its value with a version that is relative to ``base_dir``.  The
    ``fallback_anchor`` used for the conversion is taken from the column
    ``anchor_col`` of the same row, unless the column is listed in
    :data:`ANCHOR_FORCE_DEFAULT`, in which case ``default_anchor`` is used instead.

    Args:
        row: Mapping representing a CSV row; will be modified directly.
        base_dir: Directory against which path columns are relativised.
        anchor_col: Column name whose value is used as the fallback anchor when
            needed.  Defaults to ``"network_state"``.
        default_anchor: Anchor to use when a column is listed in
            :data:`ANCHOR_FORCE_DEFAULT`.  If ``None`` and the column requires a
            fallback, the value from ``anchor_col`` is used.

    Returns:
        None

    Raises:
        None.
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
) -> Iterable[Dict[str, object]]:
    """Yield each CSV row as a dictionary with appropriate Python types.

    The CSV is streamed row‑by‑row, converting a handful of known columns to
    richer Python objects:

    * ``da_level`` and ``duration`` → ``float``
    * ``input_conductance`` and ``input_scale_factor`` → ``list[float]``
      (``None`` elements are preserved for the latter)

    Path columns defined in :data:`PATH_COLUMNS` are normalised by
    :func:`relative_row`.  All other columns retain their original string values.

    Args:
        csv_path: Path to the CSV file to be read.
        delimiter: Field delimiter used in the CSV.  Defaults to a comma.
        quotechar: Character used to quote fields containing special characters.
            Defaults to ``"``.
        encoding: Text encoding of the CSV file.  Defaults to ``"utf-8"``.
        default_state: Optional fallback anchor that is passed to
            :func:`relative_row` when a row does not contain the ``anchor_col``
            value.

    Yields:
        A ``dict`` mapping column names to values with the conversions described
        above.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV file is empty or missing a header line.
        ValueError: If literal evaluation of ``input_conductance`` or
            ``input_scale_factor`` fails due to malformed data.
        SyntaxError: Propagated from ``ast.literal_eval`` when the column value
            cannot be parsed as a Python literal.
        KeyError: Propagated from :func:`relative_row` when required anchor
            information is missing.
    """
    # Conversion map – keeps the hot‑loop tiny and avoids repetitive code.
    _conv = {
        "da_level": float,
        "duration": float,
        "input_conductance": lambda s: [float(v) for v in ast.literal_eval(s)],
        "input_scale_factor": lambda s: [
            float(v) if v is not None else None
            for v in ast.literal_eval(s)
        ],
    }

    base_dir = csv_path.parent
    with csv_path.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
        if reader.fieldnames is None:
            raise ValueError(
                f"The file '{csv_path}' appears to be empty or missing a header line."
            )
        for row in reader:
            # Relativise path columns (original behaviour)
            relative_row(
                row,
                base_dir,
                anchor_col="network_state",
                default_anchor=default_state,
            )
            # Fast, in‑place type conversion for the selected columns
            for key, fn in _conv.items():
                if (val := row.get(key)) is not None:  # guard missing columns
                    row[key] = fn(val)
            yield row


def group_by_memory(
    rows_: Iterable[Mapping[str, str]],
    key: str,
) -> Dict[str, List[Mapping[str, str]]]:
    """Group rows in memory by the value of a specified column.

    The function builds a ``defaultdict(list)`` where each key corresponds to a
    distinct value found in ``key`` across the provided rows, and the associated
    value is a list of the rows that share that key.

    Args:
        rows_: An iterable of row mappings (e.g., dictionaries) to be grouped.
        key: Column name used as the grouping key.

    Returns:
        A dictionary mapping each distinct key value to a list of row mappings
        that contain that value.

    Raises:
        KeyError: If any row does not contain the specified ``key`` column.
    """
    groups_: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows_:
        try:
            groups_[row[key]].append(row)
        except KeyError:
            raise KeyError(f"The column '{key}' does not exist in the CSV header.")
    return groups_