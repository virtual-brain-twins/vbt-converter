import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Dict, MutableMapping, Mapping, List

# Path handling helpers
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
) -> Iterable[Dict[str, object]]:
    """
    Yield each CSV row as a ``dict`` with proper Python types.

    The columns ``da_level`` and ``duration`` become ``float``.
    ``input_conductance`` and ``input_scale_factor`` become ``list[float]``
    (``None`` values are left as ``None``).  All other columns keep the
    original string values.  Path columns are relativised by ``relative_row``.
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
