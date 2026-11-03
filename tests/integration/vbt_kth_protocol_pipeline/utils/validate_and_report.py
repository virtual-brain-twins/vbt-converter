# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
"""
validate_and_report.py

Validate a ``rates_voltages.npz`` archive produced by
:class:`RateVoltageExporter`, optionally compare it with a ground‑truth
archive and write a human‑readable HTML report.

Usage
-----
    python validate_and_report.py -i path/to/rates_voltages.npz \
                                  -o report.html \
                                  [-g ground_truth.npz]

The script exits with status 0 on success, 1 on validation failure.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Helper functions for HTML rendering
# ----------------------------------------------------------------------
def _format_array(name: str, arr: np.ndarray) -> str:
    """Return a table row describing *arr* (shape, dtype, min, max)."""
    shape = "×".join(map(str, arr.shape)) if arr.shape else "Ø"
    dtype = str(arr.dtype)

    # Compute human‑readable min / max – protect against empty arrays
    if arr.size:
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        # Use a compact format, but fall back to full repr if conversion fails
        try:
            arr_min_str = f"{arr_min:.6g}"
            arr_max_str = f"{arr_max:.6g}"
        except Exception:          # pragma: no cover
            arr_min_str = repr(arr_min)
            arr_max_str = repr(arr_max)
    else:
        arr_min_str = arr_max_str = "n/a"

    name = html.escape(name)

    return f"""
        <tr>
            <td><code>{name}</code></td>
            <td>{shape}</td>
            <td>{dtype}</td>
            <td>{arr_min_str}</td>
            <td>{arr_max_str}</td>
        </tr>
    """


# ----------------------------------------------------------------------
# Validation logic
# ----------------------------------------------------------------------
def validate_rates_voltages(npz_path: pathlib.Path) -> Dict[str, Any]:
    """
    Load *npz_path* (produced by :class:`RateVoltageExporter`) and run a set
    of consistency checks.

    Returns a dict with the parsed ``results`` entry (nested structure).  Any
    problem raises :class:`ValueError` with a clear message.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"File not found: {npz_path}")

    try:
        archive = np.load(npz_path, allow_pickle=True)   # pickled dict inside
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not read {npz_path}: {exc}") from exc

    if "results" not in archive:
        raise ValueError("Missing top‑level key ``results`` in the archive.")

    # ``results`` is a pickled dict → we need the actual python object
    results = archive["results"].item() if isinstance(archive["results"], np.ndarray) else archive["results"]
    if not isinstance(results, dict):
        raise ValueError("The ``results`` entry is not a dict as expected.")

    # ------------------------------------------------------------------
    # 1️⃣ Basic top‑level sanity checks
    # ------------------------------------------------------------------
    if not results:
        raise ValueError("The ``results`` dictionary is empty – nothing to validate.")

    # Gather the length of the firing‑rate vectors; they must all be identical.
    rate_lengths: List[int] = []

    for sim_id, sim_content in results.items():
        if not isinstance(sim_content, dict):
            raise ValueError(f"Simulation entry {sim_id!r} is not a dict.")
        if "global_averages" not in sim_content:
            raise ValueError(f"Simulation {sim_id!r} missing the required ``global_averages`` key.")

        # ------------------------------------------------------------------
        # 1.1️⃣ Global averages sanity
        # ------------------------------------------------------------------
        global_avg = sim_content["global_averages"]
        if not isinstance(global_avg, dict):
            raise ValueError(f"Simulation {sim_id!r} – ``global_averages`` is not a dict.")
        if "firing_rate" not in global_avg:
            raise ValueError(f"Simulation {sim_id!r} – missing ``global_averages['firing_rate']``.")
        if not isinstance(global_avg["firing_rate"], np.ndarray):
            raise ValueError(f"Simulation {sim_id!r} – global firing_rate is not an ndarray.")
        rate_lengths.append(global_avg["firing_rate"].shape[0])

        # ------------------------------------------------------------------
        # 1.2️⃣ Per‑population sanity
        # ------------------------------------------------------------------
        for pop_name, pop_data in sim_content.items():
            if pop_name == "global_averages":
                continue
            if not isinstance(pop_data, dict):
                raise ValueError(f"Simulation {sim_id!r}, population {pop_name!r}: not a dict.")
            # firing_rate
            fr = pop_data.get("firing_rate")
            if not isinstance(fr, np.ndarray):
                raise ValueError(f"Simulation {sim_id!r}, pop {pop_name!r}: missing/invalid ``firing_rate``.")
            if fr.shape != (rate_lengths[-1],):
                raise ValueError(
                    f"Simulation {sim_id!r}, pop {pop_name!r}: firing_rate shape {fr.shape} "
                    f"does not match expected ({rate_lengths[-1]},)."
                )
            # avg_voltage (may be empty)
            av = pop_data.get("avg_voltage")
            if not isinstance(av, np.ndarray):
                raise ValueError(f"Simulation {sim_id!r}, pop {pop_name!r}: ``avg_voltage`` is not an ndarray.")
            # If non‑empty, we just check it is 1‑D
            if av.size and av.ndim != 1:
                raise ValueError(
                    f"Simulation {sim_id!r}, pop {pop_name!r}: avg_voltage must be 1‑D, got shape {av.shape}"
                )

            # Numerical sanity
            if np.isnan(fr).any():
                raise ValueError(f"Simulation {sim_id!r}, pop {pop_name!r}: firing_rate contains NaN.")
            if np.isinf(fr).any():
                raise ValueError(f"Simulation {sim_id!r}, pop {pop_name!r}: firing_rate contains Inf.")
            if av.size and (np.isnan(av).any() or np.isinf(av).any()):
                raise ValueError(f"Simulation {sim_id!r}, pop {pop_name!r}: avg_voltage contains NaN/Inf.")

    # ------------------------------------------------------------------
    # 2️⃣ Global consistency of the rate vector length across simulations
    # ------------------------------------------------------------------
    if len(set(rate_lengths)) != 1:
        raise ValueError(
            f"Inconsistent number of time‑bins across simulations: {sorted(set(rate_lengths))}"
        )
    # If we reach this point the archive meets all checks.
    return results


# ----------------------------------------------------------------------
# Comparison logic (similar to the one you already had)
# ----------------------------------------------------------------------
def compare_npz(
    file_a: pathlib.Path,
    file_b: pathlib.Path,
    rtol: float = 1e-04,
    atol: float = 1e-08,
) -> Tuple[bool, str]:
    """
    Compare two ``rates_voltages.npz`` archives.

    Returns ``(True, "…")`` if they match within the tolerances,
    otherwise ``(False, description)``.
    """
    try:
        with np.load(file_a, allow_pickle=True) as a, np.load(file_b, allow_pickle=True) as b:
            # Keys at the top level must be identical:
            if set(a.files) != set(b.files):
                return False, f"Top‑level key mismatch: {a.files} vs {b.files}"

            # Both must contain a ``results`` dict:
            if "results" not in a or "results" not in b:
                return False, "`results` key missing in one of the files."

            # Unpickle the dicts:
            dict_a = a["results"].item() if isinstance(a["results"], np.ndarray) else a["results"]
            dict_b = b["results"].item() if isinstance(b["results"], np.ndarray) else b["results"]

            # Same set of simulation IDs?
            if set(dict_a) != set(dict_b):
                return False, f"Simulation‑ID mismatch: {sorted(dict_a)} vs {sorted(dict_b)}"

            for sim_id in dict_a:
                sim_a = dict_a[sim_id]
                sim_b = dict_b[sim_id]

                # Global averages
                for key in ("firing_rate", "avg_voltage"):
                    arr_a = sim_a["global_averages"].get(key, np.array([]))
                    arr_b = sim_b["global_averages"].get(key, np.array([]))
                    ok, msg = _compare_arrays(arr_a, arr_b, f"global_averages['{key}'] (sim {sim_id})", rtol, atol)
                    if not ok:
                        return False, msg

                # Populations – same set of population names?
                pops_a = {k for k in sim_a if k != "global_averages"}
                pops_b = {k for k in sim_b if k != "global_averages"}
                if pops_a != pops_b:
                    return False, f"Population name mismatch in sim {sim_id}: {pops_a} vs {pops_b}"

                for pop in pops_a:
                    for key in ("firing_rate", "avg_voltage"):
                        arr_a = sim_a[pop].get(key, np.array([]))
                        arr_b = sim_b[pop].get(key, np.array([]))
                        ok, msg = _compare_arrays(arr_a, arr_b, f"{pop}/{key} (sim {sim_id})", rtol, atol)
                        if not ok:
                            return False, msg

        return True, "Files are identical (within tolerances)."
    except Exception as exc:  # pragma: no cover
        return False, f"Error during comparison: {exc}"


def _compare_arrays(arr1: np.ndarray, arr2: np.ndarray, label: str, rtol: float, atol: float) -> Tuple[bool, str]:
    """Utility used by :func:`compare_npz` – compare two arrays."""
    if arr1.shape != arr2.shape:
        return False, f"Shape mismatch for {label}: {arr1.shape} vs {arr2.shape}"
    if arr1.dtype != arr2.dtype:
        return False, f"Dtype mismatch for {label}: {arr1.dtype} vs {arr2.dtype}"
    if np.issubdtype(arr1.dtype, np.floating):
        if not np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
            # Show the first differing element to aid debugging
            diff = np.abs(arr1 - arr2)
            idx = tuple(np.unravel_index(np.argmax(diff), diff.shape))
            return (
                False,
                f"Value mismatch for {label} at index {idx}: {arr1[idx]} vs {arr2[idx]}",
            )
    else:
        if not np.array_equal(arr1, arr2):
            return False, f"Value mismatch for non‑float {label}"
    return True, ""


# ----------------------------------------------------------------------
# HTML report generation
# ----------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Rates‑Voltages Validation – {filename}</title>
    <style>
        body {{font-family: Arial, Helvetica, sans-serif; margin: 2rem;}}
        h1 {{color: #333;}}
        .banner {{padding: 1rem; border-radius: 8px; margin-bottom: 2rem;}}
        .success {{background-color: #e6ffed; color: #2e7d32;}}
        .error   {{background-color: #ffebee; color: #c62828;}}
        table {{border-collapse: collapse; width: 100%; margin-top: 1rem;}}
        th, td {{border: 1px solid #ddd; padding: 0.5rem; text-align: left;}}
        th {{background-color: #f2f2f2;}}
        code {{background: #f9f9f9; padding: 2px 4px; border-radius: 4px;}}
    </style>
</head>
<body>
    <h1>Rates‑Voltages Validation Report</h1>
    <p><strong>File:</strong> {filename}<br>
       <strong>Generated on:</strong> {timestamp}</p>

    <div class="banner {status_class}">
        <strong>{status_message}</strong>
    </div>

    {content}
    {comparison_section}
</body>
</html>
"""


def build_summary_section(results: Dict[str, Any]) -> str:
    """High‑level overview – number of simulations, populations, bin count, etc."""
    n_sim = len(results)
    # All firing‑rate vectors must have the same length – we can pick the first one.
    first_sim = next(iter(results.values()))
    n_bins = first_sim["global_averages"]["firing_rate"].shape[0]

    # Count distinct population names across all simulations
    pop_names = set()
    for sim in results.values():
        pop_names.update(k for k in sim if k != "global_averages")
    n_pops = len(pop_names)

    table = f"""
    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Simulations</td><td>{n_sim}</td></tr>
        <tr><td>Distinct populations</td><td>{n_pops}</td></tr>
        <tr><td>Time‑bins per firing‑rate vector</td><td>{n_bins}</td></tr>
    </table>
    """
    return table


def build_population_tables(results: Dict[str, Any]) -> str:
    """
    Produce a detailed table for *every* population (including the
    ``global_averages`` pseudo‑population).  We flatten the hierarchy into a
    single table for readability.
    """
    rows = ""
    for sim_id, sim_content in results.items():
        for pop_name, pop_data in sim_content.items():
            # Use a qualified name so a reader can tell which simulation it belongs to
            qualified_name = f"{sim_id}/{pop_name}"

            # firing_rate
            fr = pop_data.get("firing_rate", np.array([]))
            rows += _format_array(f"{qualified_name} – firing_rate", fr)

            # avg_voltage (may be empty)
            av = pop_data.get("avg_voltage", np.array([]))
            rows += _format_array(f"{qualified_name} – avg_voltage", av)

    html_table = f"""
    <h2>Population‑level tensors (summary)</h2>
    <table>
        <tr><th>Name</th><th>Shape</th><th>dtype</th><th>Min</th><th>Max</th></tr>
        {rows}
    </table>
    """
    return html_table


def build_artifact_section() -> str:
    """
    Link to the stored NPZ artifact.
    """
    url = "https://vbt-minio.fz-juelich.de/console/browser/vbt-gitlab-cache/project%2F8681%2Fvbt-kth-bg-protocol-cache-protected"
    return f"""
    <h2>Artifact</h2>
    <p><a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">
        converted.npz</a></p>
    """


def build_comparison_section(comp: Optional[Tuple[bool, str]]) -> str:
    """Render the ground‑truth comparison block (or nothing)."""
    if comp is None:
        return ""
    success, msg = comp
    status = "success" if success else "error"
    emoji = "✅" if success else "❌"
    return f"""
    <h2>Ground‑truth comparison</h2>
    <div class="banner {status}">
        <strong>{emoji} {html.escape(msg)}</strong>
    </div>
    """


def write_html_report(
    out_path: pathlib.Path,
    *,
    results: Optional[Dict[str, Any]],
    success: bool,
    error_msg: Optional[str] = None,
    comparison: Optional[Tuple[bool, str]] = None,
) -> None:
    """Render the final HTML file and write it to *out_path*."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = html.escape(str(out_path.name))

    if success:
        status_class = "success"
        status_message = "✅ Validation successful"
        content = "\n".join(
            [
                build_summary_section(results),
                build_population_tables(results),
                build_artifact_section(),
            ]
        )
    else:
        status_class = "error"
        status_message = f"❌ Validation failed – {html.escape(error_msg or 'unknown error')}"
        content = "<p>No additional details are shown because validation failed.</p>"

    comparison_section = build_comparison_section(comparison)

    html_text = HTML_TEMPLATE.format(
        filename=filename,
        timestamp=timestamp,
        status_class=status_class,
        status_message=status_message,
        content=content,
        comparison_section=comparison_section,
    )

    out_path.write_text(html_text, encoding="utf-8")
    print(f"HTML report written to: {out_path}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a ``rates_voltages.npz`` archive generated by "
            "RateVoltageExporter, optionally compare it with a ground‑truth file "
            "and emit an HTML report."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="Path to the .npz file to validate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("validation_report.html"),
        help="Destination HTML report file (default: validation_report.html).",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        type=pathlib.Path,
        help="Path to a ground‑truth .npz file for comparison.",
    )
    args = parser.parse_args()

    try:
        # ------------------------------------------------------------------
        # 1️⃣ Validate the input file
        # ------------------------------------------------------------------
        results = validate_rates_voltages(args.input)

        # ------------------------------------------------------------------
        # 2️⃣ Optional ground‑truth comparison
        # ------------------------------------------------------------------
        comp_result: Optional[Tuple[bool, str]] = None
        if args.ground_truth:
            comp_result = compare_npz(args.input, args.ground_truth)

        # ------------------------------------------------------------------
        # 3️⃣ Render the HTML report
        # ------------------------------------------------------------------
        write_html_report(
            args.output,
            results=results,
            success=True,
            comparison=comp_result,
        )
        return 0

    except Exception as exc:  # pragma: no cover – we want a clean error page
        # Even on failure we still produce a tiny HTML file that tells the user
        # what went wrong.
        write_html_report(
            args.output,
            results=None,
            success=False,
            error_msg=str(exc),
        )
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())