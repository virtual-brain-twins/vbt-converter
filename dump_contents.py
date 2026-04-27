import html
import os
from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _safe_str(value: object) -> str:
    """Return a short, HTML‑escaped string representation.

    Numbers, tuples, numpy scalars, etc. are converted with ``str()``.
    ``bytes`` objects are shown as ``b'...'`` (truncated if very long).
    ``None`` becomes an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        # Show only the first 30 bytes to keep the table readable
        preview = value[:30]
        suffix = "…" if len(value) > 30 else ""
        return html.escape(f"{preview!r}{suffix}")
    # ``html.escape`` also works for normal strings
    return html.escape(str(value))


def create_hdf5_html_report(
        hdf5_path: os.PathLike | str,
        output_html: os.PathLike | str = "index.html",
) -> Path:
    """
    Walk through *hdf5_path* and write a tiny HTML report that lists
    every object (group or dataset) together with its shape and data type.

    Parameters
    ----------
    hdf5_path : str or pathlib.Path
        Path to the HDF5 file that should be inspected.
    output_html : str or pathlib.Path, default="index.html"
        Where to write the generated HTML file.

    Returns
    -------
    pathlib.Path
        The full path to the written HTML file (convenient for tests).

    Notes
    -----
    * The function never raises because of an unexpected HDF5 object.
      Problems are logged to ``stderr`` and the table entry falls back
      to ``-`` placeholders.
    * All textual content is escaped with :func:`html.escape` to avoid
      broken HTML when dataset names contain ``<`` or ``&``.
    """
    hdf5_path = Path(hdf5_path)
    output_html = Path(output_html)

    # -----------------------------------------------------------------
    # 1️⃣ Gather rows
    # -----------------------------------------------------------------
    rows: List[str] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:  # noqa: D401
        """
        Callback used by ``h5py.File.visititems``.  It extracts:
        * the absolute *name* inside the file,
        * object **type** (Group/Dataset),
        * **shape** (or “‑” for groups),
        * **dtype** (or “‑” for groups).
        """
        # -------------------------------------------------------------
        # Determine type string and CSS class
        # -------------------------------------------------------------
        if isinstance(obj, h5py.Dataset):
            obj_type = "Dataset"
            css_class = "type-dataset"
        elif isinstance(obj, h5py.Group):
            obj_type = "Group"
            css_class = "type-group"
        else:  # Catch‑all for anything else (e.g. soft links)
            obj_type = type(obj).__name__
            css_class = "type-unknown"

        # -------------------------------------------------------------
        # Shape & dtype – both may raise if the object is weird.
        # -------------------------------------------------------------
        try:
            shape = _safe_str(obj.shape) if isinstance(obj, h5py.Dataset) else "-"
        except Exception as exc:  # pragma: no cover – defensive
            shape = f"✖ {_safe_str(exc)}"

        try:
            dtype = _safe_str(obj.dtype) if isinstance(obj, h5py.Dataset) else "-"
        except Exception as exc:  # pragma: no cover
            dtype = f"✖ {_safe_str(exc)}"

        # -------------------------------------------------------------
        # Build the <tr> … </tr> fragment
        # -------------------------------------------------------------
        rows.append(
            f"""
            <tr>
                <td>{html.escape(name)}</td>
                <td class="{css_class}">{obj_type}</td>
                <td>{shape}</td>
                <td>{dtype}</td>
            </tr>
            """
        )

    # -----------------------------------------------------------------
    # 2️⃣ Open the file and walk it
    # -----------------------------------------------------------------
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5:
        h5.visititems(visitor)

    # -----------------------------------------------------------------
    # 3️⃣ Assemble final HTML
    # -----------------------------------------------------------------
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HDF5 Report – {html.escape(hdf5_path.name)}</title>
    <style>
        body {{font-family: system-ui, sans-serif; margin: 2rem; background:#f8f9fa;}}
        h2   {{color:#212529;}}
        table {{width:100%; border-collapse:collapse; background:#fff;
                box-shadow:0 2px 8px rgba(0,0,0,.1);}}
        th, td {{padding:.75rem 1rem; text-align:left; border-bottom:1px solid #dee2e6;}}
        th {{background:#007bff; color:#fff;}}
        .type-group    {{color:#6c757d; font-weight:600;}}
        .type-dataset  {{color:#28a745; font-weight:600;}}
        .type-unknown  {{color:#dc3545; font-weight:600;}}
        tr:hover {{background:#f1f3f5;}}
    </style>
</head>
<body>
    <h2>Contents of {html.escape(hdf5_path.name)}</h2>
    <table>
        <thead>
            <tr>
                <th>Path</th>
                <th>Type</th>
                <th>Shape</th>
                <th>Data type</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
</body>
</html>
"""

    # -----------------------------------------------------------------
    # 4️⃣ Write the file
    # -----------------------------------------------------------------
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_template, encoding="utf-8")

    print(f"✅ HTML report written to {output_html}")  # side‑effect, useful for CLI use
    return output_html


def dump_all_h5_contents(file_path, output_dir="h5_visual_dump"):
    """Recursively saves all datasets in an H5 file as images or text."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(file_path, 'r') as f:
        def visitor(name, obj):
            # We only process Datasets (the actual data files)
            if isinstance(obj, h5py.Dataset):
                try:
                    data = obj[()]
                    clean_name = name.replace("/", "_")

                    # Case 1: 2D or 3D Data -> Save as Image
                    if len(data.shape) >= 2:
                        plt.figure(figsize=(8, 6))
                        # Render 3D as a slice or RGB if appropriate
                        display_data = data if len(data.shape) == 2 else data[0]
                        plt.imshow(display_data, cmap='viridis')
                        plt.title(f"Dataset: {name}\nShape: {data.shape}")
                        plt.colorbar()
                        plt.savefig(os.path.join(output_dir, f"{clean_name}.png"))
                        plt.close()

                    # Case 2: 1D Data -> Save as Line Plot
                    elif len(data.shape) == 1 and np.issubdtype(data.dtype, np.number):
                        plt.figure()
                        plt.plot(data)
                        plt.title(f"Signal: {name}")
                        plt.savefig(os.path.join(output_dir, f"{clean_name}_plot.png"))
                        plt.close()

                    # Case 3: Scalars or Strings -> Save to Text Log
                    else:
                        with open(os.path.join(output_dir, "metadata_dump.txt"), "a") as log:
                            log.write(f"Path: {name} | Value: {data}\n")
                except Exception as e:
                    print(f"Could not dump {name}: {e}")

        # Start the recursive walk
        f.visititems(visitor)
    print(f"Dump complete. Check the '{output_dir}' folder.")


# Run the dump
#dump_all_h5_contents('test_convert_output.h5')
