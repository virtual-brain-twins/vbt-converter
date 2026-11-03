# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import html
import json
import logging
import os
import pathlib
from pathlib import Path
from typing import List, Tuple, Sequence, Any, Dict
from typing import Union

import h5py
import numpy as np

LinkSpec = Union[
    Tuple[str, str],  # (path, link_text) → uses default `file`
    Tuple[str, str, bool],  # (path, link_text, is_file, is_path_absolute)
    Tuple[str, str, bool, bool],  # (path, link_text, is_file, is_path_absolute)
]


def create_link_html(
        *,
        output_file: Path,
        logger: logging.Logger,
        links: Sequence[LinkSpec] | None = None,
        default_is_file: bool = True,
) -> None:
    """
    Create a minimal HTML document that contains one or more links to
    Allure artefacts.

    Parameters
    ----------
    output_file:
        Destination path (including filename) for the HTML file.
        Parent directories are created automatically.

    logger:
        A ``logging.Logger`` instance that will receive an INFO message
        once the file has been written.

    links:
        A sequence of ``(path, link_text)`` **or**
        ``(path, link_text, is_file, is_path_absolute)`` tuples.

        * ``path`` is interpreted **relative to** the Allure
          ``/artifacts/<segment>/`` directory.
        * If a 4‑tuple is supplied, ``is_file`` determines whether the URL
          contains the ``file`` segment (``True``) or the ``browse`` segment
          (``False``).  If omitted, the value of ``default_is_file`` is used.
        * The ``is_path_absolute`` flag determines whether the path is
          interpreted as absolute (``True``) or relative (``False``).

        If ``None`` (the default) the function falls back to the *converted
        artefacts* use‑case and generates two links:

            converted.hdf5 → “Converted HDF5 File”
            converted.npz → “Generated NPZ File”

        The default URLs are built with the path
        ``file/attachments/<filename>`` (i.e. they always use the **file**
        segment).

    default_is_file:
        The segment to use for links that do **not** specify their own
        ``is_file`` flag.  The historic default was ``True`` (i.e. ``file``).

    Raises
    ------
    RuntimeError
        If the required environment variable ``VBT_ALLURE_LINK_JOB`` is not
        defined.

    Notes
    -----
    * The function does **not** perform any network request – it merely
      builds the URLs.
    * The generated HTML is intentionally simple but includes a small
      style block to make the output readable in a browser.
    """
    # ------------------------------------------------------------------- #
    # Resolve the base URL from the environment
    # ------------------------------------------------------------------- #
    base_url = os.getenv("VBT_ALLURE_LINK_JOB")
    if not base_url:
        raise RuntimeError(
            "Environment variable 'VBT_ALLURE_LINK_JOB' is not set."
        )
    # Normalise – we never want a trailing slash because we will add one
    # ourselves when joining parts.
    base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------- #
    # Build the collection of links
    # ------------------------------------------------------------------- #
    if links is None:
        # Default – two “converted” artefacts.  They always live under the
        #   /artifacts/file/converted/ directory.
        default_paths = (
            ("attachments/converted.hdf5", "Converted HDF5 File"),
            ("attachments/converted.npz", "Generated NPZ File"),
        )
        # ``default_is_file`` is deliberately ignored for the historic
        # default case – the original helper always used the “file” segment.
        link_data: List[Tuple[str, str, bool, bool]] = [
            (rel_path, text, True, False) for rel_path, text in default_paths
        ]
    else:
        # Normalise the user‑supplied collection to the 3‑tuple form.
        link_data = []
        for spec in links:
            if len(spec) == 2:
                rel_path, text = spec  # type: ignore[assignment]
                is_file = default_is_file
                is_path_absolute = False
            elif len(spec) == 3:
                rel_path, text, is_file = spec
                is_path_absolute = False
            elif len(spec) == 4:
                rel_path, text, is_file, is_path_absolute = spec  # type: ignore[assignment]
                if not isinstance(is_file, bool):
                    raise TypeError(
                        f"is_file flag must be bool, got {type(is_file)!r} "
                        f"for link {spec!r}"
                    )
                if not isinstance(is_path_absolute, bool):
                    raise TypeError(
                        f"is_path_absolute flag must be bool, got {type(is_path_absolute)!r} "
                        f"for link {spec!r}"
                    )
            else:
                raise ValueError(
                    f"Link spec must be a 2‑ or 4‑tuple, got {spec!r}"
                )
            link_data.append((rel_path, text, is_file, is_path_absolute))

    # ------------------------------------------------------------------- #
    # Helper that turns a (relative_path, is_file, is_path_absolute: bool) pair into a full URL.
    # ------------------------------------------------------------------- #
    def _full_url(rel_path: str, is_file: bool, is_path_absolute: bool) -> str:
        # If the caller gave us an absolute URL/path we simply hand it back.
        # Otherwise strip any leading “/” (so url‑joining works) and prepend the
        # appropriate artifact segment.
        return (
            rel_path
            if is_path_absolute
            else f"{base_url}/artifacts/{'file' if is_file else 'browse'}/{rel_path.lstrip('/')}"
        )

    # ------------------------------------------------------------------- #
    # Build the <p>…</p> blocks – one per link.
    # ------------------------------------------------------------------- #
    link_paragraphs = "\n".join(
        f'        <p class="artifact-list">• '
        f'<a href="{_full_url(path, is_file, is_path_absolute)}" target="_blank" '
        f'rel="noopener noreferrer">{text}</a></p>'
        for path, text, is_file, is_path_absolute in link_data
    )

    # ------------------------------------------------------------------- #
    # Final HTML payload
    # ------------------------------------------------------------------- #
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Artifact Links</title>
    <style>
        body {{ font-family: Arial, Helvetica, sans-serif; line-height: 1.6; margin: 2rem; }}
        a   {{ color:#0066CC; text-decoration:none; }}
        a:hover {{ text-decoration:underline; }}
        .artifact-list {{ margin-top: 0.5rem; }}
    </style>
</head>
<body>
    <h1>Artifact Links</h1>
{link_paragraphs}
</body>
</html>
"""

    # ------------------------------------------------------------------- #
    # Persist the file (parents are created automatically)
    # ------------------------------------------------------------------- #
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding="utf-8")

    logger.info(f"HTML file created at: {output_file.resolve()}")

MAX_CHILDREN = 5               # How many members of a group are rendered
TRUNCATE_TEXT = "… {remaining:d} more item(s)"  # Shown when a group is cut


# --------------------------------------------------------------------------- #
# Tiny utility helpers
# --------------------------------------------------------------------------- #
def _escape(text: Any) -> str:
    """HTML‑escape anything that is not ``None``."""
    return html.escape(str(text)) if text is not None else ""


def _is_small_dataset(ds: h5py.Dataset) -> bool:
    """True if the dataset is a scalar or a single element – cheap to read."""
    return ds.shape == () or ds.size == 1


# --------------------------------------------------------------------------- #
# Attribute rendering
# --------------------------------------------------------------------------- #
def _format_attrs(attrs: h5py.AttributeManager) -> str:
    """
    Return an HTML <dl class="attrs"> block with *user‑defined* attributes.
    Empty attribute collections produce an empty string.

    The ``attrs`` mapping now yields ``key -> (value, unit)`` tuples.
    If a unit is supplied it is shown after the value in parentheses.
    """
    if not attrs:
        return ""

    lines: List[str] = ['<dl style="display:none;" class="attrs">']
    for key, raw_val in attrs.items():
        # Detect the optional unit.  ``raw_val`` can be either:
        #   * a plain value, e.g.   42
        #   * a 2‑tuple/list, e.g.   (42, "Hertz")
        # Anything else is treated as a plain value.
        if isinstance(raw_val, (tuple, list)) and len(raw_val) == 2:
            val, unit = raw_val
        else:
            val, unit = raw_val, None
        try:
            # ``tolist`` gives native Python objects; if it fails we fall back.
            native = val.tolist() if hasattr(val, "tolist") else val
            val_str = json.dumps(native)
        except Exception:
            # Fallback – ``repr`` is safe for everything.
            val_str = _escape(repr(val))
        dd_content = f"Value: {val_str} Unit: {_escape(unit)}" if unit is not None else val_str
        # Emit one <dt>/<dd> pair.
        lines.append(f"<dt>{_escape(key)}:</dt><dd>{dd_content}</dd>")

    lines.append("</dl>")
    lines.append("</dl>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset rendering
# --------------------------------------------------------------------------- #
def _dataset_meta(ds: h5py.Dataset) -> str:
    """Collect dtype, shape, chunking and compression information."""
    parts = [
        f"dtype: {_escape(ds.dtype)}",
        f"shape: {_escape(ds.shape)}",
    ]
    if ds.chunks:
        parts.append(f"chunks: {_escape(ds.chunks)}")
    if ds.compression:
        parts.append(f"compression: {_escape(ds.compression)}")
    return ", ".join(parts)


def _render_dataset(name: str, ds: h5py.Dataset) -> str:
    """HTML for a single dataset – metadata + optional scalar value."""
    meta = _dataset_meta(ds)

    # -------------------------------------------------------------------- #
    #   Value handling – only for scalars / single‑element arrays
    # -------------------------------------------------------------------- #
    if _is_small_dataset(ds):
        try:
            value = ds[()]                     # read the scalar
            # ``tolist`` handles NumPy scalars nicely
            native = value.tolist() if hasattr(value, "tolist") else value
            value_str = json.dumps(native)
        except Exception as exc:                # very defensive
            value_str = f"<i>error reading value: {html.escape(str(exc))}</i>"
        value_html = f"<div class=\"meta\">value: {_escape(value_str)}</div>"
    else:
        # For anything larger we *never* read the data.
        value_html = "<div class=\"meta\"><i>array data omitted (size > 1)</i></div>"
    # -------------------------------------------------------------------- #

    attr_html = _format_attrs(ds.attrs)

    return (
        f'<li>'
        f'<span class="toggle">Dataset: {_escape(name)} ({meta})</span>'
        f'{value_html}'
        f'{attr_html}'
        f'</li>'
    )

def _render_group(name: str, grp: h5py.Group, depth: int = 0) -> str:
    """
    Render a group and (recursively) its children.

    *Only the first ``MAX_CHILDREN`` members are expanded*; the rest are
    represented by a short placeholder.  ``depth`` is only used for
    debugging / future extensions – it is not part of the HTML output.
    """
    attr_html = _format_attrs(grp.attrs)

    # -------------------------------------------------------------------- #
    #   Fast child enumeration – we obtain the names first, then slice.
    # -------------------------------------------------------------------- #
    all_names = list(grp.keys())                      # cheap
    total = len(all_names)

    # The slice that we actually render.
    render_names = all_names[:MAX_CHILDREN]

    children_html: List[str] = []
    for child_name in render_names:
        child_obj = grp[child_name]                  # cheap – only metadata
        if isinstance(child_obj, h5py.Group):
            children_html.append(_render_group(child_name, child_obj, depth + 1))
        elif isinstance(child_obj, h5py.Dataset):
            children_html.append(_render_dataset(child_name, child_obj))
        else:
            children_html.append(
                f"<li>Unsupported object: {_escape(child_name)} ({type(child_obj)})</li>"
            )

    if total > MAX_CHILDREN:
        remaining = total - MAX_CHILDREN
        placeholder = (
            f'<li class="truncated"><i>{TRUNCATE_TEXT.format(remaining=remaining)}</i></li>'
        )
        children_html.append(placeholder)

    children_block = (
        f'<ul style="display:none;">' + "\n".join(children_html) + "</ul>"
        if children_html else ""
    )

    return (
        f'<li class="expanded">'
        f'<span class="toggle">Group: {_escape(name)}</span>'
        f'{attr_html}'
        f'{children_block}'
        f'</li>'
    )

def _build_hdf5_tree_html(h5_path: pathlib.Path) -> str:
    """
    Open *h5_path* and return a ready‑to‑inject HTML fragment.

    The fragment is a single <ul> containing the root members of the file.
    All heavy‑weight I/O (reading large datasets, traversing massive groups)
    is avoided – groups are truncated after ``MAX_CHILDREN`` items and only
    scalar values are materialised.
    """
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(str(h5_path), "r") as root:
        root_children: List[str] = []
        for name, obj in root.items():
            if isinstance(obj, h5py.Group):
                root_children.append(_render_group(name, obj))
            elif isinstance(obj, h5py.Dataset):
                root_children.append(_render_dataset(name, obj))
            else:
                root_children.append(
                    f"<li>Unsupported object: {_escape(name)} ({type(obj)})</li>"
                )
        # ``''.join`` is a tad faster than ``+`` inside a loop.
        return f"<ul>{''.join(root_children)}</ul>"