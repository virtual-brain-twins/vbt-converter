from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple, Sequence
from typing import Union

LinkSpec = Union[
    Tuple[str, str],  # (path, link_text) → uses default `file`
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
            elif len(spec) == 3:
                rel_path, text, is_file = spec  # type: ignore[assignment]
                if not isinstance(is_file, bool):
                    raise TypeError(
                        f"is_file flag must be bool, got {type(is_file)!r} "
                        f"for link {spec!r}"
                    )
            else:
                raise ValueError(
                    f"Link spec must be a 2‑ or 3‑tuple, got {spec!r}"
                )
            link_data.append((rel_path, text, is_file))

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
