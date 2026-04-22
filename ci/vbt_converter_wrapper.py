from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ----------------------------------------------------------------------
# CLI handling
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """
    Parse command‑line arguments.  The interface mimics the original script
    so existing CI pipelines do not need to change.
    """
    parser = argparse.ArgumentParser(
        description="Wrapper that runs the VBT converter and copies the result."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory that contains the raw data to be converted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where the generated *.hdf* file will be copied.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Core wrapper logic
# ----------------------------------------------------------------------
def main() -> None:
    """Entry point – run the converter and copy its output."""
    args = _parse_args()

    # ------------------------------------------------------------------
    # Resolve the CI_PROJECT_DIR environment variable
    # ------------------------------------------------------------------
    ci_project_dir = os.getenv("CI_PROJECT_DIR")
    if not ci_project_dir:
        print("[ERROR] CI_PROJECT_DIR environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    ci_root = Path(ci_project_dir).resolve()

    # ------------------------------------------------------------------
    # Build paths to the converter script and the expected HDF5 output
    # ------------------------------------------------------------------
    converter_script = ci_root / "converters" / "convert_direc_to_hdf.py"
    if not converter_script.is_file():
        print(f"[ERROR] Converter script not found: {converter_script}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        output_hdf = (
                Path(tmp_dir_str)
                / "data"
                / "converted"
                / "parametersweep_8s_3d_orientation_selective_12012026.hdf"
        )
        output_hdf.parent.mkdir(parents=True, exist_ok=True)  # make sure the target folder exists

        # ------------------------------------------------------------------
        # Assemble the command line for the converter
        # ------------------------------------------------------------------
        cmd = [
            sys.executable,  # use the same Python interpreter
            str(converter_script),
            "--input_dir",
            str(args.input_dir.resolve() / "parametersweep_8s_3d_orientation_selective_12012026"),
            "--output_file",
            str(output_hdf),
        ]

        # ------------------------------------------------------------------
        # Execute the converter
        # ------------------------------------------------------------------
        try:
            print(f"[INFO] Running converter: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"[ERROR] Converter exited with non‑zero status {exc.returncode}.",
                file=sys.stderr,
            )
            sys.exit(exc.returncode)
        except OSError as exc:
            print(f"[ERROR] Failed to start converter: {exc}", file=sys.stderr)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Verify that the converter produced the expected file
        # ------------------------------------------------------------------
        if not output_hdf.is_file():
            print(
                f"[ERROR] Expected output file was not created: {output_hdf}",
                file=sys.stderr,
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # Copy the resulting HDF5 file to the user‑specified output directory
        # ------------------------------------------------------------------
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / output_hdf.name

        try:
            shutil.copy2(output_hdf, destination)
        except OSError as exc:
            print(
                f"[ERROR] Could not copy {output_hdf} → {destination}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[INFO] Conversion successful – file copied to {destination}")


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
