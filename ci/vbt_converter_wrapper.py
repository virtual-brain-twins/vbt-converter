from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    """
    Configure a logger that writes to a file inside the directory pointed to by
    the ``VBT_JOB_LOGS`` environment variable.  The logger also streams to
    ``stderr`` so that messages are still visible in the console (useful when the
    script is run interactively).

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logs_dir = os.getenv("VBT_JOB_LOGS")
    if not logs_dir:
        # Keep the same behaviour as the original script – abort if the required
        # environment variable is missing.
        print(
            "[ERROR] VBT_JOB_LOGS environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    logs_path = Path(logs_dir).resolve()
    logs_path.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
    log_file = logs_path / "convert_wrapper.log"

    logger = logging.getLogger("convert_wrapper")
    logger.setLevel(logging.INFO)

    # File handler (writes to the log file)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    # Stream handler (writes to stderr – mimics the original `print` behaviour)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(
        logging.Formatter("%(levelname)s – %(message)s")
    )
    logger.addHandler(stream_handler)

    logger.debug("Logging initialised – file: %s", log_file)
    return logger


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
    logger = _setup_logging()
    args = _parse_args()

    # ------------------------------------------------------------------
    # Resolve the CI_PROJECT_DIR environment variable
    # ------------------------------------------------------------------
    ci_project_dir = os.getenv("CI_PROJECT_DIR")
    if not ci_project_dir:
        logger.error("CI_PROJECT_DIR environment variable is not set.")
        sys.exit(1)

    ci_root = Path(ci_project_dir).resolve()
    logger.info("CI project root resolved to %s", ci_root)

    # ------------------------------------------------------------------
    # Build paths to the converter script and the expected HDF5 output
    # ------------------------------------------------------------------
    converter_script = ci_root / "converters" / "convert_direc_to_hdf.py"
    if not converter_script.is_file():
        logger.error("Converter script not found: %s", converter_script)
        sys.exit(1)

    logger.info("Using converter script: %s", converter_script)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        output_hdf = (
                Path(tmp_dir_str)
                / "data"
                / "converted"
                / "parametersweep_8s_3d_orientation_selective_12012026.hdf"
        )
        output_hdf.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists
        logger.debug("Temporary HDF5 output will be written to %s", output_hdf)

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
            logger.info("Running converter: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Converter exited with non‑zero status %s.", exc.returncode
            )
            sys.exit(exc.returncode)
        except OSError as exc:
            logger.error("Failed to start converter: %s", exc)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Verify that the converter produced the expected file
        # ------------------------------------------------------------------
        if not output_hdf.is_file():
            logger.error("Expected output file was not created: %s", output_hdf)
            sys.exit(1)

        logger.info("Converter produced expected file: %s", output_hdf)

        # ------------------------------------------------------------------
        # Copy the resulting HDF5 file to the user‑specified output directory
        # ------------------------------------------------------------------
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / output_hdf.name

        try:
            shutil.copy2(output_hdf, destination)
        except OSError as exc:
            logger.error(
                "Could not copy %s → %s: %s", output_hdf, destination, exc
            )
            sys.exit(1)

        logger.info("Conversion successful – file copied to %s", destination)

        # ------------------------------------------------------------------
        # Compare with ground‑truth using h5diff and generate a tiny HTML report
        # ------------------------------------------------------------------
        ground_truth = (
                args.input_dir.resolve()
                / "ground_truth"
                / "test_convert_output.h5"
        )
        logger.debug("Ground‑truth file: %s", ground_truth)

        result = subprocess.run(
            ["h5diff", "-q", str(ground_truth), str(output_hdf)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        success = result.returncode == 0
        # Build the status_message string
        # --------------------------------------------------------------
        if result.returncode == 0:
            # Files are identical – nothing else to report
            status_message = "✅  Files are identical."

        elif result.returncode == 1:
            # Files differ – attach the detailed diff
            status_message = (
                "❌  Files differ – see the diff below:\n"
                f"{result.stdout}"  # the diff output from h5diff
            )

        elif result.returncode == 2:
            # h5diff itself failed – report the error text
            status_message = (
                "⚠️  h5diff reported an error:\n"
                f"{result.stderr}"
            )

        else:  # returncode 3 or any unexpected value
            status_message = (
                f"⚠️  Unexpected return code {success}\n"
                "stdout:\n"
                f"{result.stdout}\n"
                "stderr:\n"
                f"{result.stderr}"
            )
        logger.info(
            "Comparison result: %s, status message: %s",
            "success" if success else "failure",
            status_message
        )

        template_path = ci_root / "ci" / "test_result.html"
        template = template_path.read_text()
        filled = (
            template
            .replace("{ground_truth}", str(ground_truth))
            .replace("{output_hdf}", str(output_hdf))
            .replace("{status_class}", "success" if success else "failure")
            .replace("{status_message}", status_message,
                     )
        )
        report_path: Path = args.output_dir / "test_result_report.html"
        report_path.write_text(filled)

        logger.info("Report written to %s", report_path)


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
