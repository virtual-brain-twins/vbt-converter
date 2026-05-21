from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.integration.utils.helper_functions import create_link_html


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
    log_file = logs_path / Path(__file__).with_suffix(".log").name

    logger = logging.getLogger(Path(__file__).with_suffix("").name)
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
    converted_hdf5 = None

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        output_hdf5 = (
                Path(tmp_dir_str)
                / "data"
                / "converted"
                / "parametersweep_8s_3d_orientation_selective_12012026.hdf5"
        )
        output_hdf5.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists
        logger.debug("Temporary HDF5 output will be written to %s", output_hdf5)

        # ------------------------------------------------------------------
        # Assemble the command line for the converter
        # ------------------------------------------------------------------
        cmd = [
            sys.executable,  # use the same Python interpreter
            str(converter_script),
            "--input_dir",
            str(args.input_dir.resolve() / "parametersweep_8s_3d_orientation_selective_12012026"),
            "--output_file",
            str(output_hdf5),
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
        if not output_hdf5.is_file():
            logger.error("Expected output file was not created: %s", output_hdf5)
            sys.exit(1)

        logger.info("Converter produced expected file: %s", output_hdf5)

        # ------------------------------------------------------------------
        # Copy the resulting HDF5 file to the user‑specified output directory
        # ------------------------------------------------------------------
        args.output_dir.mkdir(parents=True, exist_ok=True)
        converted_hdf5 = Path(str(os.getenv("OUTPUT_HDF5_PATH"))).expanduser().resolve()
        converted_hdf5.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(output_hdf5, converted_hdf5)
        except OSError as exc:
            logger.error(
                "Could not copy %s → %s: %s", output_hdf5, converted_hdf5, exc
            )
            sys.exit(1)

        logger.info("Conversion successful – file copied to %s", converted_hdf5)

    # ------------------------------------------------------------------
    # Compare with ground‑truth using h5diff and generate a tiny HTML report
    # ------------------------------------------------------------------
    ground_truth = (
            ci_root
            / "tests"
            / "test_data"
            / "ground_truth"
            / "test_convert_output.h5"
    )
    logger.debug("Ground‑truth file: %s", ground_truth)

    result = subprocess.run(
        ["h5diff", "-q", str(ground_truth), str(converted_hdf5)],
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
            "stdout:\n"
            f"{result.stdout}\n"
            "stderr:\n"
            f"{result.stderr}"
        )

    elif result.returncode == 2:
        # h5diff itself failed – report the error text
        status_message = (
            "⚠️  h5diff reported an error:\n"
            "stdout:\n"
            f"{result.stdout}\n"
            "stderr:\n"
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

    template_path = ci_root / "tests" / "integration" / "utils" / "hdf5_comparison_template.html"
    template = template_path.read_text()
    filled = (
        template
        .replace("{ground_truth_h5}", str(ground_truth))
        .replace("{output_hdf}", str(converted_hdf5))
        .replace("{status_class_h5}", "success" if success else "failure")
        .replace("{status_message_h5}", status_message,
                 )
    )
    report_path: Path = args.output_dir / "test_result_report.html"
    report_path.write_text(filled)
    create_link_html(
        output_file=args.output_dir / "converted_hdf5_link.html",
        logger=logger,
        links=[
            ("attachments/converted.hdf5", "Converted HDF5 File", True),
        ]
    )
    logger.info("Report written to %s", report_path)

    # Allure report gets burdened by input sweep directory, hence attaching it as an HTML entry with the actual directory attached to CI job
    src_sweep = (
            ci_root
            / "tests"
            / "test_data"
            / "parametersweep_8s_3d_orientation_selective_12012026"
    )
    dest_sweep_dir = (
        Path(os.getenv("INPUT_SWEEP_DIR_PATH", ""))
        .expanduser()
        .resolve()
    )
    if not src_sweep.is_dir():
        logger.error("Source sweep directory does not exist: %s", src_sweep)
        sys.exit(1)
    dest_sweep_dir.mkdir(parents=True, exist_ok=True)
    for entry in src_sweep.iterdir():
        target = dest_sweep_dir / entry.name
        try:
            # fastest when on the same filesystem
            entry.replace(target)
        except OSError:
            # fallback for cross‑filesystem moves
            shutil.move(str(entry), str(target))
    logger.info("Successfully moved contents of %s → %s", src_sweep, dest_sweep_dir)
    create_link_html(
        output_file=src_sweep / "input_sweep_dir_link.html",
        logger=logger,
        links=[
            ("attachments/parametersweep_8s_3d_orientation_selective_12012026", "Parameter Sweep Directory", False),
        ]
    )


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
