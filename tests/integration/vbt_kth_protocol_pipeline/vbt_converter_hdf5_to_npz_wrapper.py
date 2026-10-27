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


# Logging helpers
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


# CLI handling
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


def move_converted_hdf5(logger: logging.Logger, ci_root: Path, input_dir: Path) -> None:
    """
    Move ``ci_root/attachments/converted.hdf5`` into the folder that the
    converter expects: ``<input_dir>/attachments/converted.hdf5``.
    """
    src = ci_root / "attachments" / "bundle" / "master.h5"
    dest_dir = Path(input_dir) / "attachments"
    dst = dest_dir / "converted.hdf5"

    # Make sure we have something to move
    if not src.is_file():
        logger.error("Source HDF5 file does not exist: %s", src)
        sys.exit(1)

    # Ensure the destination folder exists (no error if it already does)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Move / rename the file
    try:
        # Path.replace works like os.rename but raises an OSError if the
        # source and destination are on different filesystems.
        # fastest when on same FS
        src.replace(dst)
    except OSError:  # (different FS)
        # Fall back to shutil.move which handles cross‑filesystem moves.
        try:
            shutil.move(str(src), str(dst))
        except OSError as exc:
            logger.error("Could not move %s → %s: %s", src, dst, exc)
            sys.exit(1)

    logger.info("Successfully moved %s → %s", src, dst)


# Core wrapper logic
def main() -> None:
    """Entry point – run the converter and copy its output."""
    logger = _setup_logging()
    args = _parse_args()

    # Resolve the CI_PROJECT_DIR environment variable
    ci_project_dir = os.getenv("CI_PROJECT_DIR")
    if not ci_project_dir:
        logger.error("CI_PROJECT_DIR environment variable is not set.")
        sys.exit(1)

    ci_root = Path(ci_project_dir).resolve()
    logger.info("CI project root resolved to %s", ci_root)

    # Move the hdf5 file to the expected location for the converter
    move_converted_hdf5(logger, ci_root, args.input_dir)

    # Build paths to the converter script and the expected HDF5 output
    converter_script = ci_root / "converters" / "generic_converter" / "convert_hdf5_to_format.py"
    if not converter_script.is_file():
        logger.error("Converter script not found: %s", converter_script)
        sys.exit(1)

    logger.info("Using converter script: %s", converter_script)
    converted_npz = None

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        output_npz_dir = (
                Path(tmp_dir_str)
                / "data"
                / "converted"
        )
        output_npz_dir.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists
        logger.debug("Temporary numpy output will be written to %s", output_npz_dir)

        # Assemble the command line for the converter
        cmd = [
            sys.executable,  # use the same Python interpreter
            str(converter_script),
            "--input",
            str(args.input_dir.resolve() / "attachments" / "converted.hdf5"),
            "--output-dir",
            str(output_npz_dir),
            "kth_amu_1",
            "--num-time-bins",
            "10000",
            "--smoothing-window",
            "10"
        ]

        # Execute the converter
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

        # Verify that the converter produced the expected file
        output_npz = output_npz_dir / "rates_voltages.npz"
        if not output_npz.is_file():
            logger.error("Expected output file was not created: %s", output_npz)
            sys.exit(1)

        logger.info("Converter produced expected file: %s", output_npz)

        # Copy the resulting npz file to the user‑specified output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        converted_npz = Path(str(os.getenv("OUTPUT_NPZ_PATH"))).expanduser().resolve()
        converted_npz.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(output_npz, converted_npz)
        except OSError as exc:
            logger.error(
                "Could not copy %s → %s: %s", output_npz, converted_npz, exc
            )
            sys.exit(1)

        logger.info("Conversion successful – file copied to %s", converted_npz)
    validate_script = ci_root / "tests" / "integration" / "vbt_kth_protocol_pipeline" / "utils" / "validate_and_report.py"
    report_path: Path = args.output_dir / "test_result_report.html"
    ground_truth = (
            ci_root /
            "tests"
            / "test_data"
            / "VBT_KTH_BG_Protocol"
            / "ground_truth"
            / "test_convert_output.npz"
    )
    if ground_truth.is_file():
        logger.info("Using ground truth file: %s", ground_truth)
    else:
        logger.error("Ground truth file not found: %s", ground_truth)
        sys.exit(1)
    cmd = [
        sys.executable,
        str(validate_script),
        "-i",
        str(converted_npz),
        "-o",
        str(report_path),
        "-g",
        str(ground_truth)
    ]

    # Execute the validator
    try:
        logger.info("Running validation: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Validation exited with non‑zero status %s.", exc.returncode
        )
        sys.exit(exc.returncode)
    except OSError as exc:
        logger.error("Failed to start validation: %s", exc)
        sys.exit(1)
    success = result.returncode == 0
    # Build the status_message string
    if result.returncode == 0:
        status_message = "✅  Validation succeeded."
    else:
        status_message = (
            f"⚠️  Unexpected return code {success}\n"
            "stdout:\n"
            f"{result.stdout}\n"
            "stderr:\n"
            f"{result.stderr}"
        )
    logger.info(
        "Validation of output npz file, result: %s, status message: %s",
        "success" if success else "failure",
        status_message
    )

    # Move the input hdf5 to the attachments
    input_file = args.input_dir.resolve() / "attachments" / "converted.hdf5"
    input_hdf5_target_path = Path(str(os.getenv("INPUT_HDF5_PATH"))).expanduser().resolve()
    input_hdf5_target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(input_file, input_hdf5_target_path)
    except OSError as exc:
        logger.error(
            "Could not move %s → %s: %s", input_file, input_hdf5_target_path, exc
        )
        sys.exit(1)
    logger.info("Input HDF5 file moved from %s → %s", input_file, input_hdf5_target_path)
    create_link_html(
        output_file=ci_root / "input" / "input_hdf5_link.html",
        logger=logger,
        links=[
            ("attachments/input.hdf5", "Input HDF5 File", True, False)
        ]
    )
    logger.info("Report written to %s", report_path)


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
