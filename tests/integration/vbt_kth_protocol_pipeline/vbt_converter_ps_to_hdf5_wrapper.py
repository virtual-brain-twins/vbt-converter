from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.integration.utils.helper_functions import create_link_html, _build_hdf5_tree_html


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

    # Build paths to the converter script and the expected HDF5 output
    converter_script = ci_root / "converters" / "generic_converter" / "convert_any_to_hdf5.py"
    if not converter_script.is_file():
        logger.error("Converter script not found: %s", converter_script)
        sys.exit(1)

    logger.info("Using converter script: %s", converter_script)
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        output_bundle = (
                Path(tmp_dir_str)
                / "output"
                / "bundle"
        )
        output_master_h5 = (
                Path(output_bundle)
                / "master.h5"
        )
        output_master_h5.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists
        logger.debug("Temporary bundle directory: %s", output_bundle)

        # Assemble the command line for the converter
        cmd = [
            sys.executable,  # use the same Python interpreter
            str(converter_script),
            "--input",
            str(args.input_dir.resolve() / "VBT_KTH_BG_Protocol" / "run_list.csv"),
            "--output",
            str(output_master_h5),
            "--protocol",
            "kth"
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

        if not output_master_h5.is_file():
            logger.error("Expected output file was not created: %s", output_master_h5)
            sys.exit(1)

        logger.info("Converter produced expected file: %s", output_master_h5)

        # Copy the results to the user‑specified output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        converted_bundle = Path(str(os.getenv("OUTPUT_BUNDLE_PATH"))).expanduser().resolve()
        converted_bundle.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copytree(output_bundle, converted_bundle, dirs_exist_ok=True)
        except OSError as exc:
            logger.error(
                "Could not copy %s → %s: %s", output_bundle, converted_bundle, exc
            )
            sys.exit(1)

        logger.info("Conversion successful – directory copied to %s", converted_bundle)
    # ------------------------------------------------------------------
    # Compare with ground‑truth using h5diff and generate a tiny HTML report
    # ------------------------------------------------------------------
    ground_truth = (
            Path(str(os.getenv("VBT_INPUT_PATH")))
            / "ground_truth"
            / "master.h5"
    )
    output_master_h5 = (
            Path(converted_bundle)
            / "master.h5"
    )
    logger.debug("Ground‑truth file: %s", ground_truth)
    logger.debug("Output file: %s", output_master_h5)

    result = subprocess.run(
        ["h5diff", "-q", "--exclude-path", "/RawBundle", str(ground_truth), str(output_master_h5)],
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
    html_tree = _build_hdf5_tree_html(output_master_h5)
    filled = (
        template
        .replace("{ground_truth_h5}", str(ground_truth))
        .replace("{output_hdf}", str(output_master_h5))
        .replace("{status_class_h5}", "success" if success else "failure")
        .replace("{status_message_h5}", status_message)
        .replace("{hdf5_tree_html}", html_tree)
    )
    report_path: Path = args.output_dir / "test_result_report.html"
    report_path.write_text(filled)
    create_link_html(
        output_file=args.output_dir / "converted_hdf5_link.html",
        logger=logger,
        links=[
            ("https://vbt-minio.fz-juelich.de/console/browser/vbt-gitlab-cache/project%2F8681%2Fvbt-kth-bg-protocol-cache-bundle-protected",
             "Converted output bundle including the master HDF5", True, True)
        ]
    )
    logger.info("Report written to %s", report_path)

    # Allure report gets burdened by input sweep directory, hence attaching it as an HTML entry with the actual directory attached to CI job
    # Delete every file in input directory, since its heavy for the final allure report
    for entry in Path(str(os.getenv("VBT_INPUT_PATH"))).expanduser().resolve().iterdir():
        # shutil.rmtree handles both files and directories (and follows symlinks safely)
        shutil.rmtree(entry, ignore_errors=True)  # ignore_errors to keep it fast & non‑blocking
    src_sweep = Path(str(os.getenv("VBT_INPUT_PATH")))
    create_link_html(
        output_file=src_sweep / "input_sweep_dir_link.html",
        logger=logger,
        links=[
            ("https://vbt-minio.fz-juelich.de/console/browser/vbt-gitlab-cache/project%2F8681%2Fvbt-kth-bg-protocol-cache-protected",
             "Input Parameter Sweep Directory", False, True)
        ]
    )


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
