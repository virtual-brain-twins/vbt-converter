from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

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


# Comparison helper
def compare_npz(file1: Path, file2: Path,
                rtol: float = 1e-04, atol: float = 1e-08) -> Tuple[bool, str]:
    """
    Consistently and optimally compares two npz files.

    Returns a tuple `(are_equal, message)`.  `are_equal` is ``True`` if the
    files match within the given tolerances, otherwise ``False`` with an
    explanatory message.
    """
    try:
        with np.load(file1) as data1, np.load(file2) as data2:
            # 1. Check if both have the same keys
            if sorted(data1.files) != sorted(data2.files):
                return False, "Key mismatch"

            for key in data1.files:
                # 2. Fast‑fail shape check
                if data1[key].shape != data2[key].shape:
                    return False, f"Shape mismatch in key: {key}"
                # 3. Optimized value comparison
                # Use allclose for floats to handle precision; array_equal for others
                if np.issubdtype(data1[key].dtype, np.floating):
                    matches = np.isclose(
                        data1[key],
                        data2[key],
                        rtol=rtol,
                        atol=atol,
                        equal_nan=True,
                    )
                    if not np.all(matches):
                        mismatch_indices = np.where(~matches)
                        first_mismatch_idx = tuple(idx[0] for idx in mismatch_indices)
                        val1 = data1[key][first_mismatch_idx]
                        val2 = data2[key][first_mismatch_idx]
                        diff = abs(val1 - val2)
                        return (
                            False,
                            f"First mismatch in '{key}' at index {first_mismatch_idx}: "
                            f"Val1={val1}, Val2={val2}, Diff={diff}",
                        )
                    # Redundant second allclose kept for strictness
                    if not np.allclose(
                            data1[key],
                            data2[key],
                            rtol=rtol,
                            atol=atol,
                            equal_nan=True,
                    ):
                        return False, f"Value mismatch (float) in key: {key}"
                else:
                    if not np.array_equal(data1[key], data2[key]):
                        return False, f"Value mismatch in key: {key}"

        return True, "Files are identical"
    except Exception as e:
        return False, f"Error during comparison: {str(e)}"


# Main comparison block
def run_comparisons(ci_root: Path,
                    converted_hdf5: Path,
                    converted_npz: Path,
                    output_dir: Path,
                    logger: logging.Logger):
    """
    Compare the produced HDF5 & NPZ files against the ground‑truth
    and write a tiny HTML report.
    """
    # Paths to the ground‑truth artefacts
    ground_truth_h5 = (
            ci_root
            / "tests"
            / "test_data"
            / "ground_truth"
            / "test_convert_output.h5"
    )
    ground_truth_npz = (
            ci_root
            / "tests"
            / "test_data"
            / "ground_truth"
            / "test_convert_output.npz"
    )
    logger.debug("Ground‑truth HDF5 file: %s", ground_truth_h5)
    logger.debug("Ground‑truth NPZ  file: %s", ground_truth_npz)

    # HDF5 comparison
    result = subprocess.run(
        ["h5diff", "-q", str(ground_truth_h5), str(converted_hdf5)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    h5_success = result.returncode == 0

    if result.returncode == 0:
        h5_message = "✅  Files are identical."
    elif result.returncode == 1:
        h5_message = (
            "❌  Files differ – see the diff below:\n"
            "stdout:\n"
            f"{result.stdout}\n"
            "stderr:\n"
            f"{result.stderr}"
        )
    elif result.returncode == 2:
        h5_message = (
            "⚠️  h5diff reported an error:\n"
            "stdout:\n"
            f"{result.stdout}\n"
            "stderr:\n"
            f"{result.stderr}"
        )
    else:  # returncode 3 or any unexpected value
        h5_message = (
            f"⚠️  Unexpected return code {result.returncode}\n"
            "stdout:\n"
            f"{result.stdout}\n"
            "stderr:\n"
            f"{result.stderr}"
        )

    logger.info(
        "HDF5 comparison result: %s, message: %s",
        "success" if h5_success else "failure",
        h5_message,
    )

    # NPZ comparison
    npz_success, npz_message = compare_npz(ground_truth_npz, converted_npz)
    npz_message = ("✅  " + npz_message) if npz_success else ("❌  " + npz_message)

    logger.info(
        "NPZ comparison result: %s, message: %s",
        "success" if npz_success else "failure",
        npz_message,
    )

    # Build the HTML report
    template_path = ci_root / "tests" / "integration" / "utils" / "hdf5_npz_comparison_template.html"
    template = template_path.read_text(encoding="utf-8")

    def _html_escape(txt: str) -> str:
        """Very small HTML‑escape (only the characters we care about here)."""
        return (
            txt.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    filled = (
        template
        .replace("{ground_truth_h5}", str(ground_truth_h5))
        .replace("{output_hdf}", str(converted_hdf5))
        .replace("{status_class_h5}", "success" if h5_success else "failure")
        .replace("{status_message_h5}", _html_escape(h5_message))

        .replace("{ground_truth_npz}", str(ground_truth_npz))
        .replace("{output_npz}", str(converted_npz))
        .replace("{status_class_npz}", "success" if npz_success else "failure")
        .replace("{status_message_npz}", _html_escape(npz_message))
    )

    report_path: Path = output_dir / "test_result_report.html"
    report_path.write_text(filled, encoding="utf-8")
    logger.info("HTML comparison report written to %s", report_path)


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
    converter_script = ci_root / "converters" / "convert_direc_to_npz.py"
    if not converter_script.is_file():
        logger.error("Converter script not found: %s", converter_script)
        sys.exit(1)

    logger.info("Using converter script: %s", converter_script)
    converted_hdf5 = None

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        output_npz = (
                Path(tmp_dir_str)
                / "data"
                / "converted"
                / "parametersweep_8s_3d_orientation_selective_12012026.npz"
        )
        output_h5 = (
                Path(tmp_dir_str)
                / "data"
                / "converted"
                / "parametersweep_8s_3d_orientation_selective_12012026.h5"
        )
        output_npz.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists
        logger.debug("Temporary npz output will be written to %s", output_npz)
        logger.debug("Temporary hdf5 output will be written to %s", output_h5)

        # ------------------------------------------------------------------
        # Assemble the command line for the converter
        # ------------------------------------------------------------------
        cmd = [
            sys.executable,  # use the same Python interpreter
            str(converter_script),
            "--input_dir",
            str(args.input_dir.resolve() / "parametersweep_8s_3d_orientation_selective_12012026"),
            "--output_file",
            str(output_npz),
            "--intermediate_hdf5",
            str(output_h5),
            "--bin_size",
            "0.1"
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
        if not output_npz.is_file():
            logger.error("Expected output file was not created: %s", output_npz)
            sys.exit(1)
        # ------------------------------------------------------------------
        if not output_h5.is_file():
            logger.error("Expected output file was not created: %s", output_h5)
            sys.exit(1)

        logger.info("Converter produced expected files: %s, %s", output_npz, output_h5)

        # ------------------------------------------------------------------
        # Copy the resulting files to the user‑specified output directory
        # ------------------------------------------------------------------
        args.output_dir.mkdir(parents=True, exist_ok=True)
        converted_hdf5 = Path(str(os.getenv("OUTPUT_HDF5_PATH"))).expanduser().resolve()
        converted_hdf5.parent.mkdir(parents=True, exist_ok=True)
        converted_npz = Path(str(os.getenv("OUTPUT_NPZ_PATH"))).expanduser().resolve()

        try:
            shutil.copy2(output_h5, converted_hdf5)
            shutil.copy2(output_npz, converted_npz)
        except OSError as exc:
            logger.error(
                "Could not copy %s → %s: %s", output_npz, converted_npz, exc
            )
            logger.error(
                "Could not copy %s → %s: %s", output_h5, converted_hdf5, exc
            )
            sys.exit(1)

        logger.info("Conversion successful – files copied to %s and %s", converted_hdf5, converted_npz)
    run_comparisons(
        ci_root=ci_root,
        converted_hdf5=converted_hdf5,
        converted_npz=converted_npz,
        output_dir=args.output_dir,
        logger=logger,
    )
    create_link_html(
        output_file=args.output_dir / "converted_hdf5_&_npz_links.html",
        logger=logger,
        links=[
            ("attachments/converted.npz", "Converted NPZ file", True, False),
            ("attachments/converted.hdf5", "Converted HDF5 file", True, False)
        ]
    )

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
