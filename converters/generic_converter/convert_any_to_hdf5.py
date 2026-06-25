import argparse
import logging
import os
import sys
from pathlib import Path

import h5py

from converters.bundles.amu_bundle import AMUBundle
from converters.bundles.basal_ganglia_bundle import BasalGangliaBundle
from converters.interfaces.data_section import DataSection
from converters.interfaces.simulation_bundle import SimulationBundle
from converters.utils.bg_protocol_helper import stream_csv_as_dicts, group_by_memory
from converters.utils.common import configure_logging
from converters.utils.zip_helper import zip_directory_stream

logger = configure_logging(
    __name__,
    log_file_name=Path(__file__).with_suffix(".log").name,
    level=logging.INFO,
)


def build_master_h5(bundles: list[SimulationBundle], input_dir: Path, out_path: Path, *,
                    embed_raw: bool = True) -> None:
    """Create a master HDF5 with *one* top‑level group per simulation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        hf.attrs["format_version"] = "1.0"
        for b in bundles:
            b.to_hdf5(hf)
        if embed_raw:
            out_dir = out_path if out_path.is_dir() else out_path.parent
            raw_zip = out_dir / f"raw_data.zip"
            zip_directory_stream(input_dir, raw_zip)
            DataSection.store_external_path(
                hf,
                name="raw_bundle",
                path=raw_zip.relative_to(out_dir),
                make_link=True,
            )
            hf["raw_bundle"].attrs["format"] = "zip"
            hf["raw_bundle"].attrs["original_root"] = str(input_dir)


def discover_basal_ganglia(csv_path: Path, output_dir: Path) -> list[SimulationBundle]:
    """
    Very small wrapper around the original CSV → dict logic.
    Turn each row into a ``BasalGangliaBundle``.
    """

    # Work relative to the CSV directory – all the simulation files are
    # expected to be organised in subdirectories of the CSV's parent.
    os.chdir(csv_path.parent)
    rows = list(
        stream_csv_as_dicts(csv_path, delimiter="\t", quotechar='"', default_state="d2oe_0.00")
    )
    groups = group_by_memory(rows, key="network_state")
    bundles = []
    for state, sim_rows in groups.items():
        for row in sim_rows:
            bundles.append(BasalGangliaBundle(row, csv_path.parent, output_dir))
    return bundles


def discover_amu(root_dir: Path, output_dir: Path) -> list[SimulationBundle]:
    """
    Each immediate sub‑directory of ``root_dir`` is assumed to be a separate AMU simulation.
    """
    bundles = []
    for sub in sorted(root_dir.iterdir()):
        if sub.is_dir():
            bundles.append(AMUBundle(sub, output_dir=output_dir))
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert any supported protocol into a single portable HDF5 bundle.")
    parser.add_argument("-i", "--input", required=True,
                        help="Input directory (BasalGanglia CSV) or AMU root directory.")
    parser.add_argument("-o", "--output", required=True, help="Path to the master *.h5* file that will be created.")
    parser.add_argument("--protocol", choices=["basal", "amu"], required=True,
                        help="Which protocol to interpret the input as.")
    parser.add_argument("--no-raw-bundle", action="store_true", help="Do NOT embed the zipped original directory.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    input_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if args.protocol == "basal":
        bundles = discover_basal_ganglia(input_path, out_path if out_path.is_dir() else out_path.parent)
    else:
        bundles = discover_amu(input_path, out_path if out_path.is_dir() else out_path.parent)

    if not bundles:
        logger.error("No simulations discovered – aborting.")
        sys.exit(1)

    build_master_h5(bundles, input_path if input_path.is_dir() else input_path.parent, out_path,
                    embed_raw=not args.no_raw_bundle)
    logger.info("Finished – master file written to %s", out_path)


if __name__ == "__main__":
    main()
