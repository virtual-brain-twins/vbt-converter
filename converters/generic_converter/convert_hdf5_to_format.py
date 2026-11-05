# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

"""
Generic master‑HDF5 → protocol‑specific conversion.

Supported protocols (built‑in):
    * amu_1          – DS‑R ``.npz`` used by the AMU group (see ``converters.exporters.rate_exporter.RateExporter``)

Adding a new protocol:
    1. Create a subclass of ``ProtocolExporter`` (see converters/exporters/…).
    2. Add it to the ``PROTOCOL_EXPORTERS`` dict below.
    3. No further changes required – the CLI will automatically expose the
       exporter’s ``configure_parser`` arguments.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path
from typing import Dict, Type

import h5py

from converters.exporters.protocol_exporter import ProtocolExporter
# Exporter registry – map the sub‑command name → concrete exporter class
from converters.exporters.rate_exporter import RateExporter
from converters.exporters.rate_voltage_exporter import RateVoltageExporter
from converters.utils.common import configure_logging
from converters.utils.validation import validate_master_bundle

PROTOCOL_EXPORTERS: Dict[str, Type[ProtocolExporter]] = {
    "uva_amu_1": RateExporter,
    "kth_amu_1": RateVoltageExporter
}


# Helper – build the top‑level argument parser
def _build_main_parser() -> argparse.ArgumentParser:
    """
    Create the main ``argparse`` parser.

    The parser has three *global* arguments (input, output‑dir, log‑level)
    and a required *sub‑command* that selects the protocol.  Each sub‑command
    is populated by the corresponding exporter’s ``configure_parser`` class
    method.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert a standard master VBT‑bundle HDF5 file into a protocol‑specific "
            "format.  At present the primary use‑case is to extract *rate* data for "
            "potential downstream visualization.  The protocol is selected by the "
            "sub‑command (e.g. `uva_amu_1`).  Protocol‑specific options are added by "
            "the exporter itself."
        )
    )

    # global options
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to the master HDF5 file produced by convert_any_to_hdf5.py",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the per‑simulation output files will be written.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of the conversion process.",
    )
    parser.add_argument(
        "-s",
        "--strict",
        action="store_true",
        help="Enable strict validation of the master bundle.",
    )

    # Sub‑parsers – one per protocol.  The *dest* argument stores the name
    # of the selected protocol (e.g. "amu_1").
    subparsers = parser.add_subparsers(
        dest="protocol",
        required=True,
        title="protocol",
        description="Target protocol – choose one of the following",
        metavar="PROTOCOL",
    )

    for name, exporter_cls in PROTOCOL_EXPORTERS.items():
        # ``add_help=False`` prevents argparse from printing two help blocks
        # (the top‑level help and the one automatically added by subparsers).
        sub = subparsers.add_parser(
            name,
            help=exporter_cls.protocol_name,
            description=exporter_cls.protocol_name,
            add_help=False,
        )
        # Exporter‑specific CLI options (e.g. ``--bin-size`` for AMU)
        exporter_cls.configure_parser(sub)

    return parser


# Main entry point
def main(argv: list[str] | None = None) -> None:
    """
    CLI driver.

    1. Parse arguments (global + protocol‑specific).
    2. Initialise logging.
    3. Open the master HDF5 file (read‑only).
    4. Instantiate the selected exporter with *only* the arguments it
       declares in its ``__init__`` signature.
    5. Run the exporter.
    """
    parser = _build_main_parser()
    args = parser.parse_args(argv)

    # Logging configuration (root logger)
    logger = configure_logging(
        __name__,
        log_file_name=Path(__file__).with_suffix(".log").name,
        level=getattr(logging, args.log_level),
    )

    # Validate input file and output directory
    master_path: Path = args.input.expanduser().resolve()
    if not master_path.is_file():
        logger.error("Master HDF5 file %s does not exist.", master_path)
        sys.exit(1)

    out_dir: Path = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the appropriate exporter class
    exporter_cls = PROTOCOL_EXPORTERS[args.protocol]

    # Build the arguments dictionary for the exporter.
    # Only keep those that the exporter’s ``__init__`` actually accepts.
    # This makes it safe to add new CLI options without touching this file.
    #   * mandatory arguments needed by the base class
    exporter_kwargs: dict = {
        "master_h5": None,
        "out_dir": out_dir,
        "log_level": getattr(logging, args.log_level),
    }

    # Add protocol‑specific arguments (everything in ``args`` that is *not*
    # a global argument and that the exporter expects).
    sig = inspect.signature(exporter_cls.__init__)
    allowed = {p.name for p in sig.parameters.values() if p.name != "self"}

    for key, value in vars(args).items():
        if key in {"input", "output_dir", "log_level", "protocol", "strict"}:
            continue
        if key in allowed:
            exporter_kwargs[key] = value

    # Open the master HDF5 file and run the exporter
    validate_master_bundle(bundle_path=master_path,
                           strict=args.strict)  # raises ValueError if the file is not a valid bundle
    with h5py.File(master_path, "r") as master_h5:
        exporter_kwargs["master_h5"] = master_h5  # replace placeholder
        exporter = exporter_cls(**exporter_kwargs)
        exporter.run()

    logger.info("All done – results are in %s", out_dir)


if __name__ == "__main__":
    main()
