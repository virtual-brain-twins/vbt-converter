"""Basal ganglia conversion utilities.

This module provides helper functions that translate the original Snudda‑style
JSON and HDF5 stimulus specifications into the new `converters.schema` data
model.  The public entry point is :func:`build_bg_master_bundle`, which builds a
complete :class:`~converters.schema.master_bundle.MasterBundle` from a CSV file
produced by the original basal‑ganglia pipeline.

Typical usage
-------------
    >>> bundle = build_bg_master_bundle(
    ...     input_csv_path=Path("run.csv"),
    ...     embed_raw=True,
    ...     embed_raw_path=Path("raw.zip"),
    ...     output_dir=Path("out"),
    ...     protocol_version="1.0",
    ...     protocol_info_path=Path("protocol_info.pdf"),
    ... )
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from converters.schema.global_section import GlobalSection
from converters.schema.master_bundle import MasterBundle
from converters.schema.neuron import Neuron
from converters.schema.population import Population
from converters.schema.protocol_group import ProtocolGroup
from converters.schema.raw_bundle_group import RawBundleGroup
from converters.schema.simulation import Simulation
from converters.schema.simulation_parameters import SimulationParameters
from converters.schema.spike_map import SpikeMap
from converters.schema.standard_format_group import StandardFormatGroup
from converters.schema.stimuli_group import StimuliGroup
from converters.schema.stimulus import Stimulus
from converters.utils.bg_protocol_helper import stream_csv_as_dicts

# Helper regular expression for simple arithmetic strings.
_ARITHMETIC_RE = re.compile(
    r"""
    ^\s*
    (?P<scale>\*?)                # optional leading '*'
    (?P<expr>[-+/*\d.eE]+)        # a very small arithmetic expression
    \s*$
    """,
    re.VERBOSE,
)


def _eval_arith_string(val: Any) -> Any:
    """Convert a string that looks like a simple arithmetic expression to a number.

    The function recognises patterns such as ``"*1.25"``, ``"30/4000e-6"``, or
    ``"2e-05"`` and evaluates the arithmetic part, returning a ``float``.  If the
    input is not a string or does not match the expected pattern, the original
    value is returned unchanged.

    Args:
        val: The value to evaluate.  Usually a ``str`` but any type is accepted.

    Returns:
        The evaluated number if the input matched the arithmetic pattern,
        otherwise the original ``val``.

    Raises:
        TypeError: If ``val`` is a string but contains characters that make the
            regular expression match fail (unlikely in normal operation).
        Exception: Any exception raised by ``eval`` is caught and results in the
            original string being returned.
    """
    if not isinstance(val, str):
        return val

    m = _ARITHMETIC_RE.match(val)
    if not m:
        return val

    expr = m.group("expr")
    try:
        # ``eval`` is safe because the regex guarantees that only numbers,
        # decimal points, exponent notation and the operators + - * / are present.
        number = eval(expr, {"__builtins__": {}}, {})
    except Exception:  # pragma: no cover
        return val

    # The optional leading '*' is ignored for the numeric result.
    return number


def _parse_cortical_sub_stimulus(
        sub_dict: Dict[str, Any],
) -> Tuple[Stimulus, Dict[str, Any]]:
    """Parse a *cortical* sub‑stimulus description.

    A cortical block contains a piece‑wise definition of a stimulus (start times
    and frequencies).  The function builds a :class:`Stimulus` instance from the
    ``start`` and ``frequency`` entries and returns the remaining scalar values
    as HDF5 attributes.

    Args:
        sub_dict: The dictionary extracted from the JSON configuration that
            describes a single cortical sub‑stimulus.

    Returns:
        A tuple ``(stimulus, attrs)`` where ``stimulus`` is a fully populated
        :class:`Stimulus` object and ``attrs`` is a mapping of any additional
        fields that should be stored as HDF5 attributes.

    Raises:
        KeyError: If required keys (``"start"`` or ``"frequency"``) are missing.
        ValueError: If ``sub_dict`` contains non‑numeric data that cannot be
            converted to ``float``.
    """
    # Convert the piece‑wise description.
    limits = np.asarray(sub_dict.get("start", []), dtype=np.float64)
    values = np.asarray(sub_dict.get("frequency", []), dtype=np.float64)

    stimulus = Stimulus(
        limits=limits,
        values=values,
        stim_type=sub_dict.get("stim_type", "piecewise_constant"),
    )

    # Keys that belong to the Stimulus itself and must not become attributes.
    stim_keys = {
        "start",
        "end",
        "frequency",
        "stim_type",
        "limits",
        "values",
        "max_value",
    }

    attrs: Dict[str, Any] = {}
    for k, v in sub_dict.items():
        if k in stim_keys:
            continue
        attrs[k] = _eval_arith_string(v)

    return stimulus, attrs


def _parse_generic_sub_stimulus(
        sub_name: str, sub_dict: Dict[str, Any]
) -> Tuple[Stimulus, Dict[str, Any]]:
    """Parse a sub‑stimulus that does not contain a piece‑wise definition.

    For sources such as ``virtual_neuron`` the function creates an *empty*
    :class:`Stimulus` object (zero‑length ``limits`` and ``values``) and stores
    the entire ``sub_dict`` as attributes after attempting to evaluate any
    arithmetic strings.

    Args:
        sub_name: Name of the sub‑stimulus (currently unused but kept for API
            compatibility).
        sub_dict: Dictionary describing the sub‑stimulus.

    Returns:
        A tuple ``(stimulus, attrs)`` where ``stimulus`` is an empty
        :class:`Stimulus` and ``attrs`` is the dictionary of attributes.

    Raises:
        None.
    """
    stimulus = Stimulus(
        limits=np.empty(0, dtype=np.float64),
        values=np.empty(0, dtype=np.float64),
        stim_type="piecewise_constant",
    )

    attrs = {k: _eval_arith_string(v) for k, v in sub_dict.items()}
    return stimulus, attrs


def _parse_input_config(input_config: Path) -> StimuliGroup:
    """Parse a JSON stimulus description and convert it to a :class:`StimulusSet`.

    The JSON file follows the legacy Snudda format where each top‑level key is a
    population name that maps to a set of sub‑stimuli.  The function builds the
    hierarchical :class:`StimulusSet` object that later can be written to HDF5.

    Args:
        input_config: Path to the JSON file containing the stimulus description.

    Returns:
        A fully populated :class:`StimulusSet` instance.

    Raises:
        FileNotFoundError: If ``input_config`` does not point to an existing file.
        json.JSONDecodeError: If the file contents are not valid JSON.
        KeyError: If the JSON structure is missing expected keys.
    """
    if not input_config.is_file():
        raise FileNotFoundError(f"Unable to locate input config: {input_config}")

    with input_config.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    populations: List[StimuliGroup] = []

    for pop_name, pop_content in cfg.items():
        sub_stimuli: Dict[str, Stimulus] = {}
        sub_attrs: Dict[str, Dict[str, Any]] = {}

        for sub_name, sub_dict in pop_content.items():
            if "start" in sub_dict and "frequency" in sub_dict:
                stimulus, attrs = _parse_cortical_sub_stimulus(sub_dict)
            else:
                stimulus, attrs = _parse_generic_sub_stimulus(sub_name, sub_dict)

            sub_stimuli[sub_name] = stimulus
            sub_attrs[sub_name] = attrs

        populations.append(
            StimuliGroup(
                stimuli_map=sub_stimuli, attr_map=sub_attrs
            )
        )

    return StimuliGroup(stimuli_map={}, attr_map={})


def _populations_from_output_spikes(h5_path: Path) -> List[Population]:
    """Read the classic Snudda output‑spikes HDF5 layout and create ``Population`` objects.

    The HDF5 file contains three top‑level groups: ``meta_data``, ``time`` and
    ``neurons``.  For each neuron (identified by its global ID) the function
    extracts the spike times and groups neurons by the population name stored in
    ``meta_data/name``.

    Args:
        h5_path: Path to the HDF5 file containing spike data.

    Returns:
        A list of :class:`Population` objects, each containing a collection of
        :class:`Neuron` instances for that population.

    Raises:
        FileNotFoundError: If ``h5_path`` does not exist.
        OSError: If the HDF5 file cannot be opened (corrupted, permission error, …).
        KeyError: If the expected groups or datasets are missing in the file.
    """
    with h5py.File(h5_path, "r") as f:
        population_name = f["meta_data"]["name"][()]
        neuron_grp = f["neurons"]
        populations: Dict[str, List[Neuron]] = {}

        for gid_str, ngrp in neuron_grp.items():
            spikes = ngrp["spikes"][()]  # shape (1, N_spikes)
            times = spikes.squeeze().astype(np.float64)

            # ``population_name`` is an array where the index corresponds to the GID.
            pop_name = population_name[int(gid_str)].decode("utf-8")
            populations.setdefault(pop_name, []).append(
                Neuron(index=int(gid_str), times=times)
            )

        return [Population(name, neurons) for name, neurons in populations.items()]


def build_bg_master_bundle(
        input_csv_path: Path,
        embed_raw: bool,
        embed_raw_path: Path,
        output_dir: Path,
        protocol_version: str,
        protocol_info_path: Path
) -> MasterBundle:
    """Build a :class:`MasterBundle` from a basal‑ganglia CSV file.

    The function reads the CSV generated by ``discover_basal_ganglia``, parses
    each row, loads the associated stimulus JSON and output‑spike HDF5 files, and
    assembles the data into the new schema representation.  If ``embed_raw`` is
    ``True``, a ``RawBundleGroup`` is added and linked to ``embed_raw_path``.

    Args:
        input_csv_path: Path to the CSV file describing the simulation runs.
        embed_raw: Whether to embed the raw data bundle in the output.
        embed_raw_path: Path to the raw data archive (used only when ``embed_raw``
            is ``True``).
        output_dir: Directory where the final bundle will be written.
        protocol_version: Protocol version to use.
        protocol_info_path: Path to the file containing protocol information.

    Returns:
        A fully populated :class:`MasterBundle` instance ready for serialisation.

    Raises:
        FileNotFoundError: If ``input_csv_path`` does not exist.
        json.JSONDecodeError: Propagated from :func:`_parse_input_config` when the
            stimulus JSON is malformed.
        OSError: If any of the underlying HDF5 files cannot be opened.
        KeyError: If required columns are missing from the CSV rows.
    """
    # Change working directory to the location of the CSV so that relative paths
    # inside the file resolve correctly.
    os.chdir(input_csv_path.parent)

    csv_rows = list(
        stream_csv_as_dicts(
            input_csv_path,
            delimiter="\t",
            quotechar='"',
            default_state="d2oe_0.00",
        )
    )

    # Global section – reuse high‑level fields from each CSV row.
    global_section = GlobalSection(
        metadata_mapping={"network_state": ("network_state", None), "duration": ("duration", None)},
        experiment_info="Basal‑Ganglia simulation",
    )

    runs: List[Simulation] = []
    for row in csv_rows:
        # Simulation parameters taken directly from the CSV row.
        meta = {
            "network_state": (row["network_state"], None),
            "duration": (row["duration"], None),
            "da_level": (row["da_level"], None),
            "input_conductance": (row["input_conductance"], None),
            "input_scale_factor": (row["input_scale_factor"], None),
        }
        parameters = SimulationParameters(kv_pairs=meta)

        # Load spikes.
        spikes_h5 = Path(row["output_file"])
        pops = _populations_from_output_spikes(spikes_h5)
        spike_map = SpikeMap(populations=pops)

        # Load stimulus.
        stimulus_set = _parse_input_config(Path(row["input_config"]))

        # Assemble a ``Simulation`` entry.
        runs.append(
            Simulation(
                simulation_id=Path(row["sim_info_file"]).stem,
                parameters=parameters,
                stimulus=stimulus_set,
                output_spike=spike_map,
            )
        )

    standard_group = StandardFormatGroup(global_section=global_section, runs=runs)

    raw_bundle = RawBundleGroup() if embed_raw else None
    if raw_bundle:
        raw_bundle.out_path = output_dir
        raw_bundle.zip_link.target = str(embed_raw_path)

    proto = ProtocolGroup(version=protocol_version, content_path=protocol_info_path)
    return MasterBundle(standard=standard_group, raw=raw_bundle, protocol=proto)
