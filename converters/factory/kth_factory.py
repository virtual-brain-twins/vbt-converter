"""KTH conversion utilities.

This module provides helper functions that translate the original Snudda‑style
JSON and HDF5 stimulus specifications into the new `converters.schema` data
model.  The public entry point is :func:`build_kth_master_bundle`, which builds a
complete :class:`~converters.schema.master_bundle.MasterBundle` from a CSV file
produced by the original KTH pipeline.

Typical usage
-------------
    >>> bundle = build_kth_master_bundle(
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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from converters.schema.configuration import Configuration
from converters.schema.global_section import GlobalSection
from converters.schema.map import Map
from converters.schema.master_bundle import MasterBundle
from converters.schema.neuron import Neuron
from converters.schema.population import Population
from converters.schema.protocol_group import ProtocolGroup
from converters.schema.raw_bundle_group import RawBundleGroup
from converters.schema.simulation import Simulation
from converters.schema.simulation_parameters import SimulationParameters
from converters.schema.standard_format_group import StandardFormatGroup
from converters.schema.stimuli_group import StimuliGroup
from converters.schema.stimulus import Stimulus
from converters.utils.kth_protocol_helper import stream_csv_as_dicts
from converters.utils.metadata import MetaDataBuilder, StandardKey
from converters.utils.units import Unit

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

DEFAULT_KTH_PARAMETERS = ["network_state", "duration", "da_level", "input_conductance", "input_scale_factor"]

def _eval_arith_string(val: Any) -> Any:
    """Convert a string that looks like a simple arithmetic expression to a number.

    The function recognises patterns such as ``"*1.25"``, ``"30/4000e-6"``, or
    ``"2e-05"`` and evaluates the arithmetic part, returning a ``float``.  If the
    input is not a string or does not match the expected pattern, the original
    value is returned unchanged along with the unit.

    Args:
        val: The value to evaluate.  Usually a ``str`` but any type is accepted.

    Returns:
        The evaluated number and the unit if the input matched the arithmetic pattern,
        otherwise the original ``val``. Unit is not available, hence kept as ``None``.

    Raises:
        TypeError: If ``val`` is a string but contains characters that make the
            regular expression match fail (unlikely in normal operation).
        Exception: Any exception raised by ``eval`` is caught and results in the
            original string being returned.
    """
    if not isinstance(val, str):
        return val, None

    m = _ARITHMETIC_RE.match(val)
    if not m:
        return val, None

    expr = m.group("expr")
    try:
        # ``eval`` is safe because the regex guarantees that only numbers,
        # decimal points, exponent notation and the operators + - * / are present.
        number = eval(expr, {"__builtins__": {}}, {})
    except Exception:  # pragma: no cover
        return val, None

    # The optional leading '*' is ignored for the numeric result.
    return number, None


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
    sub_stimuli: Dict[str, Stimulus] = {}
    sub_attrs: Dict[str, Dict[str, Any]] = {}
    pop_list_map: Dict[str, List[str]] = {}

    for pop_name, pop_content in cfg.items():
        for stim_name, sub_dict in pop_content.items():
            if "start" in sub_dict and "frequency" in sub_dict:
                stimulus, attrs = _parse_cortical_sub_stimulus(sub_dict)
            else:
                stimulus, attrs = _parse_generic_sub_stimulus(stim_name, sub_dict)

            sub_stimuli[f"{pop_name}_{stim_name}"] = stimulus
            sub_attrs[f"{pop_name}_{stim_name}"] = attrs
            pop_list_map[f"{pop_name}_{stim_name}"] = [pop_name]

    return StimuliGroup(stimuli_map=sub_stimuli, attr_map=sub_attrs, pop_list_map=pop_list_map)


def _populations_from_output_spikes(h5_path: Path) -> List[Population]:
    """Read the classic KTH *output‑spikes* HDF5 file and build ``Population`` objects.

    The Snudda layout is::

        /meta_data
            name               – 1‑D array of bytes, length = #neurons
            virtual_neuron    – 1‑D bool array, length = #neurons
        /neurons
            <gid>/spikes      – spike times (shape (1, N_spikes))
            <gid>/voltage     – membrane voltage (shape (1, N_samples))
            …                 – other datasets that we simply ignore here

    For every global‑ID (gid) we create a :class:`Neuron` and fill **all** fields
    that are available:

    * ``times``               – the spike‑time vector (float64)
    * ``times_unit``          – the ``unit`` attribute attached to the ``spikes`` dataset,
                               if it exists.
    * ``voltage``             – the voltage trace (float64)
    * ``voltage_unit``        – the ``unit`` attribute attached to the ``voltage`` dataset,
                               if it exists.
    * ``virtual``             – taken from the ``virtual_neuron`` array in ``meta_data``.

    Afterwards the neurons are grouped by the population name stored in
    ``meta_data/name`` and a list of :class:`Population` objects is returned.

    Parameters
    ----------
    h5_path : Path
        Path to the KTH ``output‑spikes`` HDF5 file.

    Returns
    -------
    List[Population]
        One ``Population`` per distinct name, each containing the corresponding
        ``Neuron`` objects.

    Raises
    ------
    FileNotFoundError
        If ``h5_path`` does not exist.
    OSError
        If the file cannot be opened (corrupt, permission error, …).
    KeyError
        If the expected groups/datasets are missing.
    """
    with h5py.File(h5_path, "r") as f:
        # Global meta‑data
        # `name` is stored as a 1‑D array of bytes; we keep it as‑is and decode
        # lazily when we need the string for a particular gid.
        population_name: np.ndarray = f["meta_data"]["name"][()]  # shape (N_neurons,)
        # ``virtual_neuron`` is a 1‑D boolean array that lines‑up with the order
        # of the groups under /neurons (i.e. with the gid value).
        virtual_flags: np.ndarray = f["meta_data"]["virtual_neuron"][()].astype(bool)

        # The per‑neuron group container
        neuron_grp = f["neurons"]
        # ``neuron_grp`` behaves like a dict mapping ``gid_str`` → h5py.Group.
        # We'll build a temporary dict mapping population‑name → list[Neuron].
        populations: Dict[str, List[Neuron]] = {}

        for gid_str, ngrp in neuron_grp.items():
            gid = int(gid_str)  # the global identifier

            # Spike times
            # KTH stores spikes as a (1, N) array; `squeeze` removes the
            # leading dimension.  We also force ``float64`` to match the
            # Neuron dataclass contract.
            spikes_raw = ngrp["spikes"][()]
            times: np.ndarray = spikes_raw.squeeze().astype(np.float64)

            # Optional unit attribute – may be missing → ``None``.
            times_unit = ngrp["spikes"].attrs.get("unit", None)
            # If the attribute is a byte‑string, keep it as‑is; the Neuron class
            # stores whatever is provided (the Unit wrapper will handle it later).
            if isinstance(times_unit, bytes):
                times_unit = times_unit.decode("utf-8")

            # Voltage trace (may be absent for some neurons)
            if "voltage" in ngrp:
                voltage_raw = ngrp["voltage"][()]
                voltage: np.ndarray = voltage_raw.squeeze().astype(np.float64)

                voltage_unit = ngrp["voltage"].attrs.get("unit", None)
                if isinstance(voltage_unit, bytes):
                    voltage_unit = voltage_unit.decode("utf-8")
            else:
                # No voltage recorded → store an empty array.
                voltage = np.empty(0, dtype=np.float64)
                voltage_unit = None

            # Virtual flag – taken from the pre‑loaded ``virtual_flags`` array.
            # Guard against out‑of‑range indices (mal‑formed files); we treat
            # missing entries as *non‑virtual* but raise a warning so the caller
            # can investigate.
            try:
                is_virtual = bool(virtual_flags[gid])
            except IndexError:
                import warnings
                warnings.warn(
                    f"virtual_neuron flag missing for gid={gid}; assuming False",
                    RuntimeWarning,
                )
                is_virtual = False

            # Build the Neuron instance
            neuron = Neuron(
                index=gid,
                times=times,
                times_unit=Unit(times_unit) if times_unit is not None else None,
                voltage=voltage,
                voltage_unit=Unit(voltage_unit) if voltage_unit is not None else None,
                virtual=is_virtual,
            )

            # Associate the neuron with its population
            # ``population_name`` is a byte array where the element index equals
            # the gid.  It must be decoded to a normal string.
            pop_name_bytes = population_name[gid]
            if isinstance(pop_name_bytes, (bytes, np.bytes_)):
                pop_name = pop_name_bytes.decode("utf-8")
            else:
                # Unexpected datatype – fall back to ``str`` conversion.
                pop_name = str(pop_name_bytes)

            populations.setdefault(pop_name, []).append(neuron)

        # Convert the temporary dict into the final list of Population
        return [Population(population_name=pop, neurons=neurons) for pop, neurons in populations.items()]


def build_kth_master_bundle(
        input_csv_path: Path,
        embed_raw: bool,
        embed_raw_path: Path,
        output_dir: Path,
        protocol_version: str,
        protocol_info_path: Path,
        parameters: List[str] | None = None,
        metadata: dict[str, str] | None = None,
) -> MasterBundle:
    """Build a :class:`MasterBundle` from a KTH CSV file.

    The function reads the CSV from the raw data directory, parses
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
        parameters: Optional list of parameter names to extract from the configuration.
            If ``None``, the default set of parameters is used.
        metadata: Optional dictionary of global metadata fields to include in the
            ``GlobalSection``.  If ``None``, default metadata is used.

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

    # Global metadata (same as before, but now lives in the GlobalSection)
    # We keep a very small set of global fields – the rest can be added later.
    mb = MetaDataBuilder()
    # Apply defaults first
    defaults: dict[StandardKey, Any] = {
        StandardKey.AUTHOR: "John Doe",
        StandardKey.CREATION_DATE: datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        StandardKey.SOFTWARE: "KTH Software",
        StandardKey.SOFTWARE_VERSION: "1.0.0",
        StandardKey.RUN_ID: str(uuid.uuid4())
    }
    for key, value in defaults.items():
        mb.set(key, value)

    # Overlay any user‑provided metadata (string → StandardKey conversion)
    if metadata:
        for raw_key, val in metadata.items():
            try:
                key = StandardKey(raw_key)  # raises ValueError if unknown
                mb.set(key, val)  # unit is None for CLI strings
            except ValueError as exc:
                raise KeyError(f"Unsupported metadata key supplied via CLI: {raw_key!r}") from exc

    global_meta = mb.to_dict()

    # Global section – reuse high‑level fields from each CSV row.
    global_section = GlobalSection(
        metadata_mapping=global_meta,
        parameter_simulation_mapping={},
        experiment_info="KTH sweep experiment",
    )

    runs: List[Simulation] = []
    for row in csv_rows:
        # Simulation parameters taken directly from the CSV row.
        # Extract the selected dynamic parameters, rename them, and wrap values as (value, None)
        # Unit is not provided inside the parent sweep directory, hence kept None
        full_cfg = {k: (v, None) for k, v in row.items()}  # copy the entire row to a new dict
        param_names = parameters if parameters is not None else DEFAULT_KTH_PARAMETERS
        sim_params = {p: full_cfg.pop(p) for p in param_names if p in full_cfg}

        # Remaining entries belong to the generic configuration object
        # `full_cfg` now contains everything except the default parameters above.
        sim_parameters = SimulationParameters(kv_pairs=sim_params)
        config = Configuration(kv_pairs=full_cfg)  # ← all other config items


        # Load spikes.
        spikes_h5 = Path(row["output_file"])
        pops = _populations_from_output_spikes(spikes_h5)
        out_map = Map(populations=pops)

        # Load stimulus.
        stimulus_set = _parse_input_config(Path(row["input_config"]))

        # Extract a friendly simulation identifier from the CSV row.
        # The CSV column sim_info_file (file_path) contains a path like: /some/dir/simulation‑metadata‑<SIM_ID>.json
        # We want only the `<SIM_ID>` part (the text after the final hyphen, without the file‑extension).
        sim_id: str = Path(row["sim_info_file"]).stem.rsplit("-", 1)[-1]
        # Assemble a ``Simulation`` entry.
        runs.append(
            Simulation(
                simulation_id=sim_id,
                configuration=config,
                parameters=sim_parameters,
                stimuli=stimulus_set,
                output_map=out_map
            )
        )

    standard_group = StandardFormatGroup(global_section=global_section, runs=runs)

    raw_bundle = RawBundleGroup() if embed_raw else None
    if raw_bundle:
        raw_bundle.zip_path = embed_raw_path
    proto = ProtocolGroup(version=protocol_version, content_path=protocol_info_path)
    return MasterBundle(standard=standard_group, raw=raw_bundle, protocol=proto)
