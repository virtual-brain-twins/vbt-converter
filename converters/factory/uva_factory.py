"""Utilities for constructing a :class:`~converters.schema.master_bundle.MasterBundle`
from the directory layout produced by the UVA (Universiteit van Amsterdam) simulation pipeline, See repository: https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt_t4-3_amu_protocol4.

The module contains small helper functions that parse NPZ spike files,
assemble stimulus definitions, and build the high‑level ``MasterBundle``
object ready for HDF5 serialisation.

Typical usage
-------------
>>> bundle = build_uva_master_bundle(
...     sim_dir=Path("/path/to/simulations"),
...     embed_raw=True,
...     embed_raw_path=Path("/path/to/raw.zip"),
...     output_dir=Path("/path/to/output")
... )
>>> bundle.to_hdf5(output_dir / "bundle.h5")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import yaml

from converters.schema.configuration import Configuration
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
from converters.utils.metadata import MetaDataBuilder
from converters.utils.units import Unit

log = logging.getLogger(__name__)

PARAMETERS = ["scale_num_neurons", "scale_syn_strength", "scale_nmda_efficacy"]
PARAMETERS_SHORT = ["numneur", "synstr", "nmdaeff"]


def _population_from_npz(name: str, npz_path: Path) -> Population:
    """Create a :class:`~converters.schema.population.Population` from a single NPZ file.

    The NPZ file must contain a ``times`` array and may optionally contain an
    ``indices`` array.  If ``indices`` is missing, a dummy neuron is created for
    each spike time; otherwise spikes are grouped per neuron according to the
    indices.

    Args:
        name: The name to assign to the resulting population.
        npz_path: Path to the ``.npz`` file containing spike data.

    Returns:
        A fully populated :class:`~converters.schema.population.Population`
        instance.

    Raises:
        FileNotFoundError: If ``npz_path`` does not exist.
        ValueError: If required arrays (e.g. ``times``) are missing from the file.
        OSError: For any I/O‑related error while reading the NPZ file.
    """
    data = np.load(npz_path, allow_pickle=False)
    times = data["times"]
    indices = data.get("indices")
    # The original format stores *one* array per population – the
    # indices array can be ``None`` (meaning “implicit ordering”).
    if indices is None:
        # Build a dummy “one neuron per spike” representation – the
        # original tools used this only for *input* spikes, not for the
        # output SpikeMap.  For the output we *must* have per‑neuron
        # grouping, therefore we build a virtual neuron per spike.
        neurons = [Neuron(index=int(i), times=np.array([t])) for i, t in enumerate(times)]
    else:
        # Build a neuron → spike‑time mapping.
        # ``indices`` is an int array of the same length as ``times``.
        mapping: Dict[int, List[float]] = {}
        for idx, t in zip(indices, times):
            mapping.setdefault(int(idx), []).append(float(t))

        neurons = [
            Neuron(index=idx, times=np.array(sorted(ts), dtype=np.float64))
            for idx, ts in sorted(mapping.items())
        ]

    return Population(population_name=name, neurons=neurons)


def build_uva_master_bundle(
        sim_dir: Path,
        *,
        embed_raw: bool = True,
        embed_raw_path: Path,
        output_dir: Path,
        protocol_version: str,
        protocol_info_path: Path) -> MasterBundle:
    """Construct a :class:`~converters.schema.master_bundle.MasterBundle` from an AMU run directory.

    The function walks the ``sim_dir`` directory, interprets each sub‑folder as
    an individual simulation run, extracts configuration, spike data and stimulus
    information, and assembles them into a ``MasterBundle`` object.  Optionally,
    the original raw directory tree can be archived and linked via an external
    dependency.

    Args:
        sim_dir: Directory that contains the whole sweep (many sub‑folders, each a
            single run).
        embed_raw: When ``True`` the original directory tree is zipped and stored
            as an external link (mimicking the behaviour of the legacy
            ``build_master_h5`` routine).
        embed_raw_path: Path where the raw zip archive should be created.
        output_dir: Destination directory for the generated raw‑bundle artefact.
        protocol_version: The version of the protocol to be incorporated.
        protocol_info_path: Path to a PDF file containing protocol information.

    Returns:
        A fully populated :class:`~converters.schema.master_bundle.MasterBundle`
        ready to be written with ``bundle.to_hdf5(out_path)``.

    Raises:
        FileNotFoundError: If ``sim_dir`` does not exist or required files such
            as ``config.yaml`` or ``record_neurons.data`` are missing in a run
            sub‑directory.
        yaml.YAMLError: If parsing a ``config.yaml`` file fails.
        ValueError: If spike files are malformed or missing required arrays.
        OSError: For any I/O‑related errors while reading files or writing the raw
            zip archive.

    """
    # Global metadata (same as before, but now lives in the GlobalSection)
    # We keep a very small set of global fields – the rest can be added later.
    mb = MetaDataBuilder()
    mb.author("John Doe")
    mb.creation_date("2026-07-31T12:00:00Z")
    mb.software("AMU Software")
    mb.software_version("1.0.2", unit=None)
    mb.run_id("e3b0c442-98fc-1c14-9af6-3b33d8fbc123")
    global_meta = mb.to_dict()
    global_section = GlobalSection(metadata_mapping=global_meta,
                                   parameter_simulation_mapping={},
                                   experiment_info="AMU sweep")

    # Walk every sub‑directory → one Simulation entry
    sims: List[Simulation] = []
    for run_dir in sorted(sim_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Load the full configuration (raw key → value)
        cfg_path = run_dir / "config.yaml"
        # Read config and convert to key → (value, unit=None)
        full_cfg: Dict[str, Tuple[Any, Optional[Unit]]] = {
            k: (v, None) for k, v in (yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}).items()
        }
        # Add the record‑neurons mapping
        rec_path = run_dir / "record_info" / "record_neurons.data"
        if rec_path.is_file():
            rec_raw = np.load(rec_path, allow_pickle=True)
            # keep the original mapping (str → int) in the configuration
            full_cfg["record_neurons"] = ({str(k): int(v) for k, v in rec_raw.items()}, None)

        # Extract the selected dynamic parameters, rename them, and wrap values as (value, None)
        sim_params: dict = {}
        for long_name, short_name in zip(PARAMETERS, PARAMETERS_SHORT):
            if long_name in full_cfg:
                # Unit is not provided inside the parent sweep directory, hence kept None
                sim_params[short_name] = full_cfg.pop(long_name)

        # Remaining entries belong to the generic configuration object
        # `full_cfg` now contains everything except the three parameters above.
        param = SimulationParameters(kv_pairs=sim_params)
        config = Configuration(kv_pairs=full_cfg)  # ← all other config items

        # spike trains (NPZ per layer)
        spike_dir = run_dir / "spike_times"
        populations: List[Population] = []
        for npz_file in sorted(spike_dir.glob("*.npz")):
            pop_name = npz_file.stem
            populations.append(_population_from_npz(pop_name, npz_file))
        spike_map = SpikeMap(populations=populations)

        # stimulus (piecewise‑constant)
        stimuli_group = _parse_stimulus_group(full_cfg, run_dir, [pop.population_name for pop in spike_map.populations])

        sim = Simulation(
            simulation_id=run_dir.name,
            configuration=config,
            parameters=param,
            stimulus=stimuli_group,
            output_spike=spike_map,
        )
        sims.append(sim)

    # Assemble the master bundle
    std_fmt = StandardFormatGroup(global_section=global_section, runs=sims)

    # Optional raw‑bundle zip (identical to the old behaviour)
    raw_bundle = RawBundleGroup() if embed_raw else None
    if raw_bundle:
        raw_bundle.zip_path = embed_raw_path
    proto = ProtocolGroup(version=protocol_version, content_path=protocol_info_path)
    return MasterBundle(standard=std_fmt, raw=raw_bundle, protocol=proto)


def _parse_stimulus_group(cfg: dict[Any, Any] | Any, run_dir: Path, populations: List[str]) -> StimuliGroup:
    """Parse piecewise‑constant stimulus files and build a :class:`StimuliGroup`.

    The function looks for files matching the pattern ``random_input_<N>ms.npz``
    within ``run_dir``.  Each file contributes one segment to the stimulus: the
    upper limit (``N`` ms) and the waveform stored under the key ``'arr_0'``.
    The final segment extends to the total simulation time defined by
    ``cfg['t_sim']`` (default ``1000``ms).

    Args:
        cfg: The configuration dictionary loaded from ``config.yaml`` (or an
            empty dictionary if the file is missing).  Only the key ``'t_sim'``
            is consulted.
        run_dir: Path to the directory that holds the current simulation run.
        populations: A list of population names for which to create stimulus data.
    Returns:
        A :class:`~converters.schema.stimuli_group.StimuliGroup` containing a
        :class:`~converters.schema.population_stimulus.PopulationStimulus` for
        each population listed in :data:`populations`.

    Raises:
        FileNotFoundError: If a matching ``*.npz`` stimulus file is expected but
            cannot be read.
        ValueError: If a stimulus file does not contain the expected array.
        OSError: For any I/O‑related error while reading the NPZ files.
    """
    limits = [0]
    values = [0.0]
    pattern = re.compile(r"random_input_(\d+)ms\.npz")
    for f in sorted(run_dir.iterdir()):
        m = pattern.fullmatch(f.name)
        if not m:
            continue
        limits.append(int(m.group(1)))
        values.append(np.load(f)["arr_0"])
    limits.append(cfg.get("t_sim", 1000)[0])  # total simulation time (ms)

    # Convert to numpy & compute max value
    limits_arr = np.array(limits, dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)
    # A single stimulus is applied uniformly across all populations in the simulation,
    # serving as the default sub-stimulus for each population.
    return StimuliGroup(
        stimuli_map={
            f"{population}_default": Stimulus(limits=limits_arr, values=values_arr) for population in populations
        },
        attr_map={},
    )
