"""Utilities for constructing a :class:`~converters.schema.master_bundle.MasterBundle`
from the directory layout produced by the AMU (Amsterdam
University) simulation pipeline.

The module contains small helper functions that parse NPZ spike files,
assemble stimulus definitions, and build the high‑level ``MasterBundle``
object ready for HDF5 serialisation.

Typical usage
-------------
>>> bundle = build_amu_master_bundle(
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
from typing import Dict, List, Any

import numpy as np
import yaml

from converters.schema.external_dependencies_group import ExternalDependenciesGrp
from converters.schema.global_section import GlobalSection
from converters.schema.master_bundle import MasterBundle
from converters.schema.neuron import Neuron
from converters.schema.population import Population
from converters.schema.population_stimulus import PopulationStimulus
from converters.schema.raw_bundle_group import RawBundleGroup
from converters.schema.simulation import Simulation
from converters.schema.simulation_parameters import SimulationParameters
from converters.schema.spike_map import SpikeMap
from converters.schema.standard_format_group import StandardFormatGroup
from converters.schema.stimulus import Stimulus
from converters.schema.stimulus_set import StimulusSet
from converters.utils.helpers import AMU_POPULATIONS

log = logging.getLogger(__name__)


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


def build_amu_master_bundle(
        sim_dir: Path,
        *,
        embed_raw: bool = True,
        embed_raw_path: Path,
        output_dir: Path,
) -> MasterBundle:
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
    global_meta = {
        "project": "VBT‑AMU",
        "author": "John Doe",
        "description": "AMU parameter‑sweep conversion",
    }
    global_section = GlobalSection(metadata_mapping=global_meta, experiment_info="AMU sweep")

    # Walk every sub‑directory → one Simulation entry
    sims: List[Simulation] = []
    for run_dir in sorted(sim_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # ---- read the config.yaml -------------------------------------------------
        cfg_path = run_dir / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        rec_raw: dict = np.load(run_dir / "record_info" / "record_neurons.data", allow_pickle=True)
        cfg["record_neurons"] = {str(k): int(v) for k, v in
                                 rec_raw.items()}  # add the record‑neurons mapping to the config and dump it as part of parameters
        # The old code kept the whole dictionary as metadata – we do the same.
        param = SimulationParameters(kv_pairs=cfg)

        # ---- spike trains (NPZ per layer) -----------------------------------------
        spike_dir = run_dir / "spike_times"
        populations: List[Population] = []
        for npz_file in sorted(spike_dir.glob("*.npz")):
            pop_name = npz_file.stem
            populations.append(_population_from_npz(pop_name, npz_file))
        spike_map = SpikeMap(populations=populations)

        # ---- stimulus (piecewise‑constant) ----------------------------------------
        stimulus_set = _parse_stimulus_set(cfg, run_dir)

        sim = Simulation(
            simulation_id=run_dir.name,
            parameters=param,
            stimulus=stimulus_set,
            output_spike=spike_map,
        )
        sims.append(sim)

    # Assemble the master bundle
    std_fmt = StandardFormatGroup(global_section=global_section, runs=sims)

    # Optional raw‑bundle zip (identical to the old behaviour)
    raw_bundle = RawBundleGroup() if embed_raw else None
    if raw_bundle:
        raw_bundle.out_path = output_dir
        raw_bundle.zip_link.target = str(embed_raw_path)
    deps = ExternalDependenciesGrp(deps={})
    return MasterBundle(standard=std_fmt, raw=raw_bundle, deps=deps)


def _parse_stimulus_set(cfg: dict[Any, Any] | Any, run_dir: Path) -> StimulusSet:
    """Parse piecewise‑constant stimulus files and build a :class:`StimulusSet`.

    The function looks for files matching the pattern ``random_input_<N>ms.npz``
    within ``run_dir``.  Each file contributes one segment to the stimulus: the
    upper limit (``N`` ms) and the waveform stored under the key ``'arr_0'``.
    The final segment extends to the total simulation time defined by
    ``cfg['t_sim']`` (default ``1000`` ms).

    Args:
        cfg: The configuration dictionary loaded from ``config.yaml`` (or an
            empty dictionary if the file is missing).  Only the key ``'t_sim'``
            is consulted.
        run_dir: Path to the directory that holds the current simulation run.

    Returns:
        A :class:`~converters.schema.stimulus_set.StimulusSet` containing a
        :class:`~converters.schema.population_stimulus.PopulationStimulus` for
        each population listed in :data:`AMU_POPULATIONS`.

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
    limits.append(cfg.get("t_sim", 1000))  # total simulation time (ms)

    # Convert to numpy & compute max value
    limits_arr = np.array(limits, dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)
    max_val = float(values_arr.max()) if values_arr.size else 0.0
    # A single stimulus is applied uniformly across all populations in the simulation,
    # serving as the default sub-stimulus for each population.
    stimulus_set = StimulusSet(
        populations=[
            PopulationStimulus(
                name=pop,
                sub_stimuli={
                    "default": Stimulus(limits=limits_arr, values=values_arr, max_value=max_val)
                },
                sub_attrs={},
            )
            for pop in AMU_POPULATIONS
        ]
    )
    return stimulus_set
