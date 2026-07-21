"""Fast validation of a MasterBundle HDF5 file.

The public entry point is :func:`validate_master_bundle`.  It raises a
:class:`ValidationError` with an informative message as soon as a rule is
violated.  The function is deliberately lightweight – it never reads large
datasets, it only inspects attributes, group names and dataset shapes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Sequence

import h5py

from converters.utils.common import configure_logging

SUPPORTED_VERSIONS = {"1.0"}  # add future versions here


# Public exception – callers may catch this to turn the message into a CLI error
class ValidationError(RuntimeError):
    """Raised when a MasterBundle does not satisfy the schema."""

    def __init__(self, message: str, location: str | None = None):
        """
        Parameters
        ----------
        message : str
            Human‑readable description of the problem.
        location : str, optional
            Path‑like identifier inside the HDF5 file (e.g. ``/StandardFormat/Sim_01``)
            that makes it easy to locate the offending element.
        """
        if location:
            message = f"{location}: {message}"
        super().__init__(message)
        self.location = location
        self.message = message


# Small convenience wrappers that raise ValidationError as soon as something
# is missing or malformed.
def _require(condition: bool, msg: str, location: str | None = None) -> None:
    if not condition:
        raise ValidationError(msg, location)


def _require_group(parent: h5py.Group, name: str) -> h5py.Group:
    _require(name in parent, f"Missing group '{name}'", parent.name)
    grp = parent[name]
    _require(isinstance(grp, h5py.Group), f"'{name}' is not a group", parent.name)
    return grp


def _require_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    _require(name in parent, f"Missing dataset '{name}'", parent.name)
    dset = parent[name]
    _require(isinstance(dset, h5py.Dataset), f"'{name}' is not a dataset", parent.name)
    return dset


def _require_attr(obj: h5py.Group | h5py.Dataset, name: str) -> None:
    _require(name in obj.attrs, f"Missing attribute '{name}'", obj.name)


def _attr_equals(obj: h5py.Group | h5py.Dataset, name: str, expected) -> None:
    _require_attr(obj, name)
    actual = obj.attrs[name]
    _require(actual == expected, f"Attribute '{name}' must be {expected!r}, got {actual!r}", obj.name)


def _attr_is_string(obj: h5py.Group | h5py.Dataset, name: str) -> None:
    """ExternalDependencies attributes must be plain strings."""
    _require_attr(obj, name)
    val = obj.attrs[name]
    _require(isinstance(val, (str, bytes)), f"Attribute '{name}' must be a string", obj.name)


# Individual rule implementations – each receives the open H5 file.
def _check_root(file: h5py.File) -> None:
    """The root attribute ``format_version`` must exist and be supported."""
    _require_attr(file, "format_version")
    version = file.attrs["format_version"]
    _require(version in SUPPORTED_VERSIONS,
             f"Unsupported format_version {version!r}", "/")


def _check_standardformat(file: h5py.File) -> None:
    """The ``/StandardFormat`` group must exist."""
    _require_group(file, "StandardFormat")


def _check_global(file: h5py.File) -> None:
    """``/StandardFormat/Global``  + attribute ``experiment_info``."""
    std = file["StandardFormat"]
    glob = _require_group(std, "Global")
    _require_attr(glob, "experiment_info")


def _check_simulations(file: h5py.File) -> List[h5py.Group]:
    """``/StandardFormat/Simulations`` with at least one child ``Sim_<ID>``."""
    std = file["StandardFormat"]
    sims_grp = _require_group(std, "Simulations")
    sim_groups = [g for n, g in sims_grp.items() if isinstance(g, h5py.Group) and n.startswith("Sim_")]
    _require(len(sim_groups) > 0, "No simulation groups (Sim_<ID>) found", sims_grp.name)
    return sim_groups  # return for the per‑simulation checks


def _check_one_simulation(sim_grp: h5py.Group, populations: List[str]) -> List[str]:
    """
    Everything that lives inside a single ``Sim_<ID>`` group.
    """
    # The group name must match the attribute ``simulation_id`` (if present)
    sim_id = sim_grp.name.split("/")[-1].replace("Sim_", "")
    # The attribute is *not* required by the schema, but we verify it if present.
    if "simulation_id" in sim_grp.attrs:
        _require(sim_grp.attrs["simulation_id"] == sim_id,
                 f"simulation_id attribute must equal '{sim_id}'",
                 sim_grp.name)

    # Required sub‑groups
    params = _require_group(sim_grp, "Parameters")
    stimuli = _require_group(sim_grp, "Stimuli")
    out_spikes = _require_group(sim_grp, "OutputSpikes")

    # ``Parameters`` must have at least one attribute
    _require(len(params.attrs) > 0,
             "Parameters group must contain at least one attribute",
             params.name)

    # Each population under ``Stimuli`` must have ≥ 1 sub‑stimulus
    for pop_name, pop_grp in stimuli.items():
        _require(isinstance(pop_grp, h5py.Group), f"Population '{pop_name}' is not a group", stimuli.name)
        _require(len(pop_grp) > 0,
                 f"Population '{pop_name}' must contain at least one sub‑stimulus group",
                 pop_grp.name)

        # Each sub‑stimulus must have the three datasets with shape checks
        for sub_name, sub_grp in pop_grp.items():
            _require(isinstance(sub_grp, h5py.Group), f"Sub‑stimulus '{sub_name}' not a group", pop_grp.name)

            limits = _require_dataset(sub_grp, "Limits")
            values = _require_dataset(sub_grp, "Values")
            maxval = _require_dataset(sub_grp, "MaxValue")

            # Shape checks – we only need the rank & length, not the data itself
            _require(limits.ndim == 1,
                     "Limits must be 1‑D", limits.name)
            _require(values.ndim == 1,
                     "Values must be 1‑D", values.name)
            _require(limits.shape[0] == values.shape[0] + 1,
                     f"len(Limits) ({limits.shape[0]}) must be len(Values)+1 ({values.shape[0] + 1})",
                     sub_grp.name)

    # ``OutputSpikes/SpikeMap`` must exist and contain at least one neuron per population
    spike_map = _require_group(out_spikes, "SpikeMap")
    populations_extracted = []
    for pop_name, pop_grp in spike_map.items():
        populations_extracted.append(pop_name)
        _require(isinstance(pop_grp, h5py.Group), f"Population '{pop_name}' in SpikeMap is not a group", spike_map.name)
        # there must be at least one neuron subgroup (the name is the neuron index)
        neuron_groups = [g for n, g in pop_grp.items() if isinstance(g, h5py.Group)]
        _require(len(neuron_groups) > 0,
                 f"Population '{pop_name}' must contain at least one neuron group",
                 pop_grp.name)

        # Each neuron group must have a 1‑D ``times`` dataset (empty allowed)
        for neu_name, neu_grp in pop_grp.items():
            times = _require_dataset(neu_grp, "times")
            _require(times.ndim == 1,
                     "Neuron times must be 1‑D", times.name)
    if len(populations) > 0:
        # Check that the populations in SpikeMap is the same for all simulations
        _require(set(populations_extracted) == set(populations),
                 f"Populations in SpikeMap {populations_extracted} do not match expected {populations}",
                 spike_map.name)
    return populations_extracted


def _check_rawbundle(file: h5py.File) -> None:
    """External‑link dataset with the four mandatory attributes."""
    if "RawBundle" not in file:
        return  # optional group – nothing to validate
    raw_grp = _require_group(file, "RawBundle")
    dset = _require_dataset(raw_grp, "raw_bundle.zip")
    for attr in ("path", "target", "link_type", "format"):
        _require_attr(dset, attr)


def _check_external_deps(file: h5py.File) -> None:
    """All members of ``ExternalDependencies`` are string attributes."""
    if "ExternalDependencies" not in file:
        return
    deps_grp = _require_group(file, "ExternalDependencies")
    for name, val in deps_grp.attrs.items():
        _require(isinstance(val, (str, bytes)),
                 f"External dependency attribute '{name}' must be a string", deps_grp.name)


# Public API – a single function that runs all checks.
CHECKS: Sequence[Callable[[h5py.File], None | List[h5py.Group]]] = (
    _check_root,
    _check_standardformat,
    _check_global,
    _check_simulations,  # returns the list of simulation groups
    _check_rawbundle,
    _check_external_deps,
)


def validate_master_bundle(
        bundle_path: Path,
        *,
        strict: bool = True,
        logger: logging.Logger | None = None,
) -> None:
    """
    Validate a master HDF5 bundle against the minimal schema described in the
    question.

    Parameters
    ----------
    bundle_path : pathlib.Path
        Path to the ``*.h5`` file that should be checked.
    strict : bool, default=True
        When ``True`` all rules (including the optional ``RawBundle`` and
        ``ExternalDependencies`` groups) are enforced.  When ``False`` only the
        core ``StandardFormat`` hierarchy is validated – useful for quick unit
        tests.
    logger : logging.Logger, optional
        If supplied, progress and the exact location of a failure are logged at
        *DEBUG* level before raising the exception.

    Raises
    ------
    ValidationError
        The first rule that fails.
    OSError / h5py.HDF5Error
        If the file cannot be opened or is corrupted.
    """
    if logger is None:
        # Logging configuration (root logger)
        logger = configure_logging(
            __name__,
            log_file_name=Path(__file__).with_suffix(".log").name,
            level=getattr(logging, "DEBUG" if strict else "INFO"),
        )

    logger.debug("Opening master bundle %s for validation", bundle_path)
    with h5py.File(bundle_path, "r") as hf:
        # Run the static checks that do not need per‑simulation data.
        for fn in CHECKS[:4]:
            logger.debug("Running check %s", fn.__name__)
            result = fn(hf)  # type: ignore[arg-type]
            # ``_check_simulations`` returns the list of simulation groups,
            # everything else returns ``None``.
            if fn is _check_simulations:
                sim_groups: List[h5py.Group] = result  # type: ignore[assignment]

        # Per‑simulation checks – iterate only once, abort on first failure.
        logger.debug("Running per‑simulation checks (%d sims)", len(sim_groups))
        populations = []
        for sim in sim_groups:
            try:
                sim_populations = _check_one_simulation(sim, populations)
                if populations.count(sim_populations) == 0:
                    populations.extend(sim_populations)
            except ValidationError as exc:
                logger.debug("Validation failed in simulation %s → %s", sim.name, exc)
                raise

        # Optional groups
        if strict:
            logger.debug("Running optional RawBundle check")
            _check_rawbundle(hf)
            logger.debug("Running optional ExternalDependencies check")
            _check_external_deps(hf)

    logger.info("Master bundle %s passed validation", bundle_path)
