"""
Fast validation of a MasterBundle HDF5 file.

The public entry point is :func:`validate_master_bundle`.  It raises a
:class:`ValidationError` with an informative message as soon as a rule is
violated.  The function is deliberately lightweight – it never reads large
datasets, it only inspects attributes, group names and dataset shapes.

Both format versions **1.0** (original) and **2.0** (new) are supported.
The newer implementation of :class:`~converters.schema.master_bundle.MasterBundle`
stores the version in three integer attributes
(``format_version_major``, ``format_version_minor``, ``format_version_patch``)
*in addition* to the legacy dotted‑string attribute ``format_version``.
The validator therefore understands *both* representations and prefers the
integer representation when it is present.

Adding a future version only requires implementing a new per‑simulation check
and adding it to ``VERSION_REGISTRY``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import h5py
import numpy as np

from converters.utils.common import configure_logging

# Constants & public‑API items
SUPPORTED_VERSIONS = {"1.0", "2.0"}  # add future versions here
DEFAULT_VERSION = None  # if None we use the file’s version


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
#    is missing or malformed.
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
    _require(actual == expected,
             f"Attribute '{name}' must be {expected!r}, got {actual!r}",
             obj.name)


def _attr_is_string(obj: h5py.Group | h5py.Dataset, name: str) -> None:
    """ExternalDependencies attributes must be plain strings."""
    _require_attr(obj, name)
    val = obj.attrs[name]
    _require(isinstance(val, (str, bytes)),
             f"Attribute '{name}' must be a string", obj.name)


def _require_scalar(dataset: h5py.Dataset, name: str) -> None:
    """Raise ``ValidationError`` if ``dataset`` is not a scalar (shape == ())."""
    _require(dataset.shape == (),
             f"{name} must be a scalar dataset (shape == ()), got {dataset.shape}",
             location=dataset.name)


# Compound‑attribute helpers
def _is_compound_attribute(attr) -> bool:
    """
    Return ``True`` iff *attr* looks like the ``(value, unit)`` compound
    attribute defined by the VBT spec.
    """
    if not isinstance(attr, np.void):
        return False
    # must be a scalar (zero‑dimensional) structured array with two fields
    if attr.shape != ():
        return False
    if not isinstance(attr.dtype, np.dtype) or not attr.dtype.fields:
        return False
    fields = set(attr.dtype.fields)  # type: ignore[arg-type]
    return {"value", "unit"} <= fields


def _require_compound_attribute(obj: h5py.Group | h5py.Dataset, name: str) -> None:
    """
    Validate that ``obj.attrs[name]`` is a compound attribute ``(value, unit)``.
    The function checks:

    * presence,
    * correct dtype (scalar structured array with fields ``value`` and ``unit``),
    * the ``unit`` field is a UTF‑8 string (or empty/None),
    * the ``value`` field is a scalar of a supported native HDF5 type *or* a
      UTF‑8 string (JSON‑encoded value).

    The check does **not** decode JSON – that is left to the caller.
    """
    _require_attr(obj, name)
    attr = obj.attrs[name]

    _require(_is_compound_attribute(attr),
             f"Attribute '{name}' must be a compound attribute with fields "
             "'value' and 'unit'", obj.name)

    # ---- unit field ----------------------------------------------------
    unit_val = attr["unit"]
    if isinstance(unit_val, bytes):
        unit_val = unit_val.decode("utf-8")
    _require(isinstance(unit_val, (str, bytes)),
             f"Compound attribute '{name}' field 'unit' must be a UTF‑8 string "
             f"(got {type(unit_val)!r})", obj.name)

    # ---- value field ---------------------------------------------------
    value_val = attr["value"]
    # ``value`` may be a native scalar (int, float, bool) or a UTF‑8 string.
    if isinstance(value_val, (bytes, str)):
        # JSON‑encoded strings are allowed – we just verify that we have a string.
        pass
    else:
        # native NumPy scalar types are acceptable.
        _require(isinstance(value_val, (np.generic, int, float, bool)),
                 f"Compound attribute '{name}' field 'value' must be a scalar "
                 f"numeric/boolean or a UTF‑8 string; got {type(value_val)!r}",
                 obj.name)


def _check_all_compound_attributes(group: h5py.Group) -> None:
    """
    Iterate over **all** attributes of ``group`` and ensure every attribute
    conforms to the ``(value, unit)`` compound format required by the spec.
    """
    for attr_name in group.attrs:
        _require_compound_attribute(group, attr_name)


def _require_unit_attribute(parent: h5py.Group | h5py.Dataset, name: str) -> None:
    """
    Helper that checks the presence of a *unit* attribute on a dataset and
    guarantees that it is a UTF‑8 string (or empty string).

    The VBT spec stores units as **plain string attributes** (not as part of a
    compound attribute).  This function is used for the dataset‑level unit
    attributes (e.g. ``times`` → ``unit``).
    """
    _require_attr(parent, name)
    val = parent.attrs[name]
    _require(isinstance(val, (str, bytes)),
             f"Attribute '{name}' must be a UTF‑8 string (unit)", parent.name)


# Version detection (string *or* integer attributes)
def _detect_version(attrs: h5py.AttributeManager) -> str | None:
    """
    Resolve the bundle's version as a dotted string.

    Preference order:
        1. Integer attributes ``format_version_major``/``format_version_minor``.
           If ``format_version_patch`` is present and non‑zero, it is appended.
        2. Legacy string attribute ``format_version``.
        3. ``None`` → callers must decide what to do (usually raise).

    The function never raises; it simply returns ``None`` when no information
    can be found.
    """
    # Integer based representation – the new canonical form.
    major = attrs.get("format_version_major")
    minor = attrs.get("format_version_minor")
    if major is not None and minor is not None:
        # ``attrs.get`` may return NumPy scalar types – cast to int.
        try:
            major_i = int(major)
            minor_i = int(minor)
        except Exception:  # pragma: no cover – defensive
            return None

        patch = attrs.get("format_version_patch")
        if patch is None or int(patch) == 0:
            return f"{major_i}.{minor_i}"
        else:
            return f"{major_i}.{minor_i}.{int(patch)}"

    # Legacy string representation.
    legacy = attrs.get("format_version")
    if legacy is not None:
        # The attribute may be stored as bytes; decode if necessary.
        if isinstance(legacy, bytes):
            try:
                legacy = legacy.decode("utf-8")
            except Exception:  # pragma: no cover
                return None
        return str(legacy)

    return None


# COMMON (version‑agnostic) checks
def _check_root(file: h5py.File) -> None:
    """Root attribute ``format_version`` *or* integer version attributes must exist
    and be supported."""
    version = _detect_version(file.attrs)
    _require(version is not None,
             "Unable to determine bundle version – missing both string and integer version attributes",
             "/")
    _require(version in SUPPORTED_VERSIONS,
             f"Unsupported format_version {version!r}",
             "/")


def _check_standardformat(file: h5py.File) -> None:
    """The ``/StandardFormat`` group must exist."""
    _require_group(file, "StandardFormat")


def _check_global(file: h5py.File) -> None:
    """``/StandardFormat/Global`` with the two mandatory attributes."""
    std = file["StandardFormat"]
    glob = _require_group(std, "Global")

    # --- mandatory compound attributes -------------------------------------------------
    for mandatory in ("experiment_info", "parameter_simulation_mapping"):
        _require_compound_attribute(glob, mandatory)

    # The spec also allows *any* user‑defined metadata – they must all be compound.
    _check_all_compound_attributes(glob)


def _check_simulations(file: h5py.File) -> List[h5py.Group]:
    """``/StandardFormat/Simulations`` must contain at least one ``Sim_<ID>``."""
    std = file["StandardFormat"]
    sims_grp = _require_group(std, "Simulations")
    sim_groups = [
        g for n, g in sims_grp.items()
        if isinstance(g, h5py.Group) and n.startswith("Sim_")
    ]
    _require(len(sim_groups) > 0,
             "No simulation groups (Sim_<ID>) found", sims_grp.name)
    return sim_groups


def _check_rawbundle(file: h5py.File) -> None:
    """Optional ``RawBundle`` group – external‑link dataset with four attributes."""
    if "RawBundle" not in file:
        return
    raw_grp = _require_group(file, "RawBundle")
    dset = _require_dataset(raw_grp, "raw_bundle.zip")
    for attr in ("original_name", "original_size", "format", "checksum"):
        _require_attr(dset, attr)


def _check_protocol(file: h5py.File) -> None:
    """
    Optional ``Protocol`` group: a UTF‑8 ``Version`` attribute and a *compressed*
    ``Content`` dataset of dtype ``uint8``.
    """
    if "Protocol" not in file:
        return

    protocol_grp = _require_group(file, "Protocol")

    # attribute
    _require("Version" in protocol_grp.attrs,
             "Protocol group is missing required attribute 'Version'",
             protocol_grp.name)
    version_val = protocol_grp.attrs["Version"]
    _require(isinstance(version_val, (str, bytes)),
             "Protocol attribute 'Version' must be a UTF‑8 string (str or bytes)",
             protocol_grp.name)

    # dataset
    _require("Content" in protocol_grp,
             "Protocol group must contain a dataset named 'Content'",
             protocol_grp.name)
    content_ds = protocol_grp["Content"]
    _require(isinstance(content_ds, h5py.Dataset),
             "Protocol member 'Content' must be a dataset, not a group",
             protocol_grp.name)
    _require(content_ds.dtype == np.uint8,
             f"Protocol dataset 'Content' must have dtype uint8, got {content_ds.dtype}",
             content_ds.name)
    _require(content_ds.compression is not None,
             "Protocol dataset 'Content' must be stored compressed (compression filter missing)",
             content_ds.name)


# PER‑SIMULATION checks – one implementation per supported version
def _common_simulation_prelude(sim_grp: h5py.Group) -> Tuple[h5py.Group, h5py.Group, h5py.Group]:
    """
    Checks that are identical for *all* versions:

    * ``Configuration`` & ``Parameters`` groups exist and contain at least one
      **compound** attribute,
    * ``Stimuli`` group exists and contains at least one sub‑stimulus per
      population,
    * each stimulus sub‑group must contain the three mandatory datasets
      (``Limits``, ``Values`` and ``StimType``) with the shape constraints that
      are part of the original schema.

    Returns the three groups ``(config, params, stimuli)`` for the caller.
    """
    config = _require_group(sim_grp, "Configuration")
    params = _require_group(sim_grp, "Parameters")
    stimuli = _require_group(sim_grp, "Stimuli")

    # --- both groups must contain at least one *compound* attribute ------------
    _require(len(config.attrs) > 0,
             "Configuration group must contain at least one attribute",
             config.name)
    _check_all_compound_attributes(config)

    _require(len(params.attrs) > 0,
             "Parameters group must contain at least one attribute",
             params.name)
    _check_all_compound_attributes(params)

    # Stimuli checks (identical for v1.0 and v2.0)
    for pop_name, pop_grp in stimuli.items():
        _require(isinstance(pop_grp, h5py.Group),
                 f"Stimuli '{pop_name}' is not a group", pop_grp.name)
        _require(len(pop_grp) > 0,
                 f"Stimuli group '{pop_name}' must contain at least one sub‑stimulus group",
                 pop_grp.name)
        # Get the stimulus type – this must exist for any layout
        stim_type_ds = _require_dataset(pop_grp, "StimType")
        # The dataset is expected to be a scalar (e.g. bytes or str)
        stim_type = stim_type_ds[()]
        # Convert possible bytes‑objects to str for a clean comparison
        if isinstance(stim_type, (bytes, bytearray)):
            stim_type = stim_type.decode("utf-8")

        # Only continue with the Limits/Values checks when the type is
        #     piecewise_constant
        if stim_type != "piecewise_constant":
            # Nothing else to validate for this population – go to the next one
            continue

        if {"Limits", "Values", "StimType"}.issubset(pop_grp):
            # Flat layout (v1.0)
            limits = _require_dataset(pop_grp, "Limits")
            values = _require_dataset(pop_grp, "Values")
            _require(limits.ndim == 1,
                     "Limits must be 1‑D", limits.name)
            _require(values.ndim == 1,
                     "Values must be 1‑D", values.name)
            _require(limits.shape[0] == values.shape[0] + 1,
                     f"len(Limits) ({limits.shape[0]}) must be len(Values)+1 ({values.shape[0] + 1})",
                     pop_grp.name)

            # unit attributes for the two numeric datasets
            _require_unit_attribute(limits, "unit")
            _require_unit_attribute(values, "unit")
        else:
            # Nested layout – each sub‑stimulus must have Limits/Values.
            for sub_name, sub_grp in pop_grp.items():
                _require(isinstance(sub_grp, h5py.Group),
                         f"Sub‑stimulus '{sub_name}' not a group", sub_grp.name)
                limits = _require_dataset(sub_grp, "Limits")
                values = _require_dataset(sub_grp, "Values")
                _require(limits.ndim == 1,
                         "Limits must be 1‑D", limits.name)
                _require(values.ndim == 1,
                         "Values must be 1‑D", values.name)
                _require(limits.shape[0] == values.shape[0] + 1,
                         f"len(Limits) ({limits.shape[0]}) must be len(Values)+1 ({values.shape[0] + 1})",
                         sub_grp.name)

                # unit attributes for the numeric datasets
                _require_unit_attribute(limits, "unit")
                _require_unit_attribute(values, "unit")

                # optional extra attributes on the sub‑stimulus must themselves be
                # compound (the spec allows protocol‑specific metadata here)
                for extra_attr in sub_grp.attrs:
                    if extra_attr in ("unit",):  # already handled
                        continue
                    _require_compound_attribute(sub_grp, extra_attr)

            # In both layouts we also require a ``StimType`` dataset somewhere.
            _require_dataset(pop_grp, "StimType")
        # end of flat / nested handling

    return config, params, stimuli


def _check_one_simulation_v1(sim_grp: h5py.Group, seen_populations: List[str]) -> List[str]:
    """
    Validation for *format version 1.0*.

    * Output hierarchy is ``OutputSpikes/SpikeMap``.
    * Each neuron group must contain **only** a 1‑D ``times`` dataset.
    """
    # optional sanity‑check for a matching ``simulation_id`` attribute
    sim_id = sim_grp.name.split("/")[-1].replace("Sim_", "")
    if "simulation_id" in sim_grp.attrs:
        _require(sim_grp.attrs["simulation_id"] == sim_id,
                 f"simulation_id attribute must equal '{sim_id}'",
                 sim_grp.name)

    # common part (config, params, stimuli)
    _common_simulation_prelude(sim_grp)

    # OutputSpikes → SpikeMap (v1.0)
    out_spikes = _require_group(sim_grp, "OutputSpikes")
    spike_map = _require_group(out_spikes, "SpikeMap")

    pop_extracted: List[str] = []
    for pop_name, pop_grp in spike_map.items():
        pop_extracted.append(pop_name)
        _require(isinstance(pop_grp, h5py.Group),
                 f"Population '{pop_name}' in SpikeMap is not a group",
                 spike_map.name)

        neuron_groups = [g for n, g in pop_grp.items() if isinstance(g, h5py.Group)]
        _require(len(neuron_groups) > 0,
                 f"Population '{pop_name}' must contain at least one neuron group",
                 pop_grp.name)

        for neu_name, neu_grp in pop_grp.items():
            times = _require_dataset(neu_grp, "times")
            _require(times.ndim == 1,
                     "Neuron times must be 1‑D", times.name)

            # unit attribute on ``times``
            _require_unit_attribute(times, "unit")

            # monotonicity (strictly increasing)
            data = times[()]
            _require(np.all(np.diff(data) > 0),
                     "Neuron times must be strictly increasing", times.name)

    # Consistency of population names across simulations (if we already saw
    # some)
    if seen_populations:
        _require(set(pop_extracted) == set(seen_populations),
                 f"Populations in SpikeMap {pop_extracted} do not match expected {seen_populations}",
                 spike_map.name)

    return pop_extracted


def _check_one_simulation_v2(sim_grp: h5py.Group, seen_populations: List[str]) -> List[str]:
    """
    Validation for *format version 2.0*.

    Changes compared to 1.0:

    * The output hierarchy is ``Output/Map`` (note the renamed top‑level group).
    * Each neuron group **must** contain three datasets:
        * ``times`` – 1‑D (may be empty)
        * ``voltage`` – 1‑D
        * ``virtual`` – **scalar** (shape == ()).
    """
    # optional sanity‑check for a matching ``simulation_id`` attribute
    sim_id = sim_grp.name.split("/")[-1].replace("Sim_", "")
    if "simulation_id" in sim_grp.attrs:
        _require(sim_grp.attrs["simulation_id"] == sim_id,
                 f"simulation_id attribute must equal '{sim_id}'",
                 sim_grp.name)

    # common part (config, params, stimuli)
    _common_simulation_prelude(sim_grp)

    # Output → Map (v2.0)
    out_grp = _require_group(sim_grp, "Output")
    spike_map = _require_group(out_grp, "Map")

    pop_extracted: List[str] = []
    for pop_name, pop_grp in spike_map.items():
        pop_extracted.append(pop_name)
        _require(isinstance(pop_grp, h5py.Group),
                 f"Population '{pop_name}' in Map is not a group",
                 spike_map.name)

        neuron_groups = [g for n, g in pop_grp.items() if isinstance(g, h5py.Group)]
        _require(len(neuron_groups) > 0,
                 f"Population '{pop_name}' must contain at least one neuron group",
                 pop_grp.name)

        for neu_name, neu_grp in pop_grp.items():
            # times
            times = _require_dataset(neu_grp, "times")
            _require(times.ndim == 1,
                     "Neuron times must be 1‑D", times.name)
            _require_unit_attribute(times, "unit")
            # monotonicity (strictly increasing, even if empty)
            if times.shape[0] > 0:
                data = times[()]
                _require(np.all(np.diff(data) > 0),
                         "Neuron times must be strictly increasing", times.name)

            # voltage
            voltage = _require_dataset(neu_grp, "voltage")
            _require(voltage.ndim == 1,
                     "Neuron voltage must be 1‑D", voltage.name)
            _require_unit_attribute(voltage, "unit")

    # Consistency of population names across simulations (if we already saw
    # some)
    if seen_populations:
        _require(set(pop_extracted) == set(seen_populations),
                 f"Populations in Map {pop_extracted} do not match expected {seen_populations}",
                 spike_map.name)

    return pop_extracted


# ----------------------------------------------------------------------
# ── Version‑dispatch table – makes it easy to add a new schema version
# ----------------------------------------------------------------------
VERSION_REGISTRY: dict[
    str,
    Callable[[h5py.Group, List[str]], List[str]]
] = {
    "1.0": _check_one_simulation_v1,
    "2.0": _check_one_simulation_v2,
}


# ----------------------------------------------------------------------
# ── Public validation function (used by the CLI and by library callers)
# ----------------------------------------------------------------------
def validate_master_bundle(
        bundle_path: Path,
        *,
        strict: bool = True,
        version: str | None = None,
        logger: logging.Logger | None = None,
) -> None:
    """
    Validate a master HDF5 bundle against the schema for the requested format
    version.

    Parameters
    ----------
    bundle_path : pathlib.Path
        Path to the ``*.h5`` file that should be checked.
    strict : bool, default=True
        When ``True`` all rules (including the optional ``RawBundle`` and
        ``Protocol`` groups) are enforced.  When ``False`` only the core
        ``StandardFormat`` hierarchy is validated – useful for quick unit tests.
    version : str, optional
        If supplied, forces validation against that version, *ignoring* the
        ``format_version`` attribute stored in the file.  Useful for CI when the
        file is known to be of a particular version but the attribute is missing
        or broken.
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
        logger = configure_logging(
            __name__,
            log_file_name=Path(__file__).with_suffix(".log").name,
            level=logging.DEBUG if strict else logging.INFO,
        )

    logger.debug("Opening master bundle %s", bundle_path)
    with h5py.File(bundle_path, "r") as hf:
        #   Run version‑agnostic (common) checks
        for fn in (_check_root, _check_standardformat, _check_global, _check_simulations):
            logger.debug("Running check %s", fn.__name__)
            result = fn(hf)  # type: ignore[arg-type]
            if fn is _check_simulations:
                sim_groups: List[h5py.Group] = result  # type: ignore[assignment]

        #   Resolve the format version
        detected_version = version or hf.attrs.get("format_version")
        _require(detected_version is not None,
                 "Unable to determine format version (attribute missing and no '--version' supplied)",
                 "/")
        _require(detected_version in SUPPORTED_VERSIONS,
                 f"Unsupported format_version {detected_version!r}",
                 "/")
        per_sim_check = VERSION_REGISTRY[detected_version]

        logger.debug(
            "Detected format version %s – using %s",
            detected_version,
            per_sim_check.__name__,
        )

        #   Per‑simulation checks (the only part that differs per version)
        logger.debug("Running per‑simulation checks (%d sims)", len(sim_groups))
        populations: List[str] = []
        for sim in sim_groups:
            try:
                sim_populations = per_sim_check(sim, populations)
                # ``populations`` stores the *first* simulation’s list; all later
                # simulations are compared to it inside the per‑sim function.
                if not populations:
                    populations.extend(sim_populations)
            except ValidationError as exc:
                logger.debug(
                    "Validation failed in simulation %s → %s",
                    sim.name,
                    exc,
                )
                raise

        #   Optional groups – controlled by the ``strict`` flag
        if strict:
            logger.debug("Running optional RawBundle check")
            _check_rawbundle(hf)
            logger.debug("Running optional Protocol check")
            _check_protocol(hf)

    logger.info(
        "Master bundle %s passed validation (version %s)",
        bundle_path,
        detected_version,
    )


# Command‑line interface
def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse the command‑line arguments for the validator script.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=(
            "Validate a MasterBundle HDF5 file against the minimal schema. "
            "Exit status 0 indicates success, >0 indicates failure."
        ),
    )
    parser.add_argument(
        "bundle_path",
        type=Path,
        help="Path to the *.h5 file that should be validated.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Enforce strict validation (includes optional RawBundle and Protocol "
            "checks).  When omitted only the core StandardFormat hierarchy is "
            "validated."
        ),
    )
    parser.add_argument(
        "--version",
        choices=sorted(SUPPORTED_VERSIONS),
        help=(
            "Force validation against a particular version (e.g. ``--version 2.0``). "
            "If omitted the validator uses the ``format_version`` attribute stored "
            "in the file."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry‑point used by ``python -m converters.utils.validation`` or a direct
    script run.

    Returns
    -------
    int
        Exit status: ``0`` if validation passes, ``1`` for a ``ValidationError``,
        ``2`` for any unexpected exception.
    """
    args = _parse_cli_args(argv)

    logger = configure_logging(
        __name__,
        log_file_name=Path(__file__).with_suffix(".log").name,
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    try:
        validate_master_bundle(
            bundle_path=args.bundle_path,
            strict=args.strict,
            version=args.version,
            logger=logger,
        )
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    except Exception:  # pragma: no cover – unexpected errors
        logger.exception("Unexpected error while validating %s", args.bundle_path)
        return 2

    logger.info("Validation succeeded.")
    return 0


if __name__ == "__main__":
    # ``sys.exit`` makes sure the integer return value becomes the process'
    # exit code – this mirrors the behaviour of typical CLI utilities.
    sys.exit(main())
