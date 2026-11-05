# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

"""h5py compound attribute utilities.

This module provides helper functions for storing and retrieving *compound*
attributes in an HDF5 file.  A compound attribute consists of a ``value`` field
(which may be a native HDF5 type or a JSON‑encoded string) and an associated
``unit`` field (a UTF‑8 string).  The functions abstract away the details of
choosing the correct NumPy/HDF5 data type and handling unit normalisation.

Typical usage::

    from converters.utils.units import Unit
    import h5py

    with h5py.File("example.h5", "a") as f:
        grp = f.require_group("metadata")
        write_compound_attr(grp, "temperature", 300, Unit("K"))
        value, unit = read_compound_attr(grp, "temperature")
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple, Union

import h5py
import numpy as np

from converters.utils.units import Unit


def _dtype_for_value(value: Any) -> np.dtype:
    """Return the appropriate NumPy dtype for a scalar ``value``.

    The function maps Python scalar types to the corresponding HDF5/NumPy data
    types that can be stored directly in a compound attribute.  If the value
    type cannot be represented natively, a :class:`TypeError` is raised so that
    the caller can fall back to JSON encoding.

    Args:
        value: The scalar value whose dtype is required.

    Returns:
        A :class:`numpy.dtype` suitable for storing ``value`` in an HDF5
        dataset/attribute.

    Raises:
        TypeError: If ``value`` is of a type that cannot be stored directly
            (e.g., list, dict, custom object).  The caller should encode such
            values as a JSON string before writing.
    """
    if isinstance(value, bool):
        return np.dtype(np.bool_)
    if isinstance(value, int):
        # HDF5 does not have a signed‑int with arbitrary size; int64 is safe.
        return np.dtype(np.int64)
    if isinstance(value, float):
        return np.dtype(np.float64)

    if isinstance(value, str):
        # Variable‑length UTF‑8 string (the only string type HDF5 supports)
        return h5py.string_dtype()

    # Anything else must be JSON‑encoded by the caller.
    raise TypeError(
        f"Value of type {type(value)!r} cannot be stored directly. "
        "Convert it to a JSON string before calling write_compound_attr."
    )


def write_compound_attr(
        parent: h5py.Group,
        name: str,
        value: Any,
        unit: Optional[Union[str, Unit]] = None,
) -> None:
    """Write a compound attribute ``(value, unit)`` to an HDF5 group.

    The attribute consists of two fields:

    * ``value`` – stored using the most appropriate native HDF5 dtype, or as a
      JSON‑encoded UTF‑8 string when the value type is not directly supported.
    * ``unit`` – always stored as a UTF‑8 string; an empty string indicates no
      unit.

    Args:
        parent: The HDF5 group (or file) whose ``attrs`` dictionary will receive
            the new attribute.
        name: Name of the attribute to create or overwrite.
        value: The value to store.  Scalars of type ``bool``, ``int``, ``float``,
            or ``str`` are stored directly; other types are JSON‑encoded.
        unit: Optional unit associated with ``value``.  It can be a plain string
            or a :class:`~converters.utils.units.Unit` instance.  ``None`` or an
            empty string denotes the absence of a unit.

    Raises:
        TypeError: If ``value`` cannot be represented natively and JSON encoding
            also fails (unlikely, since ``json.dumps`` accepts most serialisable
            objects).
        OSError: Propagated from the underlying HDF5 library if attribute creation
            fails (e.g., due to file I/O errors or insufficient permissions).
    """
    # Normalise the unit argument.
    if unit is None:
        unit_str = ""
    else:
        # ``Unit`` objects expose the underlying string via ``.value``.
        try:
            unit_str = unit.value  # type: ignore[attr-defined]
        except AttributeError:
            # Plain string supplied.
            unit_str = str(unit)

    # Ensure we have a native scalar. If not, fall back to JSON string.
    try:
        value_dtype = _dtype_for_value(value)
        stored_value = value
    except TypeError:
        # Anything that cannot be stored natively is forced into the JSON path.
        # The spec says: “value may be a JSON‑encoded UTF‑8 string”.
        stored_value = json.dumps(value, ensure_ascii=False)
        value_dtype = h5py.string_dtype()

    # Build the per‑attribute compound dtype --> (value, unit)
    compound_dtype = np.dtype(
        [
            ("value", value_dtype),
            ("unit", h5py.string_dtype()),
        ]
    )

    # Populate a scalar structured array and write it.
    compound = np.empty((), dtype=compound_dtype)
    compound["value"] = stored_value
    compound["unit"] = unit_str

    # ``attrs.create`` overwrites an existing attribute of the same name.
    parent.attrs.create(name, compound)


def _maybe_item(v: Any) -> Any:
    """
    Helper that calls ``.item()`` only if the object actually has that method
    (i.e. it is a NumPy scalar or zero‑dimensional array).  For plain Python
    scalars, strings, bytes … the value is returned unchanged.
    """
    try:
        # ``np.generic`` (e.g. np.float64) implements ``item``.
        # ``np.ndarray`` of shape () also implements it.
        return v.item()  # type: ignore[attr-defined]
    except AttributeError:
        return v


def read_compound_attr(parent: h5py.Group, name: str) -> Tuple[Any, Optional[Unit]]:
    """Read a compound attribute ``(value, unit)`` from an HDF5 group.

    The function returns the raw ``value`` field as stored (either the original
    scalar or a JSON‑encoded string) together with the associated unit, or
    ``None`` if the unit field is empty.

    Args:
        parent: The HDF5 group (or file) whose ``attrs`` dictionary contains the
            attribute.
        name: Name of the attribute to read.

    Returns:
        A tuple ``(value, unit)`` where ``value`` is the stored value (scalar or
        JSON string) and ``unit`` is either a Unit enum member or ``None`` when no unit
        was recorded.

    Raises:
        KeyError: If ``name`` does not exist in ``parent.attrs``.
        OSError: Propagated from the HDF5 library for I/O‑related failures.
    """
    compound = parent.attrs[name]

    # Detect the storage layout
    # Case A – a structured scalar (np.void) with two fields.
    if isinstance(compound, np.void) and compound.dtype.names is not None:
        # ``compound["value"]`` and ``compound["unit"]`` are themselves
        # NumPy scalars (or zero‑dim arrays).  Use the tiny helper to safely
        # extract the Python scalar.
        raw_value = _maybe_item(compound["value"])
        raw_unit = _maybe_item(compound["unit"])
    else:
        # Anything else is interpreted as a *plain* attribute that holds the
        # value only.  The unit is considered missing.
        raw_value = compound
        raw_unit = None

    # Normalise bytes → str (h5py may return bytes on some platforms)
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode("utf-8")
    if isinstance(raw_unit, (bytes, bytearray)):
        raw_unit = raw_unit.decode("utf-8")

    try:
        unit = None if raw_unit == "" or raw_unit is None else Unit(raw_unit)  # Enum lookup by value
    except ValueError as exc:
        raise ValueError(
            f"'{raw_unit}' is not a recognised unit. "
            f"Allowed units are: {[u.value for u in Unit]}"
        ) from exc

    # The spec does **not** require automatic JSON decoding.  We therefore
    # return the raw string; callers that know the field contains JSON can
    # decode it themselves.
    return raw_value, unit
