"""h5py compound attribute utilities.

This module provides helper functions for encoding arbitrary Python values
(and optional :class:`~converters.utils.units.Unit` objects) as a compound
HDF5 attribute.  The attribute stores the value's type, a serialized
representation, and an optional unit string.  Corresponding read functions
decode the stored data back into the original Python objects.

The implementation relies on a NumPy structured dtype with three string
fields (``type``, ``value`` and ``unit``) and uses JSON for most
serialisations, falling back to a simple hex representation for ``bytes``
and to ``json.dumps`` for any other non‑standard type.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

import h5py
import numpy as np

from converters.utils.units import Unit

_COMPOUND_DTYPE = np.dtype(
    [
        ("type", h5py.string_dtype()),
        ("value", h5py.string_dtype()),
        ("unit", h5py.string_dtype()),
    ]
)


def _encode_value(value: Any) -> tuple[str, str]:
    """Encode a Python value as a ``(type_name, serialized)`` pair.

    The function chooses a simple string identifier for the value's type and
    produces a JSON (or hex) representation that can later be decoded with
    :func:`_decode_value`.

    Parameters
    ----------
    value:
        The Python object to encode.  Supported types are ``None``, ``bool``,
        ``int``, ``float``, ``str``, ``bytes``, ``numpy.ndarray`` and any
        object that can be handled by :func:`json.dumps`.  Unrecognised types
        are stored as JSON.

    Returns
    -------
    tuple[str, str]
        ``type_name`` – a short identifier (e.g. ``"int"``, ``"ndarray"``);
        ``serialized`` – the JSON or hex string representing the value.
    """
    if value is None:
        return "none", ""

    if isinstance(value, bool):
        return "bool", json.dumps(value)

    if isinstance(value, int):
        return "int", json.dumps(value)

    if isinstance(value, float):
        return "float", json.dumps(value)

    if isinstance(value, str):
        return "str", value

    if isinstance(value, bytes):
        return "bytes", value.hex()

    if isinstance(value, np.ndarray):
        return "ndarray", json.dumps(value.tolist())

    return "json", json.dumps(value)


def _decode_value(type_name: str, value_str: str) -> Any:
    """Decode a value that was previously encoded by :func:`_encode_value`.

    Parameters
    ----------
    type_name:
        The identifier string stored alongside the value (e.g. ``"int"``,
        ``"ndarray"``, ``"json"``).
    value_str:
        The serialized representation of the original value.

    Returns
    -------
    Any
        The reconstructed Python object.

    Raises
    ------
    ValueError
        If ``type_name`` does not correspond to a supported stored type.
    """
    if type_name == "none":
        return None

    if type_name == "bool":
        return bool(json.loads(value_str))

    if type_name == "int":
        return int(json.loads(value_str))

    if type_name == "float":
        return float(json.loads(value_str))

    if type_name == "str":
        return value_str

    if type_name == "bytes":
        return bytes.fromhex(value_str)

    if type_name == "ndarray":
        return np.asarray(json.loads(value_str))

    if type_name == "json":
        return json.loads(value_str)

    raise ValueError(f"Unsupported stored type '{type_name}'")


def write_compound_attr(
        parent: h5py.Group,
        name: str,
        value: Any,
        unit: Optional[Unit] = None,
) -> None:
    """Write a compound attribute to an HDF5 group.

    The attribute stores the value's type, a serialized representation and an
    optional unit string.  It uses the ``_COMPOUND_DTYPE`` structured array
    defined in this module.

    Parameters
    ----------
    parent:
        The HDF5 ``Group`` (or ``File``) whose attributes will receive the new
        entry.
    name:
        The name under which to store the attribute.
    value:
        Any Python object that can be handled by :func:`_encode_value`.
    unit:
        An optional :class:`~converters.utils.units.Unit` enum instance that
        will be stored alongside the value.  If ``None`` (default), no unit is
        recorded.
    """
    type_name, serialized = _encode_value(value)

    compound = np.empty((), dtype=_COMPOUND_DTYPE)

    compound["type"] = type_name
    compound["value"] = serialized
    compound["unit"] = "" if unit is None else unit.value

    parent.attrs.create(name, compound)


def read_compound_attr(
        parent: h5py.Group,
        name: str,
) -> Tuple[Any, Optional[Unit]]:
    """Read a compound attribute previously written with :func:`write_compound_attr`.

    Parameters
    ----------
    parent:
        The HDF5 ``Group`` (or ``File``) containing the attribute.
    name:
        The attribute name to read.

    Returns
    -------
    tuple[Any, Optional[Unit]]
        A ``(value, unit)`` pair where ``value`` is the decoded Python object
        and ``unit`` is either a :class:`~converters.utils.units.Unit` enum
        instance or ``None`` if the attribute was stored without a unit.

    Raises
    ------
    ValueError
        If the stored unit string does not correspond to any value of the
        :class:`~converters.utils.units.Unit` enum.
    """
    compound = parent.attrs[name]

    type_name = compound["type"]
    value_str = compound["value"]
    unit_str = compound["unit"]

    if isinstance(type_name, bytes):
        type_name = type_name.decode("utf-8")

    if isinstance(value_str, bytes):
        value_str = value_str.decode("utf-8")

    if isinstance(unit_str, bytes):
        unit_str = unit_str.decode("utf-8")

    value = _decode_value(type_name, value_str)
    try:
        unit = None if unit_str == "" else Unit(unit_str)  # Enum lookup by value
    except ValueError as exc:
        raise ValueError(
            f"'{unit_str}' is not a recognised unit. "
            f"Allowed units are: {[u.value for u in Unit]}"
        ) from exc
    return value, unit