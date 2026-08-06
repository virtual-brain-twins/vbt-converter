"""h5py‑compatible container for generic key‑value pairs.

This module provides :class:`~KeyValueCompoundContainer`, a lightweight
dataclass that stores an arbitrary mapping of ``key → (value, unit)`` where
*value* can be any JSON‑serialisable object (or any type supported by the
``write_compound_attr`` helper) and *unit* is an optional
:class:`~utils.units.Unit` instance.  The container knows how to serialize
itself to an HDF5 group using compound attributes and to reconstruct itself
from such a group.

Typical usage
-------------

... from mypackage.keyvalue import KeyValueCompoundContainer
... store = KeyValueCompoundContainer()
... store.set("temperature", 300, Unit("K"))
... store["pressure"] = (101325, Unit("Pa"))
... with h5py.File("example.h5", "w") as f:
...     grp = f.create_group("sensor_data")
...     store.to_hdf5(grp)
...
... # later …
... with h5py.File("example.h5", "r") as f:
...     grp = f["sensor_data"]
...     loaded = KeyValueCompoundContainer.from_hdf5(grp)
... 
... loaded.get_value("temperature")
300

The class deliberately contains no domain‑specific validation; such logic can
be added in subclasses that override :meth:`set` or provide convenience
methods.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, Type, TypeVar, ClassVar

import h5py

from .hdf5_serializable import HDF5Serializable
from ..utils.compound_attribute import write_compound_attr, read_compound_attr
from ..utils.units import Unit

# Generic type that represents a subclass of the container itself.
T = TypeVar("T", bound="KeyValueCompoundContainer")


@dataclass
class KeyValueCompoundContainer(HDF5Serializable):
    """
    A reusable container that stores arbitrary ``key → (value, unit)`` pairs
    as **compound** HDF5 attributes.

    The class is deliberately lightweight – it only knows how to (de‑)serialize
    the ``kv_pairs`` mapping.  Any domain‑specific semantics (e.g. validation,
    default values, convenience helpers) should be added in a subclass.
    """
    # Public payload – a mapping of *name* → *(value, optional Unit)*
    kv_pairs: Dict[str, Tuple[Any, Optional[Unit]]] = field(default_factory=dict)

    # Optional class attribute: the name that will be used for the HDF5 group
    # when these objects are stored via ``HDF5Serializable`` utilities.
    # Sub‑classes can override it to get a nicer group name.
    _group_name: ClassVar[str] = "KeyValueCompoundContainer"

    # Dictionary‑like helpers
    def __getitem__(self, key: str) -> Tuple[Any, Optional[Unit]]:
        """
        ``obj[key]`` → ``(value, unit)``

        Mirrors ``dict.__getitem__``; raises ``KeyError`` if *key* is absent.
        """
        return self.kv_pairs[key]

    def __setitem__(self, key: str, pair: Tuple[Any, Optional[Unit]]) -> None:
        """
        ``obj[key] = (value, unit)``

        Shortcut that forwards to :meth:`set`.  ``pair`` must be a 2‑tuple.
        """
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("value must be a tuple of the form (value, unit)")
        value, unit = pair
        self.set(key, value, unit)

    def __contains__(self, key: object) -> bool:
        """``key in obj`` – true if *key* is present in the internal mapping."""
        return key in self.kv_pairs

    # Public API for reading values
    def get(self, key: str) -> Tuple[Any, Optional[Unit]]:
        """
        Return the raw ``(value, unit)`` tuple stored under *key*.

        Parameters
        ----------
        key:
            The identifier of the entry.

        Returns
        -------
        (value, unit)

        Raises
        ------
        KeyError
            If *key* does not exist in the container.
        """
        try:
            return self.kv_pairs[key]
        except KeyError as exc:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}") from exc

    def get_value(self, key: str) -> Any:
        """
        Return **only** the value component for *key*.

        Raises
        ------
        KeyError
            If *key* is absent.
        """
        return self.get(key)[0]

    def get_unit(self, key: str) -> Optional[Unit]:
        """
        Return **only** the unit component for *key* (may be ``None``).

        Raises
        ------
        KeyError
            If *key* is absent.
        """
        return self.get(key)[1]

    # Public API for writing values
    def set(self, key: str, value: Any, unit: Optional[Unit] = None) -> None:
        """
        Store ``(value, unit)`` under *key*, overwriting any existing entry.

        Parameters
        ----------
        key:
            Identifier for the entry.
        value:
            Anything that is JSON‑serialisable or that can be written by the
            ``write_compound_attr`` helper (e.g. numbers, strings, numpy scalars,
            ``np.ndarray`` …).
        unit:
            Optional :class:`~utils.units.Unit` instance.  ``None`` means “unit‑less”.

        Notes
        -----
        - The method does **not** perform any validation beyond type‑checking
          the ``unit`` argument.  If you need domain‑specific checks, subclass
          and override this method (calling ``super().set`` at the end).
        """
        if unit is not None and not isinstance(unit, Unit):
            raise TypeError(
                f"The `unit` argument must be a `Unit` instance or None, got {type(unit)!r}"
            )
        self.kv_pairs[key] = (value, unit)

    def pop(self, key: str, default: Any = ...) -> Tuple[Any, Optional[Unit]]:
        """
        Remove *key* from the container and return its ``(value, unit)`` tuple.

        If *default* is supplied and *key* is missing, that value is returned
        instead of raising ``KeyError`` – mimicking ``dict.pop``.

        Parameters
        ----------
        key:
            Identifier to delete.
        default:
            Optional fallback value; if omitted the method behaves like
            ``dict.pop`` and raises ``KeyError`` when the key is absent.

        Returns
        -------
        (value, unit) or *default*.
        """
        if default is ...:
            # No default supplied → let dict raise the proper KeyError
            return self.kv_pairs.pop(key)
        else:
            return self.kv_pairs.pop(key, default)

    # Serialization helpers – **shared by every subclass**
    def to_hdf5(self, parent: h5py.Group) -> None:
        """
        Write every ``key → (value, unit)`` entry as a compound attribute
        on ``parent``.
        """
        for key, (value, unit) in self.kv_pairs.items():
            write_compound_attr(parent, key, value, unit)

    @classmethod
    def from_hdf5(cls: Type[T], grp: h5py.Group) -> T:
        """
        Re‑create an instance from a group that contains only compound
        attributes.
        """
        loaded: Dict[str, Tuple[Any, Optional[Unit]]] = {}
        for key in grp.attrs:
            value, unit = read_compound_attr(grp, key)
            loaded[key] = (value, unit)
        return cls(kv_pairs=loaded)
