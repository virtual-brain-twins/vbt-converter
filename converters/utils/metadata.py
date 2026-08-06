"""
Utility for building a ``meta_data_map`` that is guaranteed to contain only
the standardised attribute keys defined by the consortium.

Typical usage
-------------

>> from meta_data import MetaDataBuilder, StandardKey

>> mb = MetaDataBuilder()
>> mb.set(StandardKey.author, "John Doe")
>> mb.set(StandardKey.creation_date, "2024‑03‑15T12:34:56Z")
>> mb.set(StandardKey.software_version, "1.2.3", unit=None)
>> meta = mb.to_dict()

>> meta
{
    StandardKey.author:          ("John Doe", None),
    StandardKey.creation_date:  ("2024-03-15T12:34:56Z", None),
    StandardKey.software_version:("1.2.3", None),
}
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from converters.utils.units import Unit


# Canonical key list – keep this source‑of‑truth in sync with the docs
class StandardKey(str, Enum):
    """
    Enum that holds the *exact* strings that are allowed as keys in the
    ``meta_data_map``.  By inheriting from ``str`` the enum members behave like
    normal strings (e.g. they are JSON‑serialisable) while still giving us
    compile‑time checking.
    """
    AUTHOR = "author"
    AFFILIATION = "affiliation"
    CREATION_DATE = "creation_date"
    PROJECT = "project"
    SIMULATION_NAME = "simulation_name"
    RUN_ID = "run_id"
    SOFTWARE = "software"
    SOFTWARE_VERSION = "software_version"
    INPUT_DATA = "input_data"
    LICENSE = "license"
    CONTACT_EMAIL = "contact_email"
    COMMENT = "comment"

    # Human‑readable representation (nice for debugging / printing)
    def __repr__(self) -> str:
        return f"<StandardKey.{self.name}: {self.value!r}>"


# Type alias for the public data structure
MetaDataMap = Dict[StandardKey, Tuple[Any, Optional[Unit]]]


# Core helper – validates keys and keeps the underlying dict consistent
class MetaDataBuilder:
    """
    Builder / thin wrapper around a plain ``dict`` that enforces
    *only* ``StandardKey`` entries and stores each value as a ``(value,
    unit)`` tuple.

    The builder can be reused (you can call ``set`` many times) and finally
    turned into a plain ``dict`` via :meth:`to_dict`.

    Parameters
    ----------
    initial : optional mapping
        Pre‑populated values; will be validated the same way as subsequent
        ``set`` calls.
    """

    __slots__ = ("_data",)

    def __init__(self, initial: Optional[Mapping[StandardKey, Tuple[Any, Optional[Unit]]]] = None) -> None:
        self._data: MetaDataMap = {}
        if initial:
            for k, v in initial.items():
                self.set(k, v[0], v[1])

    # Public API
    def set(
            self,
            key: StandardKey,
            value: Any,
            unit: Optional[Unit] = None,
            *,
            overwrite: bool = True,
    ) -> None:
        """
        Insert or update a metadata entry.

        Parameters
        ----------
        key : StandardKey
            One of the enumerated canonical keys.
        value : Any
            The actual datum to store.
        unit : str | None, optional
            Unit of the value (e.g. "ms", "µV") – ``None`` for unit‑less items.
        overwrite : bool, default ``True``
            If ``False`` an attempt to replace an existing entry raises
            :class:`KeyError`.  This is handy when you want to protect a
            field that should be written only once.

        Raises
        ------
        TypeError
            If ``key`` is not a member of :class:`StandardKey`.
        KeyError
            If ``overwrite`` is ``False`` and the key already exists.
        """
        if not isinstance(key, StandardKey):
            raise TypeError(
                f"Metadata keys must be members of StandardKey, got {type(key)!r}"
            )
        if not overwrite and key in self._data:
            raise KeyError(f"Key {key!s} already set – overwrite=False")
        self._data[key] = (value, unit)

    def get(self, key: StandardKey) -> Tuple[Any, Optional[str]]:
        """Retrieve the stored ``(value, unit)`` tuple for *key* (KeyError if missing)."""
        return self._data[key]

    def to_dict(self) -> MetaDataMap:
        """
        Return a **shallow copy** of the underlying dict.

        The copy protects the internal state from accidental external mutation.
        """
        return dict(self._data)

    # Convenience “semantic” shortcuts – optional but often handy
    # The following methods are thin wrappers that make the API a tad more
    # readable (e.g. ``mb.author("John")`` instead of ``mb.set(StandardKey.AUTHOR,… )``).

    def author(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.AUTHOR, value, unit, **kw)

    def affiliation(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.AFFILIATION, value, unit, **kw)

    def creation_date(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.CREATION_DATE, value, unit, **kw)

    def project(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.PROJECT, value, unit, **kw)

    def simulation_name(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.SIMULATION_NAME, value, unit, **kw)

    def run_id(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.RUN_ID, value, unit, **kw)

    def software(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.SOFTWARE, value, unit, **kw)

    def software_version(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.SOFTWARE_VERSION, value, unit, **kw)

    def input_data(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.INPUT_DATA, value, unit, **kw)

    def license(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.LICENSE, value, unit, **kw)

    def contact_email(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.CONTACT_EMAIL, value, unit, **kw)

    def comment(self, value: Any, unit: Optional[Unit] = None, **kw: Any) -> None:
        self.set(StandardKey.COMMENT, value, unit, **kw)


# Convenience factory – one‑liner for quick dict construction
def build_meta_data(**kwargs: Tuple[Any, Optional[Unit]]) -> MetaDataMap:
    """
    One‑shot helper that builds a ``meta_data_map`` from keyword arguments.

    Example
    -------
    >>> md = build_meta_data(
    ...     author=("John Doe", None),
    ...     creation_date=("2024-01-01T00:00:00Z", None),
    ...     software_version=("2.5.1", None),
    ... )
    >>> md[StandardKey.author]
    ('John Doe', None)

    Notes
    -----
    * Keys must be *exact* members of :class:`StandardKey`.  The function raises
      ``KeyError`` otherwise.
    * Values must be supplied as a 2‑tuple ``(value, unit)`` – this mirrors the
      internal representation.
    """
    builder = MetaDataBuilder()
    for raw_key, pair in kwargs.items():
        if not isinstance(pair, tuple) or not (1 <= len(pair) <= 2):
            raise TypeError(
                f"Value for '{raw_key}' must be a tuple of (value, unit?)"
            )
        # Normalise a one‑element tuple to (value, None)
        value, *unit_opt = pair
        unit = unit_opt[0] if unit_opt else None

        # Convert raw_key (string) to the Enum – raises ValueError if unknown
        try:
            key = StandardKey(raw_key)
        except ValueError as exc:
            raise KeyError(f"Unsupported meta‑data key: {raw_key!r}") from exc
        norm_unit = None
        if unit is not None:
            if isinstance(unit, Unit):
                norm_unit = unit
            elif isinstance(unit, str):
                try:
                    norm_unit = Unit(unit)  # Enum lookup by value
                except ValueError as exc:
                    raise ValueError(
                        f"'{unit}' is not a recognised unit. "
                        f"Allowed units are: {[u.value for u in Unit]}"
                    ) from exc
            else:
                raise TypeError(
                    f"unit must be a `Unit` enum member or a string, got {type(unit)}"
                )

        builder.set(key, value, norm_unit)
    return builder.to_dict()
