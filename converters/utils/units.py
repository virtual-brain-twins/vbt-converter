"""
Neurophysiology Units

A lightweight enumeration‑based definition of the most common units used in
neurophysiology (and related fields).  The module groups units into logical
categories and provides a mapping from each :class:`Unit` to its
corresponding :class:`UnitCategory`.  This information is useful for validation,
error reporting, and documentation generation without pulling in any heavy
dependency such as *pint*.

The file purposefully contains only data – no arithmetic or conversion logic.
All values are represented as human‑readable strings, making them suitable for
display in GUIs, logs, or API responses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Dict


# Category → Units mapping
class UnitCategory(StrEnum):
    """High‑level categories that group related physical units.

    Each member’s value is a human‑readable description of the category.
    """
    TIME = "Time"
    ELECTRICAL_POTENTIAL = "Electrical Potential"
    ELECTRICAL_CURRENT = "Electrical Current"
    RESISTANCE_CONDUCTANCE = "Electrical Resistance & Conductance"
    CAPACITANCE = "Electrical Capacitance"
    FREQUENCY_RHYTHMS = "Frequency & Rhythms"
    CONCENTRATION = "Concentration & Chemical"
    SPATIAL = "Spatial & Morphology"


class Unit(StrEnum):
    """Enumeration of individual units used in neurophysiological recordings.

    The members are grouped by the logical categories defined in
    :class:`UnitCategory`.  Values are lower‑case strings that are suitable for
    direct display to end‑users.
    """
    # Time
    MILLISECOND = "milliseconds"
    MICROSECOND = "microseconds"
    NANOSECOND = "nanoseconds"
    SECOND = "seconds"
    MINUTE = "minutes"

    # Electrical Potential
    MILLIVOLT = "millivolts"
    MICROVOLT = "microvolts"

    # Electrical Current
    PICOAMPERE = "picoamperes"
    NANOAMPERE = "nanoamperes"
    MICROAMPERE = "microamperes"

    # Resistance & Conductance
    MEGAOHM = "megaohms"
    GIGAOHM = "gigaohms"
    NANOSIEMENS = "nanosiemens"
    PICOSIEMENS = "picosiemens"

    # Capacitance
    PICOFARAD = "picofarad"
    MICROFARAD_PER_CM2 = "microfarad per square centimeter"

    # Frequency & Rhythms
    HERTZ = "hertz"

    # Concentration & Chemical
    MILLIMOLAR = "millimolar"
    MICROMOLAR = "micromolar"
    NANOMOLAR = "nanomolar"

    # Spatial & Morphology
    MICROMETER = "micrometers"
    MILLIMETER = "millimeters"
    NANO_METER = "nanometers"


# Mapping from Unit → Category – useful for error messages, reporting, etc.
UNIT_TO_CATEGORY: Final[Dict[Unit, UnitCategory]] = {
    Unit.MILLISECOND: UnitCategory.TIME,
    Unit.MICROSECOND: UnitCategory.TIME,
    Unit.NANOSECOND: UnitCategory.TIME,
    Unit.SECOND: UnitCategory.TIME,
    Unit.MINUTE: UnitCategory.TIME,

    Unit.MILLIVOLT: UnitCategory.ELECTRICAL_POTENTIAL,
    Unit.MICROVOLT: UnitCategory.ELECTRICAL_POTENTIAL,

    Unit.PICOAMPERE: UnitCategory.ELECTRICAL_CURRENT,
    Unit.NANOAMPERE: UnitCategory.ELECTRICAL_CURRENT,
    Unit.MICROAMPERE: UnitCategory.ELECTRICAL_CURRENT,

    Unit.MEGAOHM: UnitCategory.RESISTANCE_CONDUCTANCE,
    Unit.GIGAOHM: UnitCategory.RESISTANCE_CONDUCTANCE,
    Unit.NANOSIEMENS: UnitCategory.RESISTANCE_CONDUCTANCE,
    Unit.PICOSIEMENS: UnitCategory.RESISTANCE_CONDUCTANCE,

    Unit.PICOFARAD: UnitCategory.CAPACITANCE,
    Unit.MICROFARAD_PER_CM2: UnitCategory.CAPACITANCE,

    Unit.HERTZ: UnitCategory.FREQUENCY_RHYTHMS,

    Unit.MILLIMOLAR: UnitCategory.CONCENTRATION,
    Unit.MICROMOLAR: UnitCategory.CONCENTRATION,
    Unit.NANOMOLAR: UnitCategory.CONCENTRATION,

    Unit.MICROMETER: UnitCategory.SPATIAL,
    Unit.MILLIMETER: UnitCategory.SPATIAL,
    Unit.NANO_METER: UnitCategory.SPATIAL,
}