# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the :class:`SimulationParameters` data class, which is a
light‑weight container for simulation configuration values.  It inherits from
:class:`~compound_kv_container.KeyValueCompoundContainer` so that the
parameters can be accessed and manipulated using the key‑value semantics
provided by that base class.

The class itself does not implement any behaviour; it merely supplies a
``_group_name`` constant that is used by the container infrastructure to
group related settings together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .compound_kv_container import KeyValueCompoundContainer


@dataclass
class SimulationParameters(KeyValueCompoundContainer):
    """Container for simulation‑specific parameters.

    The class is a thin wrapper around :class:`KeyValueCompoundContainer`,
    providing a dedicated ``_group_name`` that identifies the group of values
    as belonging to simulation parameters.  Instances behave like dictionaries
    (or other mapping types) as defined by the parent container class.
    """

    _group_name: ClassVar[str] = "SimulationParameters"
