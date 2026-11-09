# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""Configuration module.

This module defines the :class:`Configuration` dataclass, a concrete subclass
of :class:`~compound_kv_container.KeyValueCompoundContainer`.  It is used to
store simulation‑wide configuration parameters in an HDF5 file.  The class does
not introduce any new fields; its primary purpose is to provide a clearly
named group (``/Configuration``) within the file hierarchy.

The implementation relies on the generic key‑value container functionality
provided by :class:`KeyValueCompoundContainer`, while overriding the default
group name to keep the hierarchy tidy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .compound_kv_container import KeyValueCompoundContainer


@dataclass
class Configuration(KeyValueCompoundContainer):
    """
    Holds simulation‑wide configuration parameters.

    The container does **not** add any new fields – it only exists so that the
    file layout can contain a clearly‑named group (e.g. ``/Configuration``).
    """
    # Override the default group name so that the HDF5 hierarchy looks tidy.
    _group_name: ClassVar[str] = "Configuration"
