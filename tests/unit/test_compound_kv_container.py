# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""
pytest test suite for ``mypackage.keyvalue.KeyValueCompoundContainer``

All heavy‑weight dependencies (h5py and the compound‑attribute helpers) are
mocked.  The suite demonstrates how to:

* unit‑test the public API (set/get/pop/keys/iter_keys)
* verify that the serialization helpers are called with the expected
  arguments
* attach useful artefacts (input data, resulting payload, mock call args)
* organise the report with Allure titles, descriptions, features, stories,
  severities and step‑wise execution traces.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

import pytest
import allure
from unittest import mock

from converters.schema.compound_kv_container import KeyValueCompoundContainer
from converters.utils.units import Unit

# --------------------------------------------------------------------------- #
#   Global Allure configuration (optional, can be done in pytest‑allure.ini)
# --------------------------------------------------------------------------- #
ALLURE_FEATURE = "KeyValueCompoundContainer"
ALLURE_STORY_SERIALISATION = "Serialisation (to_hdf5 / from_hdf5)"
ALLURE_STORY_MANIPULATION = "Basic container manipulation"


# --------------------------------------------------------------------------- #
#   Test class
# --------------------------------------------------------------------------- #
@allure.feature(ALLURE_FEATURE)
class TestKeyValueCompoundContainer:
    """
    Test suite for :class:`KeyValueCompoundContainer`.

    All heavy‑weight calls are mocked.  The suite is deliberately split into
    two logical parts – *container manipulation* and *serialisation* – each
    receiving its own Allure story tag.
    """

    # ------------------------------------------------------------------- #
    #   Optional per‑test setup / teardown – useful for attaching state
    # ------------------------------------------------------------------- #
    def setup_method(self) -> None:
        """Create a fresh container before each test and attach it to the report."""
        self.container = KeyValueCompoundContainer()
        allure.attach(
            body=str(self.container.kv_pairs),
            name="Initial kv_pairs (empty)",
            attachment_type=allure.attachment_type.JSON,
        )

    def teardown_method(self) -> None:
        """Clear the container reference – helps the GC and makes the intent explicit."""
        del self.container

    # ------------------------------------------------------------------- #
    #   1️⃣  Manipulation tests
    # ------------------------------------------------------------------- #

    @allure.story(ALLURE_STORY_MANIPULATION)
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Set a value with a unit and retrieve it via the public API")
    @allure.description(
        """
        Verify that :meth:`KeyValueCompoundContainer.set` stores a ``(value,
        unit)`` tuple correctly and that the read‑only helpers
        :meth:`get`, :meth:`get_value` and :meth:`get_unit` return the expected
        pieces.  The test also checks for proper ``KeyError`` handling.
        """
    )
    def test_set_and_get_operations(self):
        time_key = "time"
        time_value = 300
        time_unit = Unit("milliseconds")

        # ------------------------------------------------------------------- #
        #   Step 1 – set the entry
        # ------------------------------------------------------------------- #
        with allure.step("Store a time reading (value + unit)"):
            self.container.set(time_key, time_value, time_unit)

        # Attach the internal dict after the set call
        allure.attach(
            body=str(self.container.kv_pairs),
            name="kv_pairs after set",
            attachment_type=allure.attachment_type.JSON,
        )

        # ------------------------------------------------------------------- #
        #   Step 2 – retrieve the whole tuple
        # ------------------------------------------------------------------- #
        with allure.step("Retrieve the full (value, unit) tuple via get()"):
            retrieved_tuple = self.container.get(time_key)

        assert retrieved_tuple == (time_value, time_unit), "get() must return the stored tuple"

        # ------------------------------------------------------------------- #
        #   Step 3 – retrieve value only
        # ------------------------------------------------------------------- #
        with allure.step("Retrieve the value only via get_value()"):
            retrieved_value = self.container.get_value(time_key)

        assert retrieved_value == time_value, "get_value() must return the stored value"

        # ------------------------------------------------------------------- #
        #   Step 4 – retrieve unit only
        # ------------------------------------------------------------------- #
        with allure.step("Retrieve the unit only via get_unit()"):
            retrieved_unit = self.container.get_unit(time_key)

        assert retrieved_unit == time_unit, "get_unit() must return the stored unit"

        # ------------------------------------------------------------------- #
        #   Step 5 – verify missing key handling
        # ------------------------------------------------------------------- #
        with allure.step("Calling get() on a missing key should raise KeyError"):
            with pytest.raises(KeyError) as excinfo:
                self.container.get("nonexistent")
            # Attach the exception message for the report
            allure.attach(
                body=str(excinfo.value),
                name="KeyError message",
                attachment_type=allure.attachment_type.TEXT,
            )

    @allure.story(ALLURE_STORY_MANIPULATION)
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("__setitem__ proxy forwards to .set and validates the payload")
    @allure.description(
        """
        ``container[key] = (value, unit)`` must delegate to :meth:`set` and raise
        ``TypeError`` when the RHS is not a two‑tuple.  The test also covers the
        ``__getitem__`` shortcut.
        """
    )
    def test_bracket_assignment_and_retrieval(self):
        dummy_key = "voltage"
        dummy_value = 101325
        dummy_unit = Unit("microvolts")

        # --------------------------------------------------------------- #
        #   Assignment using the mapping syntax
        # --------------------------------------------------------------- #
        with allure.step("Assign via ``container[key] = (value, unit)``"):
            self.container[dummy_key] = (dummy_value, dummy_unit)

        # Verify internal state
        allure.attach(
            body=str(self.container.kv_pairs),
            name="kv_pairs after bracket assignment",
            attachment_type=allure.attachment_type.JSON,
        )
        assert self.container.kv_pairs[dummy_key] == (dummy_value, dummy_unit)

        # --------------------------------------------------------------- #
        #   Retrieval via the mapping protocol
        # --------------------------------------------------------------- #
        with allure.step("Retrieve the same entry via ``container[key]``"):
            retrieved = self.container[dummy_key]

        assert retrieved == (dummy_value, dummy_unit)

        # --------------------------------------------------------------- #
        #   Wrong payload shape – expect TypeError
        # --------------------------------------------------------------- #
        with allure.step("Assign a malformed payload and expect TypeError"):
            with pytest.raises(TypeError) as excinfo:
                self.container["bad"] = (123,)          # too short
            allure.attach(
                body=str(excinfo.value),
                name="TypeError message",
                attachment_type=allure.attachment_type.TEXT,
            )

    @allure.story(ALLURE_STORY_MANIPULATION)
    @allure.severity(allure.severity_level.MINOR)
    @allure.title("pop() returns and removes a key, supports default fallback")
    @allure.description(
        """
        Confirm that ``pop`` removes the entry and returns the stored tuple,
        and that the optional ``default`` argument behaves like ``dict.pop``.
        """
    )
    def test_pop_behavior(self):
        key = "voltage"
        value = 0.45
        unit = Unit("microvolts")
        self.container.set(key, value, unit)

        # ------------------------------------------------------------------- #
        #   Pop an existing key
        # ------------------------------------------------------------------- #
        with allure.step("Pop an existing key and verify the returned tuple"):
            popped = self.container.pop(key)
            assert popped == (value, unit)
            assert key not in self.container

        # ------------------------------------------------------------------- #
        #   Pop with default – key absent
        # ------------------------------------------------------------------- #
        with allure.step("Pop a missing key with a default value"):
            default_fallback = ("default", None)
            popped_default = self.container.pop("missing", default=default_fallback)
            assert popped_default == default_fallback

        # ------------------------------------------------------------------- #
        #   Pop without default on missing key – should raise
        # ------------------------------------------------------------------- #
        with allure.step("Pop a missing key without default must raise KeyError"):
            with pytest.raises(KeyError):
                self.container.pop("missing_again")

    @allure.story(ALLURE_STORY_MANIPULATION)
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("keys() and iter_keys() return alphabetically sorted key lists")
    @allure.description(
        """
        The container should return its keys sorted alphabetically.
        ``keys(case_insensitive=True)`` must perform a case‑insensitive sort.
        The generator version ``iter_keys`` must emit the same ordering.
        """
    )
    def test_key_enumeration(self):
        # Insert keys in a random order
        entries = {
            "b": (1, None),
            "A": (2, None),
            "c": (3, None),
            "a": (4, None),
        }
        for k, v in entries.items():
            self.container.set(k, *v)

        with allure.step("Check alphabetical sorting (case‑sensitive)"):
            sorted_keys = self.container.keys()
            assert sorted_keys == ["A", "a", "b", "c"]

        with allure.step("Check case‑insensitive sorting"):
            sorted_keys_ci = self.container.keys(case_insensitive=True)
            assert sorted_keys_ci == ["A", "a", "b", "c"]  # same order because lowercasing yields A/a/b/c

        with allure.step("Verify that iter_keys() yields the same order as keys()"):
            iter_keys_list = list(self.container.iter_keys())
            assert iter_keys_list == sorted_keys

    # ------------------------------------------------------------------- #
    #   2️⃣  Serialisation tests (mocked HDF5 interactions)
    # ------------------------------------------------------------------- #

    @allure.story(ALLURE_STORY_SERIALISATION)
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("to_hdf5() delegates to write_compound_attr for every entry")
    @allure.description(
        """
        The container must call ``write_compound_attr`` once per stored key.
        The test patches the helper and checks the call arguments.
        """
    )
    def test_to_hdf5_calls_write_helper(self):
        # Populate the container with a couple of entries
        data = {
            "resistance": (300, Unit("nanosiemens")),
            "voltage": (101325, Unit("microvolts")),
        }
        for k, (v, u) in data.items():
            self.container.set(k, v, u)

        # Create a dummy ``h5py.Group`` – we only need an object with a name attribute
        dummy_group = mock.Mock(name="h5py.Group")
        dummy_group.name = "/dummy_group"

        with mock.patch("converters.schema.compound_kv_container.write_compound_attr") as mock_write:
            with allure.step("Invoke to_hdf5() on the container"):
                self.container.to_hdf5(dummy_group)

            # The helper must have been called exactly twice (once per key)
            assert mock_write.call_count == len(data)

            # Verify each call arguments (order is not guaranteed, so we check set)
            called_keys = {call_args[0][1] for call_args in mock_write.call_args_list}
            assert called_keys == set(data.keys())

            # Detailed attachment of the call list for the report
            call_summary = "\n".join(
                f"{index+1}. key={item.args[1]!r}, value={item.args[2]!r}, unit={item.args[3]!r}"
                for index, item in enumerate(mock_write.call_args_list)
            )
            allure.attach(
                body=call_summary,
                name="write_compound_attr call summary",
                attachment_type=allure.attachment_type.TEXT,
            )

    @allure.story(ALLURE_STORY_SERIALISATION)
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("from_hdf5() reconstructs a container from compound attributes")
    @allure.description(
        """
        The class‑method must iterate over ``grp.attrs`` and call
        ``read_compound_attr_parsed`` for each attribute name.
        The test patches the read‑helper to return deterministic values and
        asserts that the resulting container faithfully reproduces the mapping.
        """
    )
    def test_from_hdf5_rebuilds_container(self):
        # Fake HDF5 attribute names
        fake_attrs = ["voltage", "current"]
        # Prepare a stub ``Group`` object whose ``attrs`` behaves like a dict‑like
        dummy_group = mock.Mock(name="h5py.Group")
        dummy_group.attrs = mock.MagicMock()
        dummy_group.attrs.__iter__.return_value = iter(fake_attrs)

        # Mapping from attribute name to the (value, unit) that the helper should yield
        attr_to_payload: Dict[str, Tuple[Any, Optional[Unit]]] = {
            "voltage": (300, Unit("microvolts")),
            "current": (101325, Unit("microamperes")),
        }

        def fake_read(group: Any, name: str) -> Tuple[Any, Optional[Unit]]:
            """Side‑effect function for the read helper mock."""
            assert group is dummy_group
            return attr_to_payload[name]

        with mock.patch(
            "converters.schema.compound_kv_container.read_compound_attr_parsed",
            side_effect=fake_read,
        ) as mock_read:
            with allure.step("Call the class‑method from_hdf5()"):
                rebuilt = KeyValueCompoundContainer.from_hdf5(dummy_group)

            # Validate that read_compound_attr_parsed was called once per attribute name
            assert mock_read.call_count == len(fake_attrs)
            called_names = {call.args[1] for call in mock_read.call_args_list}
            assert called_names == set(fake_attrs)

            # Verify the resulting container holds the expected data
            assert rebuilt.kv_pairs == attr_to_payload

            # Attach the final mapping for the Allure report
            allure.attach(
                body=str(rebuilt.kv_pairs),
                name="Reconstructed kv_pairs",
                attachment_type=allure.attachment_type.JSON,
            )