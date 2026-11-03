# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any, Dict, List
from unittest import mock

import allure
import numpy as np
import pytest

from converters.schema.stimuli_group import StimuliGroup
# -------------------------------------------------------------------------
#  Imports from the production code (adjust the import path to your layout)
# -------------------------------------------------------------------------
from converters.schema.stimulus import Stimulus
from converters.utils.units import Unit


# -------------------------------------------------------------------------
#  Helper fixtures ---------------------------------------------------------
# -------------------------------------------------------------------------
@pytest.fixture
def mock_h5_group():
    """
    Returns a fresh mock that mimics the small subset of the h5py.Group API we
    need for the tests (create_group, __getitem__, keys, attrs, create_dataset).
    """
    grp = mock.MagicMock(name="h5py.Group")
    # .attrs behaves like a dict – we give it a real dict so we can inspect writes.
    grp.attrs = {}
    # keys() returns an iterator over sub‑group names – we will set it per test.
    grp.keys.return_value = []
    # __getitem__ returns a sub‑group mock – we will configure it later.
    grp.__getitem__.side_effect = lambda name: mock.MagicMock(name=f"subgroup:{name}")
    return grp


# -------------------------------------------------------------------------
#  Test class -------------------------------------------------------------
# -------------------------------------------------------------------------
@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for StimuliGroup")
@allure.description("Unit tests for the `StimuliGroup` class, mocking all I/O and verifying the public API.")
@allure.feature("StimuliGroup – HDF5 (de)serialization")
@allure.story("StimuliGroup – round‑trip to/from HDF5")
class TestStimuliGroupSerialization:
    """
    Test suite that validates the *behaviour* of :class:`StimuliGroup`
    while **mocking** every downstream call (Stimulus, attribute helpers,
    HDF5 objects).  The suite demonstrates how to combine **pytest**, **unittest.mock**
    and **Allure** to obtain a highly‑documented, reproducible test report.
    """

    # ---------------------------------------------------------------------
    #  Optional setup / teardown (allure steps are added here as well)
    # ---------------------------------------------------------------------
    @allure.step("Prepare fresh test environment")
    def setup_method(self, method):
        """Executed before every test – useful for logging / extra reporting."""
        self.patches: List[mock._patch] = []

    @allure.step("Release all mocks")
    def teardown_method(self, method):
        """Stop every patch that has been started in a test."""
        for p in self.patches:
            p.stop()
        self.patches.clear()

    # ---------------------------------------------------------------------
    #  Helper to register a patch and keep it for teardown
    # ---------------------------------------------------------------------
    def _start_patch(self, target: str, new: Any) -> mock._patch:
        """Start a ``mock.patch`` and remember it for later teardown."""
        patcher = mock.patch(target, new=new)
        started = patcher.start()
        self.patches.append(patcher)
        return started

    # ---------------------------------------------------------------------
    #  TEST 1 – to_hdf5
    # ---------------------------------------------------------------------
    @allure.title("StimuliGroup.to_hdf5 writes stimuli, attributes and population list")
    @allure.description(
        """
        Verify that ``StimuliGroup.to_hdf5``:

        1. Creates a sub‑group for every stimulus name.
        2. Calls ``Stimulus.to_hdf5`` on each stimulus.
        3. Writes **all** scalar meta‑data via ``write_compound_attr``.
        4. Stores a population list (when present) as a ``PopulationList`` dataset.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    def test_to_hdf5_writes_all_components(self, mock_h5_group):
        # -------------------------------------------------------------
        #  Arrange – build a StimuliGroup instance with two entries
        # -------------------------------------------------------------
        dummy_stim_a = mock.create_autospec(Stimulus, instance=True, spec_set=True)
        dummy_stim_b = mock.create_autospec(Stimulus, instance=True, spec_set=True)

        # Meta‑data (including a unit) for each sub‑group
        attr_map = {
            "stim_a": {"frequency": (440, Unit.HERTZ), "label": ("tone", None)},
            "stim_b": {"amplitude": (0.8, None)},
        }

        # Population lists – only stim_b has one
        pop_list_map = {"stim_b": ["pop1", "pop2"]}

        group = StimuliGroup(
            stimuli_map={"stim_a": dummy_stim_a, "stim_b": dummy_stim_b},
            attr_map=attr_map,
            pop_list_map=pop_list_map,
        )

        # -------------------------------------------------------------
        #  Mock the low‑level helpers
        # -------------------------------------------------------------
        mocked_write_attr = self._start_patch(
            "converters.schema.stimuli_group.write_compound_attr", mock.Mock()
        )
        # We'll also need to mock the HDF5 create_group / create_dataset calls.
        # ``parent.create_group`` returns a new mock sub‑group for each name.
        sub_group_mocks: Dict[str, mock.MagicMock] = {}

        def _create_group(name: str):
            sub = mock.MagicMock(name=f"subgroup:{name}")
            # each sub‑group needs its own attrs dict and a create_dataset mock
            sub.attrs = {}
            sub.create_dataset = mock.Mock(name=f"{name}.create_dataset")
            sub_group_mocks[name] = sub
            return sub

        mock_h5_group.create_group.side_effect = _create_group

        # -------------------------------------------------------------
        #  Act – invoke the method under test
        # -------------------------------------------------------------
        with allure.step("Serialise StimuliGroup into the mocked HDF5 group"):
            group.to_hdf5(mock_h5_group)

        # -------------------------------------------------------------
        #  Assert – verify every inner call
        # -------------------------------------------------------------
        # 1️⃣ Stimulus objects were asked to serialise themselves
        dummy_stim_a.to_hdf5.assert_called_once_with(sub_group_mocks["stim_a"])
        dummy_stim_b.to_hdf5.assert_called_once_with(sub_group_mocks["stim_b"])

        # 2️⃣ All compound attributes were written (order does not matter)
        expected_calls = [
            mock.call(sub_group_mocks["stim_a"], "frequency", 440, Unit.HERTZ),
            mock.call(sub_group_mocks["stim_a"], "label", "tone", None),
            mock.call(sub_group_mocks["stim_b"], "amplitude", 0.8, None),
        ]
        assert mocked_write_attr.call_args_list == expected_calls

        # 3️⃣ PopulationList dataset created only for the group that has one
        sub_group_mocks["stim_a"].create_dataset.assert_not_called()
        sub_group_mocks["stim_b"].create_dataset.assert_called_once_with(
            "PopulationList",
            data=["pop1", "pop2"],
            dtype=mock.ANY,  # h5py.string_dtype()
        )

        # 4️⃣ Attach useful artefacts to the Allure report
        with allure.step("Attach a snapshot of the call tree"):
            call_tree = {
                "stimuli_written": list(dummy.__class__.__name__ for dummy in [dummy_stim_a, dummy_stim_b]),
                "attributes_written": mocked_write_attr.call_args_list,
                "population_datasets": {
                    name: mock._call_args[1] if mock.called else None
                    for name, mock in sub_group_mocks.items()
                },
            }
            allure.attach(
                str(call_tree),
                name="to_hdf5 call overview",
                attachment_type=allure.attachment_type.TEXT,
            )

    # ---------------------------------------------------------------------
    #  TEST 2 – from_hdf5
    # ---------------------------------------------------------------------
    @allure.title("StimuliGroup.from_hdf5 reconstructs stimuli, attributes and population list")
    @allure.description(
        """
        Verify that ``StimuliGroup.from_hdf5``:

        1. Iterates over every sub‑group present in the parent HDF5 group.
        2. Calls ``Stimulus.from_hdf5`` for each sub‑group.
        3. Reads **all** compound attributes via ``read_compound_attr`` and stores the values
           (units are ignored on deserialisation).
        4. Retrieves a population list when the ``PopulationList`` dataset exists.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    def test_from_hdf5_reads_all_components(self, mock_h5_group):
        # -------------------------------------------------------------
        #  Arrange – mock the HDF5 layout the method will read from
        # -------------------------------------------------------------
        # Create two sub‑group mocks that mimic h5py.Group behaviour.
        sub_a = mock.MagicMock(name="subgroup:stim_a")
        sub_b = mock.MagicMock(name="subgroup:stim_b")

        # Populate the .attrs dict with raw attribute names that the method will iterate over.
        sub_a.attrs = {"frequency": mock.sentinel.attr_meta}
        sub_b.attrs = {"amplitude": mock.sentinel.attr_meta, "label": mock.sentinel.attr_meta}

        # Sub‑group B contains a PopulationList dataset.
        pop_dataset = mock.MagicMock(name="PopulationList")
        pop_dataset.__getitem__.return_value = np.array(["popX", "popY"], dtype=object)
        # Using a simple list conversion that matches the implementation
        pop_dataset[:] = np.array(["popX", "popY"], dtype=object)
        sub_b.__contains__.side_effect = lambda key: key == "PopulationList"
        sub_b.__getitem__.side_effect = lambda key: pop_dataset if key == "PopulationList" else mock.DEFAULT

        # Mock the parent group's ``keys`` iterator and ``__getitem__``.
        mock_h5_group.keys.return_value = ["stim_a", "stim_b"]
        mock_h5_group.__getitem__.side_effect = lambda name: sub_a if name == "stim_a" else sub_b

        # -------------------------------------------------------------
        #  Mock the low‑level helpers and Stimulus class‑method
        # -------------------------------------------------------------
        mocked_read_attr = self._start_patch(
            "converters.schema.stimuli_group.read_compound_attr_parsed", mock.Mock()
        )
        # Configure read_compound_attr to return (value, unit_placeholder) tuples.
        mocked_read_attr.side_effect = [
            (440, Unit.HERTZ),  # frequency for stim_a
            (0.8, None),  # amplitude for stim_b
            ("tone", None),  # label for stim_b
        ]

        mocked_stimulus_from_hdf5 = self._start_patch(
            "converters.schema.stimulus.Stimulus.from_hdf5", mock.Mock()
        )
        # Return distinct mock Stimulus objects for each call.
        dummy_stim_a = mock.create_autospec(Stimulus, instance=True, spec_set=True)
        dummy_stim_b = mock.create_autospec(Stimulus, instance=True, spec_set=True)
        mocked_stimulus_from_hdf5.side_effect = [dummy_stim_a, dummy_stim_b]

        # -------------------------------------------------------------
        #  Act – invoke the class‑method under test
        # -------------------------------------------------------------
        with allure.step("Reconstruct StimuliGroup from the mocked HDF5 hierarchy"):
            reconstructed = StimuliGroup.from_hdf5(mock_h5_group)

        # -------------------------------------------------------------
        #  Assert – verify the internal calls and final object state
        # -------------------------------------------------------------
        # 1️⃣ Stimulus.from_hdf5 called with the proper sub‑groups
        mocked_stimulus_from_hdf5.assert_has_calls(
            [mock.call(sub_a), mock.call(sub_b)], any_order=False
        )

        # 2️⃣ read_compound_attr called once for every attribute key
        assert mocked_read_attr.call_count == 3
        mocked_read_attr.assert_has_calls(
            [
                mock.call(sub_a, "frequency"),
                mock.call(sub_b, "amplitude"),
                mock.call(sub_b, "label"),
            ],
            any_order=False,
        )

        # 3️⃣ The returned StimuliGroup has the expected internal dictionaries
        assert reconstructed.stimuli_map == {"stim_a": dummy_stim_a, "stim_b": dummy_stim_b}
        assert reconstructed.attr_map == {
            "stim_a": {"frequency": 440},
            "stim_b": {"amplitude": 0.8, "label": "tone"},
        }
        # Population list should be present only for stim_b
        assert reconstructed.pop_list_map == {"stim_b": ["popX", "popY"]}

        # 4️⃣ Attach a summary of the reconstruction to the Allure report
        with allure.step("Attach verification snapshot"):
            summary = {
                "stimuli_keys": list(reconstructed.stimuli_map.keys()),
                "attributes": reconstructed.attr_map,
                "populations": reconstructed.pop_list_map,
            }
            allure.attach(
                str(summary),
                name="from_hdf5 reconstruction snapshot",
                attachment_type=allure.attachment_type.JSON,
            )
