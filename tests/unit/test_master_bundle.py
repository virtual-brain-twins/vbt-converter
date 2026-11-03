# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
"""
pytest‑style test‑module for ``converters.schema.master_bundle.MasterBundle``.

All heavy I/O is mocked, every inner call is asserted and the whole flow is
instrumented with Allure annotations to obtain a rich, step‑by‑step test
report.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import allure

# ----------------------------------------------------------------------
#  Imports of the objects that will be mocked – only the classes are needed
# ----------------------------------------------------------------------
from converters.schema.protocol_group import ProtocolGroup
from converters.schema.raw_bundle_group import RawBundleGroup
from converters.schema.standard_format_group import StandardFormatGroup


# ----------------------------------------------------------------------
#  Test‑suite for `converters.schema.master_bundle.MasterBundle`
# ----------------------------------------------------------------------

@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for MasterBundle")
@allure.description("Unit tests for the `MasterBundle` class, mocking all I/O and verifying the public API.")
@allure.feature("MasterBundle")
@allure.story("Serialisation / Deserialisation of a complete bundle")
class TestMasterBundle:
    """
    Validates the public behaviour of :class:`MasterBundle` while **mocking all
    heavy I/O calls** (`h5py.File` and the three group objects).  The tests
    demonstrate:

    * creation of the output directory,
    * delegation of serialisation to the injected groups,
    * reconstruction from an HDF5 file,
    * Allure‑driven step tracking, attachments and severity tagging.
    """

    # ------------------------------------------------------------------
    #  Setup / Teardown
    # ------------------------------------------------------------------
    def setup_method(self) -> None:
        """
        Build fresh ``MagicMock`` instances for the three sub‑groups and a
        temporary directory for the HDF5 file path.  The objects are stored on
        ``self`` and therefore reused by every test method.
        """
        # ---- Mocks for the three group objects ---------------------------------
        self.standard_group_mock = MagicMock(name="StandardFormatGroup")
        self.raw_group_mock = MagicMock(name="RawBundleGroup")
        self.protocol_group_mock = MagicMock(name="ProtocolGroup")

        # ---- Temporary directory (auto‑cleaned in teardown) --------------------
        self._tmp_dir = tempfile.TemporaryDirectory()
        # The bundle will be written under a nested ``output`` folder to
        # verify that the code creates missing parents.
        self.output_path = Path(self._tmp_dir.name) / "output" / "bundle.h5"

    def teardown_method(self) -> None:
        """Dispose the temporary directory created in ``setup_method``."""
        self._tmp_dir.cleanup()

    # ------------------------------------------------------------------
    #  Helper – Allure attachment utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _attach_json(label: str, payload: dict) -> None:
        """Attach a JSON‑serialisable payload to the Allure report."""
        allure.attach(
            json.dumps(payload, indent=2),
            name=label,
            attachment_type=allure.attachment_type.JSON,
        )

    @staticmethod
    def _attach_text(label: str, text: str) -> None:
        """Attach a plain‑text string (e.g. a file‑path) to the Allure report."""
        allure.attach(
            text,
            name=label,
            attachment_type=allure.attachment_type.TEXT,
        )

    # ------------------------------------------------------------------
    #  Test 1 – to_hdf5() – all optional groups present
    # ------------------------------------------------------------------
    @allure.title("MasterBundle.to_hdf5() writes all groups to an HDF5 file")
    @allure.description(
        """
        Verify that ``MasterBundle.to_hdf5``:

        1. Creates the parent directory of the supplied output path.
        2. Opens an ``h5py.File`` in write mode.
        3. Writes the ``format_version`` attribute.
        4. Delegates serialisation to *standard*, *raw* and *protocol* groups.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    def test_to_hdf5_writes_all_groups(self) -> None:
        """Full‑bundle serialisation – each inner ``to_hdf5`` must be called exactly once."""
        # ------------------------------------------------------------------
        #  1️⃣ Patch external dependency ``h5py.File``
        # ------------------------------------------------------------------
        with allure.step("Patch ``h5py.File`` to return a mock file handler"):
            mock_h5_file = MagicMock(name="h5py.File (write)")
            # The mock must behave as a context‑manager: ``with h5py.File(...) as f:``
            mock_h5_file.__enter__.return_value = mock_h5_file
            mock_h5_file.__exit__.return_value = None

            h5py_patch = patch("h5py.File", return_value=mock_h5_file)
            h5py_open_mock = h5py_patch.start()

        # ------------------------------------------------------------------
        #  2️⃣ Build the ``MasterBundle`` instance with the mocked groups
        # ------------------------------------------------------------------
        with allure.step("Instantiate MasterBundle using the mocked sub‑groups"):
            from converters.schema.master_bundle import MasterBundle

            bundle = MasterBundle(
                standard=self.standard_group_mock,
                raw=self.raw_group_mock,
                protocol=self.protocol_group_mock,
            )
            self._attach_json(
                "Bundle configuration",
                {
                    "standard": "mocked",
                    "raw": "mocked",
                    "protocol": "mocked",
                },
            )

        # ------------------------------------------------------------------
        #  3️⃣ Execute the method under test
        # ------------------------------------------------------------------
        with allure.step(f"Call ``to_hdf5`` with output path ``{self.output_path}``"):
            bundle.to_hdf5(self.output_path)
            self._attach_text("Output path", str(self.output_path))

        # ------------------------------------------------------------------
        #  4️⃣ Assertions – verify every mocked call
        # ------------------------------------------------------------------
        with allure.step("✅ Verify that the parent directory was created"):
            assert self.output_path.parent.exists(), "Parent directory must exist after to_hdf5()"

        with allure.step("✅ Verify that ``h5py.File`` was opened in write mode"):
            h5py_open_mock.assert_called_once_with(self.output_path, "w")

        with allure.step("✅ Verify that the ``format_version`` attribute was set"):
            expected_calls = [
                call("format_version", "2.0"),
                call("format_version_major", 2),
                call("format_version_minor", 0),
            ]

            # Check order **and** count
            mock_h5_file.attrs.__setitem__.assert_has_calls(expected_calls, any_order=False)
            assert mock_h5_file.attrs.__setitem__.call_count == 3

        with allure.step("✅ Verify that each sub‑group received the HDF5 file handle"):
            self.standard_group_mock.to_hdf5.assert_called_once_with(mock_h5_file)
            self.raw_group_mock.to_hdf5.assert_called_once_with(mock_h5_file)
            self.protocol_group_mock.to_hdf5.assert_called_once_with(mock_h5_file)

        # ------------------------------------------------------------------
        #  5️⃣ Cleanup – stop the patch
        # ------------------------------------------------------------------
        h5py_patch.stop()

    # ------------------------------------------------------------------
    #  Test 2 – from_hdf5() – optional groups absent
    # ------------------------------------------------------------------
    @allure.title("MasterBundle.from_hdf5() reconstructs bundle without optional groups")
    @allure.description(
        """
        Verify that ``MasterBundle.from_hdf5``:

        1. Opens the supplied HDF5 file in read mode.
        2. Calls ``StandardFormatGroup.from_hdf5`` unconditionally.
        3. Calls ``RawBundleGroup.from_hdf5`` **only if** the ``RawBundle`` group
           exists in the file.
        4. Calls ``ProtocolGroup.from_hdf5`` **only if** the ``Protocol`` group
           exists in the file.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    def test_from_hdf5_without_optional_groups(self) -> None:
        """Deserialisation when the HDF5 file only contains the mandatory standard group."""
        # ------------------------------------------------------------------
        #  1️⃣ Mock ``h5py.File`` to represent a file that only holds the
        #      ``StandardFormatGroup`` group.
        # ------------------------------------------------------------------
        with allure.step("Create a mock HDF5 file that pretends only the standard group exists"):
            mock_h5_file = MagicMock(name="h5py.File (read)")

            # ``'StandardFormatGroup'`` present → True, everything else → False
            mock_h5_file.__contains__.side_effect = lambda name: name == "StandardFormatGroup"

            # Context‑manager protocol for ``with h5py.File(...) as f:``
            mock_h5_file.__enter__.return_value = mock_h5_file
            mock_h5_file.__exit__.return_value = None

            h5py_read_patch = patch("h5py.File", return_value=mock_h5_file)
            h5py_read_mock = h5py_read_patch.start()

        # ------------------------------------------------------------------
        #  2️⃣ Patch the class‑methods ``from_hdf5`` of the three group classes.
        # ------------------------------------------------------------------
        with allure.step("Patch ``from_hdf5`` classmethods of each sub‑group"):
            std_from_patch = patch.object(
                StandardFormatGroup, "from_hdf5", return_value=self.standard_group_mock
            )
            raw_from_patch = patch.object(
                RawBundleGroup, "from_hdf5", return_value=self.raw_group_mock
            )
            proto_from_patch = patch.object(
                ProtocolGroup, "from_hdf5", return_value=self.protocol_group_mock
            )
            std_from_mock = std_from_patch.start()
            raw_from_mock = raw_from_patch.start()
            proto_from_mock = proto_from_patch.start()

        # ------------------------------------------------------------------
        #  3️⃣ Execute the method under test
        # ------------------------------------------------------------------
        with allure.step(f"Call ``MasterBundle.from_hdf5`` with input path ``{self.output_path}``"):
            from converters.schema.master_bundle import MasterBundle

            result_bundle = MasterBundle.from_hdf5(self.output_path)
            self._attach_text("Input path", str(self.output_path))

        # ------------------------------------------------------------------
        #  4️⃣ Assertions – verify correct calls and that optional groups are skipped
        # ------------------------------------------------------------------
        with allure.step("✅ Verify that ``h5py.File`` was opened in read mode"):
            h5py_read_mock.assert_called_once_with(self.output_path, "r")

        with allure.step("✅ Verify ``StandardFormatGroup.from_hdf5`` was called"):
            std_from_mock.assert_called_once_with(mock_h5_file)

        with allure.step("✅ Verify ``RawBundleGroup.from_hdf5`` was NOT called"):
            raw_from_mock.assert_not_called()

        with allure.step("✅ Verify ``ProtocolGroup.from_hdf5`` was NOT called"):
            proto_from_mock.assert_not_called()

        with allure.step("✅ Validate the returned ``MasterBundle`` instance"):
            assert isinstance(result_bundle, MasterBundle), "Result must be a MasterBundle"
            assert result_bundle.standard is self.standard_group_mock
            assert result_bundle.raw is None, "Raw group should be ``None`` when missing"
            assert result_bundle.protocol is None, "Protocol group should be ``None`` when missing"

        # ------------------------------------------------------------------
        #  5️⃣ Cleanup – stop all patches
        # ------------------------------------------------------------------
        std_from_patch.stop()
        raw_from_patch.stop()
        proto_from_patch.stop()
        h5py_read_patch.stop()

    # ------------------------------------------------------------------
    #  Test 3 – from_hdf5() – all groups present (positive path)
    # ------------------------------------------------------------------
    @allure.title("MasterBundle.from_hdf5() reconstructs bundle when all groups exist")
    @allure.description(
        """
        Positive scenario where the HDF5 file contains the three groups
        ``StandardFormatGroup``, ``RawBundle`` and ``Protocol``.  The test checks
        that each group's ``from_hdf5`` classmethod is called exactly once and
        that the resulting ``MasterBundle`` wires the returned objects correctly.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    def test_from_hdf5_with_all_groups(self) -> None:
        """Deserialisation when the file contains every expected group."""
        # ------------------------------------------------------------------
        #  1️⃣ Mock ``h5py.File`` to claim that every group exists.
        # ------------------------------------------------------------------
        with allure.step("Create a mock HDF5 file that pretends all groups exist"):
            mock_h5_file = MagicMock(name="h5py.File (read)")

            # Every known group name returns True for ``in`` checks
            mock_h5_file.__contains__.side_effect = lambda name: name in {
                "StandardFormatGroup",
                "RawBundle",
                "Protocol",
            }

            mock_h5_file.__enter__.return_value = mock_h5_file
            mock_h5_file.__exit__.return_value = None

            h5py_read_patch = patch("h5py.File", return_value=mock_h5_file)
            h5py_read_mock = h5py_read_patch.start()

        # ------------------------------------------------------------------
        #  2️⃣ Patch the three ``from_hdf5`` classmethods to return the mocks
        # ------------------------------------------------------------------
        with allure.step("Patch ``from_hdf5`` classmethods to return the pre‑created mocks"):
            std_from_patch = patch.object(
                StandardFormatGroup, "from_hdf5", return_value=self.standard_group_mock
            )
            raw_from_patch = patch.object(
                RawBundleGroup, "from_hdf5", return_value=self.raw_group_mock
            )
            proto_from_patch = patch.object(
                ProtocolGroup, "from_hdf5", return_value=self.protocol_group_mock
            )
            std_from_mock = std_from_patch.start()
            raw_from_mock = raw_from_patch.start()
            proto_from_mock = proto_from_patch.start()

        # ------------------------------------------------------------------
        #  3️⃣ Execute the method under test
        # ------------------------------------------------------------------
        with allure.step(f"Call ``MasterBundle.from_hdf5`` with input path ``{self.output_path}``"):
            from converters.schema.master_bundle import MasterBundle

            bundle = MasterBundle.from_hdf5(self.output_path)
            self._attach_text("Input path", str(self.output_path))

        # ------------------------------------------------------------------
        #  4️⃣ Assertions – every group method must have been invoked once
        # ------------------------------------------------------------------
        with allure.step("✅ Verify ``h5py.File`` was opened in read mode"):
            h5py_read_mock.assert_called_once_with(self.output_path, "r")

        with allure.step("✅ Verify each group's ``from_hdf5`` was called exactly once"):
            std_from_mock.assert_called_once_with(mock_h5_file)
            raw_from_mock.assert_called_once_with(mock_h5_file)
            proto_from_mock.assert_called_once_with(mock_h5_file)

        with allure.step("✅ Validate the wiring of the returned MasterBundle"):
            assert isinstance(bundle, MasterBundle)
            assert bundle.standard is self.standard_group_mock
            assert bundle.raw is self.raw_group_mock
            assert bundle.protocol is self.protocol_group_mock

        # ------------------------------------------------------------------
        #  5️⃣ Cleanup – stop all patches
        # ------------------------------------------------------------------
        std_from_patch.stop()
        raw_from_patch.stop()
        proto_from_patch.stop()
        h5py_read_patch.stop()
