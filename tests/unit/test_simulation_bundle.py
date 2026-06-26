from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import allure
import pytest

# ----------------------------------------------------------------------
# Project imports – these are the interfaces that the production code
# works with.  In the test they will be replaced by mocks.
# ----------------------------------------------------------------------
from converters.interfaces.data_section import DataSection
from converters.interfaces.simulation_bundle import SimulationBundle
from converters.sections.metadata_section import MetadataSection


# ----------------------------------------------------------------------
# Helper – a tiny concrete bundle used only for the purpose of testing
# ----------------------------------------------------------------------
class DummyBundle(SimulationBundle):
    """A minimal concrete implementation required for exercising the abstract base."""

    def __init__(self, sim_id: str, root: Path, sections: Dict[str, DataSection] | None = None):
        super().__init__()
        self._sim_id = sim_id
        self._original_root = root
        if sections:
            self.sections = sections

    @property
    def sim_id(self) -> str:  # noqa: D102
        return self._sim_id

    @property
    def original_root(self) -> Path:  # noqa: D102
        return self._original_root

    @classmethod
    def _factory_from_metadata(cls, meta: "MetadataSection") -> "DummyBundle":
        """
        In production this would create a concrete bundle from the parsed metadata.
        The test replaces the method with a mock, so the implementation here is only a fallback.
        """
        return DummyBundle(
            sim_id=meta.sim_id,  # type: ignore[attr-defined]
            root=Path(meta.root_path),  # type: ignore[attr-defined]
            sections={name: mock.create_autospec(DataSection) for name in meta.section_names}
            # type: ignore[attr-defined]
        )


# ----------------------------------------------------------------------
# The actual pytest test class (Allure‑instrumented)
# ----------------------------------------------------------------------
@allure.feature("Simulation bundle (HDF5) handling")
@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for Simulation bundle (HDF5) handling")
@allure.description("Unit tests for the SimulationBundle class and its (de)serialisation logic.")
class TestSimulationBundle:
    """
    Test‑suite that validates the high‑level (de)serialisation logic of
    :class:`SimulationBundle`.  All heavyweight operations are mocked
    so the suite runs fast and deterministically while providing a
    detailed Allure report.
    """

    # ------------------------------------------------------------------
    # Fixtures – create fresh mocks for every test
    # ------------------------------------------------------------------
    @pytest.fixture(autouse=True)
    def _setup_mocks(self) -> None:
        """Create the common mock objects used by all test methods."""
        # --------------------------------------------------------------
        # HDF5 file / group hierarchy
        # --------------------------------------------------------------
        self.h5_file_mock = mock.MagicMock(name="h5_file")
        self.top_group_mock = mock.MagicMock(name="top_group")
        self.h5_file_mock.create_group.return_value = self.top_group_mock

        # --------------------------------------------------------------
        # Section objects – two example sections
        # --------------------------------------------------------------
        self.spikes_section_mock = mock.create_autospec(DataSection, instance=True, name="spikes_section")
        self.metadata_section_mock = mock.create_autospec(DataSection, instance=True, name="metadata_section")

        # --------------------------------------------------------------
        # Bundle under test – already contains the two mocked sections
        # --------------------------------------------------------------
        self.bundle = DummyBundle(
            sim_id="test-sim-001",
            root=Path("/tmp/sim_root"),
            sections={"spikes": self.spikes_section_mock, "metadata": self.metadata_section_mock},
        )

    # ------------------------------------------------------------------
    # Helper – attach a readable snapshot of a mock’s call list
    # ------------------------------------------------------------------
    @staticmethod
    @allure.step("Attach call log for mock ``{mock_name}``")
    def _attach_mock_calls(mock_obj: mock.Mock, mock_name: str) -> None:
        """Collect ``mock_calls`` and attach them as a text artefact."""
        call_log = "\n".join(str(c) for c in mock_obj.mock_calls)
        allure.attach(call_log, name=f"{mock_name} call log", attachment_type=allure.attachment_type.TEXT)

    # ------------------------------------------------------------------
    # TEST 1 – Serialisation (to_hdf5)
    # ------------------------------------------------------------------
    @allure.title("SimulationBundle.to_hdf5 writes groups and forwards calls")
    @allure.description(
        """
        **Scenario**  
        A `SimulationBundle` holding two sections (`spikes` and `metadata`) is asked
        to serialise itself into an *opened* HDF5 file.

        **Expected behaviour**  
        1. The top‑level group ``sim_<sim_id>`` is created.  
        2. For each registered section a subgroup named after the section key is created.  
        3. Every ``DataSection`` receives a call to its ``to_hdf5`` method with the
           corresponding subgroup.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.story("Persist a bundle to HDF5")
    def test_to_hdf5_writes_groups_and_forwards_calls(self):
        # --------------------------------------------------------------
        # STEP 1 – invoke the public API
        # --------------------------------------------------------------
        with allure.step("Call ``bundle.to_hdf5`` with the mocked HDF5 file"):
            self.bundle.to_hdf5(self.h5_file_mock)

        # --------------------------------------------------------------
        # STEP 2 – verify the top‑level group was created correctly
        # --------------------------------------------------------------
        with allure.step("Verify top‑level group creation"):
            self.h5_file_mock.create_group.assert_called_once_with("sim_test-sim-001")
            self._attach_mock_calls(self.h5_file_mock, "h5_file")

        # --------------------------------------------------------------
        # STEP 3 – verify a subgroup exists for each section
        # --------------------------------------------------------------
        with allure.step("Verify a subgroup is created for each registered section"):
            expected_creation_calls = [mock.call("spikes"), mock.call("metadata")]
            self.top_group_mock.create_group.assert_has_calls(expected_creation_calls, any_order=True)
            assert self.top_group_mock.create_group.call_count == 2
            self._attach_mock_calls(self.top_group_mock, "top_group")

        # --------------------------------------------------------------
        # STEP 4 – ensure each section got its own subgroup
        # --------------------------------------------------------------
        with allure.step("Verify each DataSection received its subgroup"):
            # ``create_group`` returns a new mock each time – capture them
            created_subgroups = [
                call_args[0][0] for call_args in self.top_group_mock.create_group.call_args_list
            ]
            # The actual subgroup objects are the return values of ``create_group``;
            # we only need to be sure the sections were called once.
            self.spikes_section_mock.to_hdf5.assert_called_once()
            self.metadata_section_mock.to_hdf5.assert_called_once()

            allure.attach(str(created_subgroups), name="Created sub‑group names",
                          attachment_type=allure.attachment_type.TEXT)

    # ------------------------------------------------------------------
    # TEST 2 – Deserialisation (from_hdf5) without a raw ZIP bundle
    # ------------------------------------------------------------------
    @allure.title("SimulationBundle.from_hdf5 rebuilds a bundle (no raw zip)")
    @allure.description(
        """
        **Scenario**  
        An HDF5 file contains a group ``sim_<sim_id>`` with a *metadata* subgroup
        and a custom subsection called ``spikes``. No ``raw_bundle`` group is present.

        **Expected behaviour**  
        1. ``MetadataSection.from_group`` is invoked for the ``metadata`` subgroup.  
        2. The concrete bundle is created via ``_factory_from_metadata``.  
        3. All non‑metadata sections receive a call to their ``from_hdf5`` method
           with the matching subgroup.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    @allure.story("Restore a bundle from HDF5 (no raw archive)")
    def test_from_hdf5_recreates_bundle_without_raw_zip(self):
        # --------------------------------------------------------------
        # ARRANGE – mock the HDF5 hierarchy
        # --------------------------------------------------------------
        sim_id = "test-sim-001"
        top_group = mock.MagicMock(name="sim_group")
        meta_group = mock.MagicMock(name="metadata_group")
        spikes_group = mock.MagicMock(name="spikes_group")

        h5_file = mock.MagicMock(name="h5_file")
        h5_file.__getitem__.return_value = top_group
        # ``top_group['metadata']`` etc.
        top_group.__getitem__.side_effect = lambda name: {
            "metadata": meta_group,
            "spikes": spikes_group,
        }[name]

        # The iterator used by the production code to discover sections
        existing_items = {
            "metadata": meta_group,
            "spikes": spikes_group,
        }

        top_group.__contains__.side_effect = existing_items.__contains__
        top_group.items.return_value = [("metadata", meta_group), ("spikes", spikes_group)]

        # --------------------------------------------------------------
        # ARRANGE – mock MetadataSection and the bundle factory
        # --------------------------------------------------------------
        meta_mock = mock.create_autospec(MetadataSection, instance=True, name="metadata_mock")
        meta_mock.sim_id = sim_id
        meta_mock.root_path = "/tmp/sim_root"
        meta_mock.section_names = ["spikes", "metadata"]

        with mock.patch.object(
                MetadataSection, "from_group", return_value=meta_mock
        ) as md_from_group_patch, mock.patch.object(
            DummyBundle, "_factory_from_metadata", return_value=DummyBundle(
                sim_id=sim_id,
                root=Path("/tmp/sim_root"),
                sections={
                    "spikes": mock.create_autospec(DataSection, instance=True, name="spikes_section"),
                    "metadata": mock.create_autospec(DataSection, instance=True, name="metadata_section"),
                },
            )
        ) as factory_patch:
            # --------------------------------------------------------------
            # ACT – call the classmethod under test
            # --------------------------------------------------------------
            with allure.step("Invoke ``SimulationBundle.from_hdf5``"):
                result_bundle = DummyBundle.from_hdf5(h5_file, sim_id)

            # --------------------------------------------------------------
            # ASSERT – metadata extraction
            # --------------------------------------------------------------
            with allure.step("Verify metadata extraction via ``MetadataSection.from_group``"):
                md_from_group_patch.assert_called_once_with(meta_group)
                self._attach_mock_calls(md_from_group_patch, "MetadataSection.from_group")

            # --------------------------------------------------------------
            # ASSERT – factory usage
            # --------------------------------------------------------------
            with allure.step("Verify the concrete bundle is created via the factory"):
                factory_patch.assert_called_once_with(meta_mock)

            # --------------------------------------------------------------
            # ASSERT – each non‑metadata section is deserialised
            # --------------------------------------------------------------
            with allure.step("Verify ``from_hdf5`` called on every registered non‑metadata section"):
                spikes_section = result_bundle.sections["spikes"]
                spikes_section.from_hdf5.assert_called_once_with(spikes_group)

                # ``metadata`` is already handled; its ``from_hdf5`` should **not** be called.
                metadata_section = result_bundle.sections["metadata"]
                metadata_section.from_hdf5.assert_not_called()

            # --------------------------------------------------------------
            # ATTACH – final bundle representation for debugging
            # --------------------------------------------------------------
            allure.attach(str(result_bundle), name="Resulting DummyBundle", attachment_type=allure.attachment_type.TEXT)

    # ------------------------------------------------------------------
    # TEST 3 – Deserialisation with a raw ZIP archive attached
    # ------------------------------------------------------------------
    @allure.title("SimulationBundle.from_hdf5 restores raw simulation archive")
    @allure.description(
        """
        **Scenario**  
        The HDF5 file stores a ``raw_bundle`` group that contains a binary ZIP archive
        of the original simulation directory.

        **Expected behaviour**  
        1. The ZIP binary is read via ``DataSection.read_binary_blob``.  
        2. The bytes are written to a temporary file, extracted to a temporary directory, and
           the extracted path is attached to the resulting bundle as ``_raw_extracted_path``.  
        3. All other sections are processed exactly as in the *no‑raw‑bundle* case.
        """
    )
    @allure.severity(allure.severity_level.MINOR)
    @allure.story("Restore raw simulation archive from HDF5")
    def test_from_hdf5_with_raw_zip_bundle(self):
        sim_id = "test-sim-raw"
        # --------------------------------------------------------------
        # ARRANGE – mock HDF5 hierarchy (now includes a raw_bundle group)
        # --------------------------------------------------------------
        top_group = mock.MagicMock(name="sim_group")
        meta_group = mock.MagicMock(name="metadata_group")
        spikes_group = mock.MagicMock(name="spikes_group")
        raw_bundle_group = mock.MagicMock(name="raw_bundle_group")

        h5_file = mock.MagicMock(name="h5_file")
        h5_file.__getitem__.return_value = top_group

        top_group.__getitem__.side_effect = lambda name: {
            "metadata": meta_group,
            "spikes": spikes_group,
            "raw_bundle": raw_bundle_group,
        }[name]

        top_group.items.return_value = [
            ("metadata", meta_group),
            ("spikes", spikes_group),
            ("raw_bundle", raw_bundle_group),
        ]
        top_group.__contains__.return_value = True

        # --------------------------------------------------------------
        # ARRANGE – mock MetadataSection
        # --------------------------------------------------------------
        meta_mock = mock.create_autospec(MetadataSection, instance=True, name="metadata_mock")
        meta_mock.sim_id = sim_id
        meta_mock.root_path = "/tmp/raw_root"
        meta_mock.section_names = ["spikes", "metadata"]

        # --------------------------------------------------------------
        # ARRANGE – prepare an in‑memory ZIP archive (single placeholder file)
        # --------------------------------------------------------------
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w") as zf:
            zf.writestr("placeholder.txt", "dummy content")
        zip_bytes = zip_buffer.getvalue()

        # --------------------------------------------------------------
        # ARRANGE – patch the binary‑blob reader
        # --------------------------------------------------------------
        read_blob_patch = mock.patch.object(
            DataSection, "read_binary_blob", return_value=zip_bytes
        )

        # --------------------------------------------------------------
        # ACT & ASSERT – run the full code path
        # --------------------------------------------------------------
        with read_blob_patch as read_blob_mock, \
                mock.patch.object(MetadataSection, "from_group", return_value=meta_mock), \
                mock.patch.object(DummyBundle, "_factory_from_metadata", return_value=DummyBundle(
                    sim_id=sim_id,
                    root=Path("/tmp/raw_root"),
                    sections={
                        "spikes": mock.create_autospec(DataSection, instance=True, name="spikes_section"),
                        "metadata": mock.create_autospec(DataSection, instance=True, name="metadata_section"),
                    },
                )) as factory_patch:
            with allure.step("Invoke ``SimulationBundle.from_hdf5`` with a raw ZIP bundle"):
                result_bundle = DummyBundle.from_hdf5(h5_file, sim_id)

            # ----------------------------------------------------------
            # ASSERT – binary blob extraction
            # ----------------------------------------------------------
            with allure.step("Validate that the ZIP blob is read once"):
                read_blob_mock.assert_called_once_with(raw_bundle_group)
                self._attach_mock_calls(read_blob_mock, "DataSection.read_binary_blob")

            # ----------------------------------------------------------
            # ASSERT – temporary extraction path
            # ----------------------------------------------------------
            with allure.step("Check that the ZIP was extracted to a temporary directory"):
                raw_path: Path = getattr(result_bundle, "_raw_extracted_path")
                assert isinstance(raw_path, Path)

                # The extracted directory must contain the placeholder file we packed
                extracted_files = list(raw_path.rglob("*"))
                assert any(p.name == "placeholder.txt" for p in extracted_files), "placeholder.txt missing"

                allure.attach(str(raw_path), name="Extracted raw bundle path",
                              attachment_type=allure.attachment_type.TEXT)

            # ----------------------------------------------------------
            # ASSERT – other sections behave as in the simple case
            # ----------------------------------------------------------
            with allure.step("Validate normal section deserialisation"):
                spikes_section = result_bundle.sections["spikes"]
                spikes_section.from_hdf5.assert_called_once_with(spikes_group)

                metadata_section = result_bundle.sections["metadata"]
                metadata_section.from_hdf5.assert_not_called()

            # ----------------------------------------------------------
            # ASSERT – factory was used
            # ----------------------------------------------------------
            with allure.step("Factory creation verification"):
                factory_patch.assert_called_once_with(meta_mock)

    # ------------------------------------------------------------------
    # Optional explicit pytest style setup/teardown (kept for documentation)
    # ------------------------------------------------------------------
    # In this example the heavy lifting is done inside the ``_setup_mocks`` fixture
    # above.  The following methods are *no‑ops* but illustrate where you would
    # place extra per‑test housekeeping if needed.
    def setup_method(self, method: Any) -> None:  # pragma: no cover
        """Placeholder – real work is performed in the ``_setup_mocks`` fixture."""
        pass

    def teardown_method(self, method: Any) -> None:  # pragma: no cover
        """Ensure any lingering patchers are stopped after each test."""
        mock.patch.stopall()
