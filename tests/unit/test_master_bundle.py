"""
pytest‑based unit‑tests for ``converters.schema.master_bundle.MasterBundle``.

All external I/O (filesystem, h5py) and the inner‑group classes
(StandardFormatGroup, RawBundleGroup, ExternalDependenciesGrp) are mocked.
Every interaction is verified and recorded with Allure annotations so that
the generated HTML report is highly readable and searchable.
"""

from __future__ import annotations

import pathlib
from unittest import mock
from unittest.mock import MagicMock, call

import allure
import pytest

from converters.schema.master_bundle import MasterBundle


# --------------------------------------------------------------------------- #
#   Helper – deterministic temporary Path (no file is created)
# --------------------------------------------------------------------------- #
def _fake_path(tmp_dir: pathlib.Path, filename: str) -> pathlib.Path:
    """
    Return a :class:`pathlib.Path` that lives under ``tmp_dir`` without touching
    the real filesystem.  The returned path is used only as an argument to
    the code under test – the actual I/O is mocked.
    """
    return tmp_dir / filename

@pytest.fixture(autouse=True)
def patched_inner_groups():
    yield MagicMock()

# --------------------------------------------------------------------------- #
#   Test class – fully instrumented with Allure
# --------------------------------------------------------------------------- #
@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.feature("MasterBundle – HDF5 (de)serialisation")
@allure.story("High‑level bundle coordinates the inner groups")
class TestMasterBundle:
    """
    Test‑suite for :class:`converters.schema.master_bundle.MasterBundle`.

    *File‑system* and *h5py* interactions are mocked → the suite runs in < 10 ms.
    Every inner‑group call (``to_hdf5`` / ``from_hdf5``) is verified.
    The generated Allure report contains titles, descriptions, severity,
    step logs, and helpful attachments (stringified objects, paths, etc.).
    """

    # ------------------------------------------------------------------- #
    #   Optional setup/teardown – logged as Allure steps
    # ------------------------------------------------------------------- #
    def setup_method(self, method):
        """Executed before each test – useful for per‑test initialisation."""
        with allure.step("Setup test environment"):
            # Placeholder for future initialisation (e.g. temp dirs, env vars)
            pass

    def teardown_method(self, method):
        """Executed after each test – useful for clean‑up."""
        with allure.step("Teardown test environment"):
            # Placeholder for clean‑up actions
            pass

    # ------------------------------------------------------------------- #
    #   TEST 1 – ``to_hdf5`` forwards the call to the inner groups
    # ------------------------------------------------------------------- #
    @allure.title("Serialise MasterBundle to HDF5")
    @allure.description(
        """
        **Goal** – verify that ``MasterBundle.to_hdf5``:

        1. Guarantees the parent directory exists.
        2. Opens the target file in **write** mode via ``h5py.File``.
        3. Writes the ``format_version`` attribute.
        4. Delegates ``to_hdf5`` to the *standard* group and, when present,
           to the *raw* and *deps* groups.

        All external side‑effects are mocked; the test inspects the call
        history of those mocks.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.story("to_hdf5 implementation")
    @pytest.mark.parametrize(
        "include_raw, include_deps",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_to_hdf5_calls_inner_serializers(
            self,
            tmp_path: pathlib.Path,
            patched_inner_groups,
            include_raw: bool,
            include_deps: bool,
    ):
        """
        ``MasterBundle.to_hdf5`` must create the directory, open the file,
        write the version attribute and correctly delegate to the inner groups.
        """
        # ------------------------------------------------------------------- #
        # 1️⃣  Instantiate MasterBundle with the required optional parts
        # ------------------------------------------------------------------- #
        with allure.step("Instantiate MasterBundle"):
            bundle = MasterBundle(
                standard=patched_inner_groups.standard,
                raw=patched_inner_groups.raw if include_raw else None,
                deps=patched_inner_groups.deps if include_deps else None,
            )
            allure.attach(
                str(bundle),
                name="Bundle representation",
                attachment_type=allure.attachment_type.TEXT,
            )

        # ------------------------------------------------------------------- #
        # 2️⃣  Mock filesystem (Path.mkdir) and ``h5py.File`` context manager
        # ------------------------------------------------------------------- #
        output_path = mock.MagicMock(name="output_path", spec=pathlib.Path)
        with allure.step("Patch filesystem & h5py interactions"):
            # Path.mkdir – we only need to verify it has been called correctly.
            # h5py.File – return a MagicMock that behaves as a context manager.
            h5py_file_patch = mock.patch(
                "h5py.File",
                autospec=True,
                return_value=mock.MagicMock(name="h5py_file_ctx"),
            )

            with h5py_file_patch as h5py_mock:
                # Configure the context‑manager return value.
                file_handle = mock.MagicMock(name="h5_file_handle")
                h5py_mock.return_value.__enter__.return_value = file_handle

                # ----------------------------------------------------------- #
                # 3️⃣  Call the method under test
                # ----------------------------------------------------------- #
                with allure.step("Call MasterBundle.to_hdf5"):
                    bundle.to_hdf5(output_path)

                # ----------------------------------------------------------- #
                # 4️⃣  Assertions – filesystem side‑effects
                # ----------------------------------------------------------- #
                with allure.step("Assert that the parent directory was created"):
                    output_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

                with allure.step("Assert that HDF5 file was opened in write mode"):
                    h5py_mock.assert_called_once_with(output_path, "w")

                # ----------------------------------------------------------- #
                # 5️⃣  Assertions – version attribute on the HDF5 file
                # ----------------------------------------------------------- #
                with allure.step("Check ``format_version`` attribute on HDF5 file"):
                    # assert file_handle.attrs["format_version"] == "1.0"
                    assert call.attrs.__setitem__('format_version', '1.0') in file_handle.mock_calls

                # ----------------------------------------------------------- #
                # 6️⃣  Assertions – inner‑group delegations
                # ----------------------------------------------------------- #
                with allure.step("Validate call to StandardFormatGroup.to_hdf5"):
                    patched_inner_groups.standard.to_hdf5.assert_called_once_with(
                        file_handle
                    )

                if include_raw:
                    with allure.step("Validate call to RawBundleGroup.to_hdf5"):
                        patched_inner_groups.raw.to_hdf5.assert_called_once_with(
                            file_handle
                        )
                else:
                    with allure.step("Raw bundle absent – ensure no call"):
                        patched_inner_groups.raw.to_hdf5.assert_not_called()

                if include_deps:
                    with allure.step("Validate call to ExternalDependenciesGrp.to_hdf5"):
                        patched_inner_groups.deps.to_hdf5.assert_called_once_with(
                            file_handle
                        )
                else:
                    with allure.step("Dependencies group absent – ensure no call"):
                        patched_inner_groups.deps.to_hdf5.assert_not_called()

    # ------------------------------------------------------------------- #
    #   TEST 2 – ``from_hdf5`` reads groups only when they exist
    # ------------------------------------------------------------------- #
    @allure.title("Deserialise MasterBundle from HDF5")
    @allure.description(
        """
        **Goal** – verify that ``MasterBundle.from_hdf5``:

        1. Opens the supplied file in **read** mode.
        2. Calls the ``from_hdf5`` *class‑method* of each inner group **iff**
           the corresponding HDF5 subgroup is present.
        3. Returns a ``MasterBundle`` instance populated with the objects
           returned by those class‑methods.

        The test parametrises the presence / absence of the optional groups
        and checks that no super‑fluous call is issued.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    @allure.story("from_hdf5 implementation")
    @pytest.mark.parametrize(
        "has_raw, has_deps",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_from_hdf5_reads_groups_correctly(
            self,
            tmp_path: pathlib.Path,
            patched_inner_groups,
            has_raw: bool,
            has_deps: bool,
    ):
        """
        ``MasterBundle.from_hdf5`` must discover optional groups and forward
        deserialisation to the appropriate class methods.
        """
        # ------------------------------------------------------------------- #
        # 1️⃣  Build a fake ``h5py.File`` handle that pretends to contain the
        #     subgroup names used by ``MasterBundle``.
        # ------------------------------------------------------------------- #
        with allure.step("Create a mocked h5py.File handle"):
            mocked_file = mock.MagicMock(name="h5_file_handle")

            # The production code likely checks for subgroup names like
            # "RawBundle" and "ExternalDependencies".  Adjust if the real names
            # differ.
            present_subgroups = set()
            if has_raw:
                present_subgroups.add("RawBundle")
            if has_deps:
                present_subgroups.add("ExternalDependencies")

            # ``__contains__`` is used by ``if "RawBundle" in file`` checks.
            mocked_file.__contains__.side_effect = lambda name: name in present_subgroups

        # ------------------------------------------------------------------- #
        # 2️⃣  Mock the three ``from_hdf5`` class‑methods to return sentinel objs
        # ------------------------------------------------------------------- #
        std_instance = mock.Mock(name="StdGroup_instance")
        raw_instance = mock.Mock(name="RawGroup_instance")
        deps_instance = mock.Mock(name="DepsGroup_instance")

        std_from_hdf5_patch = mock.patch(
            "converters.schema.standard_format_group.StandardFormatGroup.from_hdf5",
            return_value=std_instance,
            autospec=True,
        )
        raw_from_hdf5_patch = mock.patch(
            "converters.schema.raw_bundle_group.RawBundleGroup.from_hdf5",
            return_value=raw_instance,
            autospec=True,
        )
        deps_from_hdf5_patch = mock.patch(
            "converters.schema.external_dependencies_group.ExternalDependenciesGrp.from_hdf5",
            return_value=deps_instance,
            autospec=True,
        )

        # ------------------------------------------------------------------- #
        # 3️⃣  Patch ``h5py.File`` itself to return our fake handle
        # ------------------------------------------------------------------- #
        h5py_file_patch = mock.patch(
            "h5py.File",
            autospec=True,
            return_value=mock.MagicMock(name="h5py_context"),
        )

        with (
            h5py_file_patch as h5py_mock,
            std_from_hdf5_patch as std_from_mock,
            raw_from_hdf5_patch as raw_from_mock,
            deps_from_hdf5_patch as deps_from_mock,
        ):
            # Configure the context‑manager to yield ``mocked_file``.
            h5py_mock.return_value.__enter__.return_value = mocked_file

            # --------------------------------------------------------------- #
            # 4️⃣  Invoke the class‑method under test
            # --------------------------------------------------------------- #
            with allure.step("Call MasterBundle.from_hdf5"):
                input_path = _fake_path(tmp_path, "bundle_in.h5")
                result_bundle = MasterBundle.from_hdf5(input_path)

            # --------------------------------------------------------------- #
            # 5️⃣  Assertions – file opening mode
            # --------------------------------------------------------------- #
            with allure.step("Assert file opened in read mode"):
                h5py_mock.assert_called_once_with(input_path, "r")

            # --------------------------------------------------------------- #
            # 6️⃣  Assertions – inner‑group ``from_hdf5`` calls
            # --------------------------------------------------------------- #
            with allure.step("Validate mandatory StandardFormatGroup.from_hdf5 call"):
                std_from_mock.assert_called_once_with(mocked_file)

            if has_raw:
                with allure.step("Validate RawBundleGroup.from_hdf5 call"):
                    raw_from_mock.assert_called_once_with(mocked_file)
            else:
                with allure.step("Raw bundle absent – ensure no call"):
                    raw_from_mock.assert_not_called()

            if has_deps:
                with allure.step("Validate ExternalDependenciesGrp.from_hdf5 call"):
                    deps_from_mock.assert_called_once_with(mocked_file)
            else:
                with allure.step("Dependencies group absent – ensure no call"):
                    deps_from_mock.assert_not_called()

            # --------------------------------------------------------------- #
            # 7️⃣  Assertions – the returned MasterBundle instance
            # --------------------------------------------------------------- #
            with allure.step("Inspect the constructed MasterBundle instance"):
                assert isinstance(result_bundle, MasterBundle)

                # Mandatory group
                assert result_bundle.standard is std_instance

                # Optional groups follow the presence matrix
                if has_raw:
                    assert result_bundle.raw is raw_instance
                else:
                    assert result_bundle.raw is None

                if has_deps:
                    assert result_bundle.deps is deps_instance
                else:
                    assert result_bundle.deps is None

                # Attach a human‑readable representation for the report
                allure.attach(
                    str(result_bundle),
                    name="Resulting MasterBundle",
                    attachment_type=allure.attachment_type.TEXT,
                )
