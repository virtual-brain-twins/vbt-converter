from __future__ import annotations

import pathlib

import allure
import h5py
import numpy as np
import pytest

from converters.interfaces.data_section import DataSection


# ----------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def h5file(tmp_path: pathlib.Path) -> h5py.File:
    """Create a temporary HDF5 file that is automatically closed."""
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        yield f
    # file is closed when leaving the context manager


@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for DataSection helper methods")
@allure.description("Unit tests for the static helper methods of DataSection.")
@allure.feature("DataSection – Helper utilities")
class TestDataSection:
    """
    Test suite for the static helper methods of ``DataSection``.
    All tests are **stand‑alone** – they do not depend on one another
    and they clean up after themselves.
    """

    @allure.story("Store a NumPy array")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Store and retrieve a simple float array (compression on)")
    def test_store_numpy_array_compressed(self, h5file: h5py.File) -> None:
        grp = h5file.create_group("numpy")
        arr = np.arange(10000, dtype=np.float64).reshape((2, 5000))

        # Act
        DataSection._store_numpy_array(grp, "mydata", arr, compress=True)

        # Assert
        dset = grp["mydata"]
        assert isinstance(dset, h5py.Dataset)
        np.testing.assert_array_equal(dset[()], arr)
        # dtype attribute must be the *string* representation of the numpy dtype
        assert dset.attrs["dtype"] == str(arr.dtype)
        # Compression should be gzip (default chosen by ``make_dataset``)
        assert dset.compression == "gzip"

    @allure.story("Store a NumPy array")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Store and retrieve an integer array (compression off)")
    def test_store_numpy_array_uncompressed(self, h5file: h5py.File) -> None:
        grp = h5file.create_group("numpy")
        arr = np.array([1, 2, 3, 4], dtype=np.int32)

        DataSection._store_numpy_array(grp, "ints", arr, compress=False)

        dset = grp["ints"]
        assert isinstance(dset, h5py.Dataset)
        np.testing.assert_array_equal(dset[()], arr)
        assert dset.attrs["dtype"] == str(arr.dtype)
        # No compression when ``compress=False``
        assert dset.compression is None

    @allure.story("Store a NumPy array")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Edge case – store an empty array")
    def test_store_numpy_array_empty(self, h5file: h5py.File) -> None:
        grp = h5file.create_group("numpy")
        arr = np.array([], dtype=np.float32)

        DataSection._store_numpy_array(grp, "empty", arr)

        dset = grp["empty"]
        assert dset.shape == (0,)
        np.testing.assert_array_equal(dset[()], arr)

    @allure.story("Store a UTF‑8 text blob")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Store and read back a regular string")
    def test_store_text_blob(self, h5file: h5py.File) -> None:
        grp = h5file.create_group("text")
        text = "Hello, 🌍!" * 80

        DataSection._store_text_blob(grp, "greeting", text, compress=True)

        dset = grp["greeting"]
        assert isinstance(dset, h5py.Dataset)
        # h5py stores a 1‑element array for our implementation
        stored = dset[()][0].decode("utf-8") if isinstance(dset[()][0], bytes) else dset[()][0]
        assert stored == text
        # The helper always adds the ``is_vlen`` flag
        assert bool(dset.attrs["is_vlen"]) is True
        # Compression flag should be gzip (because compress=True)
        assert dset.compression == "gzip"

    @allure.story("Store a UTF‑8 text blob")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Edge case – store an empty string")
    def test_store_text_blob_empty(self, h5file: h5py.File) -> None:
        grp = h5file.create_group("text")
        DataSection._store_text_blob(grp, "empty", "", compress=False)

        dset = grp["empty"]
        assert dset.compression is None
        stored = dset[()][0]
        # ``stored`` can be a ``bytes`` object or a numpy scalar string – both
        # compare equal to the empty Python string.
        assert stored == "" or stored == b""

    @allure.story("Store a binary blob")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Store and read back a non‑empty bytes object")
    def test_store_and_read_binary_blob(self, h5file: h5py.File) -> None:
        grp = h5file.create_group("binary")
        payload = b"\x00\xFF\x10\x20ABC"

        DataSection._store_binary_blob(grp, "payload", payload, compress=False)

        dset = grp["payload"]
        assert isinstance(dset, h5py.Dataset)
        # Underlying dtype must be uint8
        assert dset.dtype == np.uint8
        assert dset[()].tolist() == list(payload)  # raw integer view

        # Use the public helper to go back to bytes
        read_back = DataSection.read_binary_blob(dset)
        assert isinstance(read_back, bytes)
        assert read_back == payload

    @allure.story("Store a binary blob")
    @allure.severity(allure.severity_level.MINOR)
    @allure.title("Edge case – empty bytes")
    def test_store_binary_blob_empty(self, h5file: h5py.File) -> None:
        grp = h5file.create_group("binary")
        empty = b""

        DataSection._store_binary_blob(grp, "nothing", empty, compress=True)

        dset = grp["nothing"]
        assert dset.shape == (0,)
        assert dset.dtype == np.uint8
        # read_binary_blob must return an empty ``bytes`` object
        assert DataSection.read_binary_blob(dset) == b""

    @allure.story("Reference an external HDF5 file (link)")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Create an external link when target is a real HDF5 file")
    def test_store_external_path_as_link(self, h5file: h5py.File, tmp_path: pathlib.Path) -> None:
        """``make_link=True`` and target is a genuine HDF5 file ⇒ ExternalLink."""
        # create a dummy external HDF5 file
        external_path = tmp_path / "external.h5"
        with h5py.File(external_path, "w") as ext:
            ext.create_dataset("foo", data=np.arange(3))

        grp = h5file.create_group("refs")
        DataSection.store_external_path(
            grp, "link_to_ext", external_path, make_link=True
        )

        # The group now contains an HDF5 ExternalLink object.
        link: h5py.ExternalLink = grp.get("link_to_ext", getlink=True)
        assert isinstance(link, h5py.ExternalLink)
        # The link points to the *exact* file we created.
        assert link.filename == str(external_path)
        assert link.path == "/"  # we link to the root of the external file.

        # Link‑specific attribute should be present on the *parent* group.
        assert grp["link_to_ext"].attrs["link_type"] == "external_hdf5"
        # Human‑readable copy of the target stored as a group attribute.
        assert grp.attrs["link_to_ext_target"] == str(external_path)

    @allure.story("Reference an external HDF5 file (plain path)")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Fallback to plain path when make_link=False")
    def test_store_external_path_plain_fallback_no_link(
            self, h5file: h5py.File, tmp_path: pathlib.Path
    ) -> None:
        """When ``make_link=False`` the function must store a tiny dataset."""
        txt_path = tmp_path / "some.txt"
        txt_path.write_text("dummy")

        grp = h5file.create_group("refs")
        DataSection.store_external_path(
            grp, "plain_ref", txt_path, make_link=False
        )

        # No link object → should be a regular dataset
        dset = grp["plain_ref"]
        assert isinstance(dset, h5py.Dataset)

        # Verify content
        stored_path = dset[()][0].decode("utf-8") if isinstance(dset[()][0], bytes) else dset[()][0]
        assert stored_path == str(txt_path)

        # Attribute “reference_type” should be present
        assert dset.attrs["reference_type"] == "plain_path"
        # Group also stores a copy of the path as an attribute
        assert grp.attrs["plain_ref_path"] == str(txt_path)

    @allure.story("Reference an external HDF5 file (plain path fallback)")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Fallback when make_link=True but target is NOT an HDF5 file")
    def test_store_external_path_fallback_non_hdf5(
            self, h5file: h5py.File, tmp_path: pathlib.Path
    ) -> None:
        # Create a plain text file (not an HDF5 file)
        non_h5 = tmp_path / "random.bin"
        non_h5.write_bytes(b"\x01\x02\x03")

        grp = h5file.create_group("refs")
        DataSection.store_external_path(
            grp, "fallback", non_h5, make_link=True
        )

        # Should end up as a tiny dataset, not a link
        dset = grp["fallback"]
        assert isinstance(dset, h5py.Dataset)
        stored_path = dset[()][0].decode("utf-8") if isinstance(dset[()][0], bytes) else dset[()][0]
        assert stored_path == str(non_h5)

        # Verify attributes
        assert dset.attrs["reference_type"] == "plain_path"
        assert grp.attrs["fallback_path"] == str(non_h5)

    @allure.story("Reference an external HDF5 file (plain path fallback)")
    @allure.severity(allure.severity_level.MINOR)
    @allure.title("Fallback when the supplied path does not exist")
    def test_store_external_path_nonexistent(self, h5file: h5py.File, tmp_path: pathlib.Path) -> None:
        bogus = tmp_path / "does_not_exist.h5"

        grp = h5file.create_group("refs")
        DataSection.store_external_path(
            grp, "missing", bogus, make_link=True
        )

        # Should be a regular dataset (fallback)
        dset = grp["missing"]
        assert isinstance(dset, h5py.Dataset)

        stored_path = dset[()][0].decode("utf-8") if isinstance(dset[()][0], bytes) else dset[()][0]
        assert stored_path == str(bogus)

        # Attribute guarantees the fallback path was stored
        assert dset.attrs["reference_type"] == "plain_path"
        assert grp.attrs["missing_path"] == str(bogus)

    # ------------------------------------------------------------------
    # Additional sanity checks – ensuring that the helpers do **not**
    # interfere with each other when used in the same file.
    # ------------------------------------------------------------------
    @allure.story("Mixed usage in one HDF5 file")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("All helpers can operate together without clobbering")
    def test_multiple_helpers_coexist(self, h5file: h5py.File, tmp_path: pathlib.Path) -> None:
        """A single root group contains datasets created by every helper."""
        root = h5file.create_group("mixed")
        # numpy
        DataSection._store_numpy_array(root, "arr", np.arange(5), compress=False)
        # text
        DataSection._store_text_blob(root, "msg", "Test ☃", compress=False)
        # binary
        DataSection._store_binary_blob(root, "bin", b"\xDE\xAD\xBE\xEF", compress=False)
        # external reference (plain)
        ref_path = tmp_path / "ref.txt"
        ref_path.write_text("ref")
        DataSection.store_external_path(root, "ref", ref_path, make_link=False)

        # Verify each entry independently
        # numpy
        np.testing.assert_array_equal(root["arr"][()], np.arange(5))
        # text
        txt = root["msg"][()][0]
        txt = txt.decode("utf-8") if isinstance(txt, bytes) else txt
        assert txt == "Test ☃"
        # binary
        assert DataSection.read_binary_blob(root["bin"]) == b"\xDE\xAD\xBE\xEF"
        # reference
        ref_txt = root["ref"][()][0]
        ref_txt = ref_txt.decode("utf-8") if isinstance(ref_txt, bytes) else ref_txt
        assert ref_txt == str(ref_path)

        attributes = {
            "arr": {'dtype': 'int64'},
            "msg": {'is_vlen': True},
            "bin": {},
            "ref": {'reference_type': 'plain_path'}
        }

        # Ensure that attributes from one helper did not leak into another
        for name in ("arr", "msg", "bin", "ref"):
            assert attributes[name] == dict(root[name].attrs), "Attribute mismatch"
