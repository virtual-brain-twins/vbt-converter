import json
from typing import Any, Dict, Tuple, Optional
from unittest import mock

import allure
import pytest

from converters.schema.compound_kv_container import KeyValueCompoundContainer
from converters.utils.units import Unit

# ----------------------------------------------------------------------
# Helper constants used across several tests
# ----------------------------------------------------------------------
_SAMPLE_KV: Dict[str, Tuple[Any, Optional[Unit]]] = {
    "frequency": (300, Unit.HERTZ),
    "time": (101325, Unit.MICROSECOND),
    "comment": ("ok", None),
}


@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for KeyValueCompoundContainer")
@allure.description("Unit tests for the `KeyValueCompoundContainer` class, focusing on public API behaviour.")
@allure.feature("Key‑Value Compound Container")
class TestKeyValueCompoundContainer:
    """
    Test‑suite for :class:`KeyValueCompoundContainer`.

    The suite focuses on **behavioural** tests (public API) while
    **mocking** the low‑level HDF5 helpers so the unit tests stay fast,
    deterministic and do not touch the filesystem.
    """

    # ------------------------------------------------------------------
    # Setup / teardown – start patches once per test method
    # ------------------------------------------------------------------
    def setup_method(self):
        """Patch the low‑level HDF5 helpers before each test."""
        self.patcher_write = mock.patch(
            "converters.schema.compound_kv_container.write_compound_attr"
        )
        self.mock_write = self.patcher_write.start()

        self.patcher_read = mock.patch(
            "converters.schema.compound_kv_container.read_compound_attr"
        )
        self.mock_read = self.patcher_read.start()

        # a very small fake h5py.Group‑like object (only .attrs is accessed)
        self.fake_group = mock.Mock(name="fake_h5py_group")
        self.fake_group.attrs = mock.MagicMock(name="attrs")

    def teardown_method(self):
        """Stop all patches."""
        mock.patch.stopall()

    # ------------------------------------------------------------------
    # 1️⃣  Basic set / get / item access
    # ------------------------------------------------------------------
    @allure.title("✅ Store and retrieve values with Unit objects")
    @allure.description(
        """
        Verify that ``set`` stores the `(value, unit)` tuple correctly and that
        the convenience getters (`get`, `get_value`, `get_unit`) return the
        expected parts.

        The test also checks the dictionary‑like ``__getitem__`` and
        ``__contains__`` helpers.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.story("Basic CRUD operations")
    def test_set_and_get_basic(self):
        container = KeyValueCompoundContainer()

        with allure.step("Insert several key‑value pairs using `set`"):
            for key, (val, unit) in _SAMPLE_KV.items():
                container.set(key, val, unit)

        with allure.step("Verify internal mapping matches the input data"):
            allure.attach(
                json.dumps(container.kv_pairs, default=str, indent=2),
                name="kv_pairs after set",
                attachment_type=allure.attachment_type.JSON,
            )
            assert container.kv_pairs == _SAMPLE_KV

        with allure.step("Read values back via the public getter methods"):
            for key, (expected_val, expected_unit) in _SAMPLE_KV.items():
                # full tuple
                assert container.get(key) == (expected_val, expected_unit)
                # value only
                assert container.get_value(key) == expected_val
                # unit only (may be None)
                assert container.get_unit(key) == expected_unit

        with allure.step("Check dictionary‑like shortcuts (`[]`, `in`)"):
            for key in _SAMPLE_KV:
                assert container[key] == _SAMPLE_KV[key]  # __getitem__
                assert key in container  # __contains__
            # non‑existent key raises KeyError
            with pytest.raises(KeyError):
                _ = container["does_not_exist"]

    # ------------------------------------------------------------------
    # 2️⃣  Validation of the `unit` argument
    # ------------------------------------------------------------------
    @allure.title("❌ Reject non‑Unit objects as unit argument")
    @allure.description(
        """
        The ``set`` method must raise ``TypeError`` when the supplied ``unit``
        is not an instance of :class:`~converters.utils.units.Unit` and also
        not ``None``.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    @allure.story("Input validation")
    def test_set_invalid_unit_raises(self):
        container = KeyValueCompoundContainer()
        with allure.step("Attempt to store a value with a wrong unit type"):
            with pytest.raises(TypeError) as excinfo:
                container.set("bad_unit", 42, unit="not_a_unit")
            assert "must be a `Unit` instance or None" in str(excinfo.value)

    # ------------------------------------------------------------------
    # 3️⃣  to_hdf5 – ensure write_compound_attr is called for every entry
    # ------------------------------------------------------------------
    @allure.title("💾 Serialise container → HDF5 (write_compound_attr called)")
    @allure.description(
        """
        ``to_hdf5`` must iterate over the internal mapping and delegate
        each entry to the helper ``write_compound_attr``.  The test patches
        the helper and asserts the exact calls and order.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.story("Serialization")
    def test_to_hdf5_calls_write_compound_attr(self):
        container = KeyValueCompoundContainer(kv_pairs=_SAMPLE_KV)

        with allure.step("Serialise the container into the mocked HDF5 group"):
            container.to_hdf5(self.fake_group)

        with allure.step("Assert that write_compound_attr was called once per key"):
            # Build the set of expected call arguments
            expected_calls = [
                mock.call(self.fake_group, key, value, unit)
                for key, (value, unit) in _SAMPLE_KV.items()
            ]
            self.mock_write.assert_has_calls(expected_calls, any_order=False)
            assert self.mock_write.call_count == len(_SAMPLE_KV)

        with allure.step("Attach the call list for the report"):
            call_list = [
                f"{c.args[1]} → value={c.args[2]}, unit={c.args[3]}"
                for c in self.mock_write.mock_calls
            ]
            allure.attach(
                "\n".join(call_list),
                name="write_compound_attr call log",
                attachment_type=allure.attachment_type.TEXT,
            )

    # ------------------------------------------------------------------
    # 4️⃣  from_hdf5 – ensure read_compound_attr is called for every attribute
    # ------------------------------------------------------------------
    @allure.title("📥 Deserialise container ← HDF5 (read_compound_attr called)")
    @allure.description(
        """
        ``from_hdf5`` must walk over ``grp.attrs`` and invoke
        ``read_compound_attr`` for each attribute name.  The test supplies a
        mocked attribute iterator and a stubbed return value for each key.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.story("Deserialization")
    def test_from_hdf5_calls_read_compound_attr(self):
        # Fake attribute names that the group pretends to contain
        fake_attrs = ["frequency", "time", "comment"]
        self.fake_group.attrs.__iter__.return_value = iter(fake_attrs)

        # Configure read_compound_attr to return the original values
        def side_effect(_grp, key):
            return _SAMPLE_KV[key]

        self.mock_read.side_effect = side_effect

        with allure.step("Re‑create a container from the mocked group"):
            loaded = KeyValueCompoundContainer.from_hdf5(self.fake_group)

        with allure.step("Validate that read_compound_attr was called correctly"):
            expected_calls = [mock.call(self.fake_group, key) for key in fake_attrs]
            self.mock_read.assert_has_calls(expected_calls, any_order=False)
            assert self.mock_read.call_count == len(fake_attrs)

        with allure.step("The loaded container must match the original mapping"):
            assert loaded.kv_pairs == _SAMPLE_KV
            allure.attach(
                json.dumps(loaded.kv_pairs, default=str, indent=2),
                name="kv_pairs after from_hdf5",
                attachment_type=allure.attachment_type.JSON,
            )

    # ------------------------------------------------------------------
    # 5️⃣  pop – dict‑like removal with and without default
    # ------------------------------------------------------------------
    @allure.title("🗑️ Pop elements with and without default values")
    @allure.description(
        """
        The ``pop`` method should behave like ``dict.pop``:
        * return the stored tuple and remove the key,
        * raise ``KeyError`` when the key is missing and no default is supplied,
        * return the supplied default otherwise.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    @allure.story("Dictionary‑style helpers")
    def test_pop_behaviour(self):
        container = KeyValueCompoundContainer(kv_pairs=_SAMPLE_KV.copy())

        with allure.step("Pop an existing key and verify removal"):
            popped = container.pop("time")
            assert popped == _SAMPLE_KV["time"]
            assert "time" not in container.kv_pairs

        with allure.step("Pop a missing key with a default fallback"):
            default_value = ("fallback", None)
            popped_default = container.pop("nonexistent", default=default_value)
            assert popped_default == default_value

        with allure.step("Attempt pop without default on a missing key → KeyError"):
            with pytest.raises(KeyError):
                container.pop("still_missing")

    # ------------------------------------------------------------------
    # 6️⃣  __setitem__ validation – ensures TypeError on malformed tuple
    # ------------------------------------------------------------------
    @allure.title("⚙️ __setitem__ validates input tuple shape")
    @allure.description(
        """
        ``container[key] = (value, unit)`` forwards to ``set`` but also checks
        that the right‑hand side is a two‑element tuple.  Bad input must raise
        ``TypeError``.
        """
    )
    @allure.severity(allure.severity_level.MINOR)
    @allure.story("Convenience operator overloads")
    def test_setitem_invalid_tuple_raises(self):
        container = KeyValueCompoundContainer()

        with allure.step("Assign a non‑tuple value → TypeError"):
            with pytest.raises(TypeError):
                container["foo"] = "not a tuple"

        with allure.step("Assign a tuple of wrong length → TypeError"):
            with pytest.raises(TypeError):
                container["bar"] = (1, Unit.MICROSECOND, "extra")

    # ------------------------------------------------------------------
    # 7️⃣  Full round‑trip (set → to_hdf5 → from_hdf5) using mocks
    # ------------------------------------------------------------------
    @allure.title("🔄 Full round‑trip serialization & deserialization")
    @allure.description(
        """
        Combine the previous unit tests: create a container, serialise it,
        then deserialise it again.  The mock helpers guarantee that the data
        flow is preserved without touching the file system.
        """
    )
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.story("End‑to‑end behaviour (mocked I/O)")
    def test_round_trip(self):
        original = KeyValueCompoundContainer(kv_pairs=_SAMPLE_KV.copy())

        with allure.step("Serialise to the mocked group"):
            original.to_hdf5(self.fake_group)

        # make the read‑side return exactly what the write side supplied
        def read_side_effect(_grp, key):
            # write_compound_attr was called with (group, key, value, unit)
            # Retrieve those arguments from the call log to simulate a perfect round‑trip.
            for call in self.mock_write.mock_calls:
                if call.args[1] == key:
                    return call.args[2], call.args[3]
            raise KeyError(key)

        self.mock_read.side_effect = read_side_effect
        # make the attribute iterator reflect the written keys
        self.fake_group.attrs.__iter__.return_value = iter(
            [call.args[1] for call in self.mock_write.mock_calls]
        )

        with allure.step("Deserialise from the same mocked group"):
            reconstructed = KeyValueCompoundContainer.from_hdf5(self.fake_group)

        with allure.step("Validate that the reconstructed container matches the original"):
            assert reconstructed.kv_pairs == original.kv_pairs
            allure.attach(
                json.dumps(reconstructed.kv_pairs, default=str, indent=2),
                name="reconstructed kv_pairs",
                attachment_type=allure.attachment_type.JSON,
            )
