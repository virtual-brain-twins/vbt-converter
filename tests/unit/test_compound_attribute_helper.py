"""pytest test suite for ``compound_attr.py`` with full Allure instrumentation."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest import mock

import allure
import h5py
import numpy as np
import pytest

from converters.utils.compound_attribute_helper import write_compound_attr, read_compound_attr
from converters.utils.units import Unit


# --------------------------------------------------------------------------- #
# Helper – a very small fake HDF5 group that only implements the ``attrs`` API #
# --------------------------------------------------------------------------- #
class _FakeAttrs:
    """A minimal ``attrs`` container that records ``create`` calls."""

    def __init__(self) -> None:
        self._created: Dict[str, Any] = {}

    def create(self, name: str, value: Any) -> None:
        """Mimic ``h5py.Group.attrs.create``."""
        self._created[name] = value

    def __getitem__(self, name: str) -> Any:
        """Allow ``parent.attrs[name]`` in the *read* path."""
        return self._created[name]


class _FakeH5Group:
    """A tiny stand‑in for ``h5py.Group`` used only for its ``attrs`` attribute."""

    def __init__(self) -> None:
        self.attrs = _FakeAttrs()


# --------------------------------------------------------------------------- #
# Test class – Allure‑rich, fully mocked, pytest‑compatible                     #
# --------------------------------------------------------------------------- #
@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for HDF5 Compound Attribute Utilities")
@allure.description("Unit tests for the `write_compound_attr` and `read_compound_attr` helpers.")
@allure.feature("HDF5 Compound Attribute Utilities")
@allure.story("Write & read compound attributes with automatic dtype handling")
class TestCompoundAttributeUtilities:
    """All tests are isolated; each method receives a pristine mock HDF5 group."""

    # ------------------------------------------------------------------- #
    # 1️⃣  Common (set‑up / tear‑down) helpers                               #
    # ------------------------------------------------------------------- #
    def setup_method(self, method):
        """Create a fresh fake HDF5 group and attach a log step."""
        with allure.step("Setup – create a fresh fake HDF5 group"):
            self.mock_group = _FakeH5Group()
            allure.attach(
                body=str(self.mock_group),
                name="Fake HDF5 group",
                attachment_type=allure.attachment_type.TEXT,
            )

    def teardown_method(self, method):
        """Log the teardown – useful when the test suite runs in parallel."""
        with allure.step("Teardown – discard the fake HDF5 group"):
            del self.mock_group

    # ------------------------------------------------------------------- #
    # 2️⃣  Test: writing a *native* integer together with a Unit object       #
    # ------------------------------------------------------------------- #
    @allure.title("Write a scalar integer attribute with a Unit object")
    @allure.description(
        """
        Verify that ``write_compound_attr``:

        * Calls the private ``_dtype_for_value`` helper with the exact integer.
        * Normalises a ``Unit`` instance to its string representation.
        * Stores a NumPy compound array whose fields contain the expected
          ``value`` and ``unit``.
        * Invokes ``parent.attrs.create`` exactly once with the correct arguments.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    def test_write_int_with_unit(self, monkeypatch):
        attr_name = "time"
        input_value = 42
        input_unit = Unit("milliseconds")

        # --------------------------------------------------- #
        # Mock the internal helpers we want to *spy* on      #
        # --------------------------------------------------- #
        dtype_spy = mock.Mock(return_value=np.dtype(np.int64))
        monkeypatch.setattr(
            "converters.utils.compound_attribute_helper._dtype_for_value", dtype_spy, raising=True
        )

        # ``parent.attrs.create`` is a real method on the fake group,
        # but we replace it with a mock to capture the call arguments.
        create_spy = mock.Mock(wraps=self.mock_group.attrs.create)
        self.mock_group.attrs.create = create_spy

        # --------------------------------------------------- #
        # Step 1 – invoke the function under test               #
        # --------------------------------------------------- #
        with allure.step("Call ``write_compound_attr`` with integer value"):
            write_compound_attr(
                parent=self.mock_group,
                name=attr_name,
                value=input_value,
                unit=input_unit,
            )

        # --------------------------------------------------- #
        # Step 2 – assertions on the *internal* interactions   #
        # --------------------------------------------------- #
        with allure.step("Assert that ``_dtype_for_value`` was called once"):
            dtype_spy.assert_called_once_with(input_value)

        with allure.step("Assert the HDF5 ``attrs.create`` call"):
            create_spy.assert_called_once()
            created_name, created_compound = create_spy.call_args[0]

            # ---- name ------------------------------------------------------- #
            assert created_name == attr_name, "Attribute name mismatch"

            # ---- value ------------------------------------------------------ #
            assert (
                    created_compound["value"] == input_value
            ), "Stored integer value does not match the input"

            # ---- unit ------------------------------------------------------- #
            # The ``Unit`` class exposes the actual string via the ``value`` attribute.
            expected_unit_str = input_unit.value
            assert (
                    created_compound["unit"] == expected_unit_str
            ), "Unit string was not normalised correctly"

        # --------------------------------------------------- #
        # Attach the final compound array to the Allure report #
        # --------------------------------------------------- #
        allure.attach(
            body=json.dumps(
                {"value": int(created_compound["value"]), "unit": str(created_compound["unit"])}
            ),
            name="Created compound attribute (JSON view)",
            attachment_type=allure.attachment_type.JSON,
        )

    # ------------------------------------------------------------------- #
    # 3️⃣  Test: writing a non‑serialisable object → JSON encoding path     #
    # ------------------------------------------------------------------- #
    @allure.title(
        "Write a complex Python object (dict) that must be JSON‑encoded"
    )
    @allure.description(
        """
        This test forces the ``write_compound_attr`` logic down the JSON‑fallback
        branch:

        * ``_dtype_for_value`` raises ``TypeError``.
        * ``json.dumps`` is called with ``ensure_ascii=False``.
        * The resulting UTF‑8 string is stored as the ``value`` field of the compound.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    def test_write_dict_is_json_encoded(self, monkeypatch):
        attr_name = "metadata"
        input_value = {"a": 1, "b": [2, 3]}
        input_unit = "m·s⁻¹"  # plain string – no ``Unit`` object

        # --------------------------------------------------- #
        # 1️⃣  Mock _dtype_for_value → raise TypeError          #
        # --------------------------------------------------- #
        dtype_mock = mock.Mock(side_effect=TypeError("fallback to JSON"))
        monkeypatch.setattr(
            "converters.utils.compound_attribute_helper._dtype_for_value", dtype_mock, raising=True
        )

        # --------------------------------------------------- #
        # 2️⃣  Spy on json.dumps                               #
        # --------------------------------------------------- #
        json_dump_spy = mock.Mock(return_value=json.dumps(input_value, ensure_ascii=False))
        monkeypatch.setattr(json, "dumps", json_dump_spy)

        # --------------------------------------------------- #
        # 3️⃣  Mock the HDF5 attribute creation                 #
        # --------------------------------------------------- #
        create_spy = mock.Mock(wraps=self.mock_group.attrs.create)
        self.mock_group.attrs.create = create_spy

        # --------------------------------------------------- #
        # Execute the function under test                       #
        # --------------------------------------------------- #
        with allure.step("Invoke ``write_compound_attr`` with a dict"):
            write_compound_attr(
                parent=self.mock_group,
                name=attr_name,
                value=input_value,
                unit=input_unit,
            )

        # --------------------------------------------------- #
        # Verify the chain of internal calls                    #
        # --------------------------------------------------- #
        with allure.step("Check that dtype helper raised TypeError"):
            dtype_mock.assert_called_once_with(input_value)

        with allure.step("Check that json.dumps was called correctly"):
            json_dump_spy.assert_called_once_with(
                input_value, ensure_ascii=False
            )

        with allure.step("Validate the stored compound structure"):
            create_spy.assert_called_once()
            created_name, created_compound = create_spy.call_args[0]

            assert created_name == attr_name
            # The ``value`` field must be the JSON string, not the original dict.
            expected_json = json.dumps(input_value, ensure_ascii=False)
            assert created_compound["value"] == expected_json
            # Unit handling – plain string must be stored unchanged.
            assert created_compound["unit"] == input_unit

        # --------------------------------------------------- #
        # Attach a visual representation of the stored data   #
        # --------------------------------------------------- #
        allure.attach(
            body=f"value (JSON) = {expected_json}\nunit = {input_unit}",
            name="Resulting compound attribute (raw view)",
            attachment_type=allure.attachment_type.TEXT,
        )

    # ------------------------------------------------------------------- #
    # 4️⃣  Test: reading back a previously written attribute             #
    # ------------------------------------------------------------------- #
    @allure.title("Read a stored compound attribute and verify its tuple")
    @allure.description(
        """
        The ``read_compound_attr`` function should:

        * Retrieve the raw ``value`` and ``unit`` fields from the compound.
        * Decode bytes to ``str`` when the HDF5 library returns a `bytes` object.
        * Return ``None`` when the unit field is an empty string.
        """
    )
    @allure.severity(allure.severity_level.MINOR)
    @pytest.mark.parametrize(
        "raw_value, raw_unit, expected_value, expected_unit",
        [
            # Normal case – both fields are plain strings
            (b"42", b"milliseconds", "42", Unit.MILLISECOND),
            # Numeric value – returned as a native Python scalar
            (np.int64(300), "nanoseconds", 300, Unit.NANOSECOND),
            # Empty unit → ``None``
            ("hello", "", "hello", None),
        ],
    )
    def test_read_compound_attribute(
            self,
            raw_value: Any,
            raw_unit: Any,
            expected_value: Any,
            expected_unit: Any,
            monkeypatch,
    ):
        attr_name = "sample"

        # --------------------------------------------------- #
        # Build a *real* NumPy compound matching the implementation #
        # --------------------------------------------------- #
        value_dtype = (
            h5py.string_dtype() if isinstance(raw_value, (bytes, str)) else np.dtype(type(raw_value))
        )
        compound_dtype = np.dtype(
            [("value", value_dtype), ("unit", h5py.string_dtype())]
        )
        compound = np.empty((), dtype=compound_dtype)
        compound["value"] = raw_value
        compound["unit"] = raw_unit

        # --------------------------------------------------- #
        # Inject the compound into the fake group's attrs dict #
        # --------------------------------------------------- #
        self.mock_group.attrs._created[attr_name] = compound

        # --------------------------------------------------- #
        # Call the function under test                           #
        # --------------------------------------------------- #
        with allure.step("Call ``read_compound_attr`` and capture the result"):
            result_value, result_unit = read_compound_attr(
                parent=self.mock_group,
                name=attr_name,
            )

        # --------------------------------------------------- #
        # Assert correct conversion & normalisation               #
        # --------------------------------------------------- #
        with allure.step("Validate the returned tuple"):
            assert result_value == expected_value, "Incorrect value returned"
            assert result_unit == expected_unit, "Incorrect unit handling"

        # --------------------------------------------------- #
        # Attach the raw NumPy compound for debugging purposes   #
        # --------------------------------------------------- #
        allure.attach(
            body=str(compound),
            name="Raw compound array (debug)",
            attachment_type=allure.attachment_type.TEXT,
        )
