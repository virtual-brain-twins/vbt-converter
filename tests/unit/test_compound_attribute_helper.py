# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

"""
Pytest‑style test suite for ``converters.utils.h5compound`` (the module that
contains ``_dtype_for_value``, ``write_compound_attr`` and ``read_compound_attr``).

All tests are heavily **mocked** – we never touch a real HDF5 file – and they are
fully instrumented with **Allure** annotations so that the generated report
contains a clear hierarchy, step‑by‑step execution trace and useful artefacts
(inputs, intermediate values and expected calls).

Run the suite with the Allure plugin, e.g.:

    pytest -q --alluredir=./allure-results
    allure serve ./allure-results
"""

from __future__ import annotations

import json
import sys
from typing import Any, Union
from unittest.mock import MagicMock, call, patch

import allure
import h5py
import numpy as np
import pytest

from converters.utils.compound_attribute_helper import _dtype_for_value, write_compound_attr, read_compound_attr


@allure.feature("HDF5 compound attribute utilities")
@allure.story("Low‑level dtype mapping")
class TestDtypeMapping:
    """Tests for the private ``_dtype_for_value`` helper."""

    @allure.title("Supported Python scalars are mapped to correct NumPy dtypes")
    @allure.description(
        """
        The function must recognise the four scalar types that can be stored
        directly in HDF5 (``bool``, ``int``, ``float`` and ``str``) and return a
        NumPy ``dtype`` that matches the HDF5 representation.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.parametrize(
        "python_value, expected_dtype",
        [
            (True, np.dtype(np.bool_)),
            (42, np.dtype(np.int64)),
            (3.14, np.dtype(np.float64)),
            ("Ω", h5py.string_dtype()),
        ],
    )
    def test_supported_scalars(self, python_value: Any, expected_dtype: np.dtype):
        """Direct call – no mocking required."""
        with allure.step("Call the private helper"):
            result_dtype = _dtype_for_value(python_value)

        with allure.step("Verify the returned dtype"):
            assert result_dtype == expected_dtype

    @allure.title("Unsupported values raise ``TypeError``")
    @allure.description(
        """
        Anything that is not one of the supported scalar types must raise a
        ``TypeError`` so that the caller can fall back to JSON encoding.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.parametrize("bad_value", [[], {}, object(), (1, 2)])
    def test_unsupported_values_raise(self, bad_value: Any):
        with allure.step("Calling the helper with an unsupported type"):
            with pytest.raises(TypeError) as exc:
                _dtype_for_value(bad_value)

        # Attach the error message for the report
        allure.attach(
            str(exc.value),
            name="Exception message",
            attachment_type=allure.attachment_type.TEXT,
        )


@allure.feature("HDF5 compound attribute utilities")
@allure.story("Writing a compound attribute")
class TestWriteCompoundAttr:
    """All tests mock the internal helpers and the HDF5 ``attrs`` API."""

    @allure.title("Native scalar value is written without JSON fallback")
    @allure.description(
        """
        When the value can be represented by a native HDF5 datatype, the function
        must:
        * call ``_dtype_for_value`` → receive the correct dtype,
        * **not** call ``json.dumps``,
        * build a compound dtype with ``value`` and ``unit`` fields,
        * invoke ``parent.attrs.create`` with a correctly populated scalar
          structured array.
        """
    )
    @allure.severity(allure.severity_level.BLOCKER)
    def test_write_native_scalar(self):
        # -------------------------------------------------------------------
        # Arrange – a fake HDF5 group and the mocks for the inner calls
        # -------------------------------------------------------------------
        parent = MagicMock(name="h5group")
        parent.attrs = MagicMock(name="attrs")

        test_name = "temperature"
        test_value = 300  # int → native
        test_unit = "K"

        # Mock the private dtype helper to return a predictable dtype
        with patch(
                "converters.utils.compound_attribute_helper._dtype_for_value",
                return_value=np.dtype(np.int64),
        ) as mock_dtype, patch(
            "converters.utils.compound_attribute_helper.json.dumps"
        ) as mock_json_dump:
            # ----------------------------------------------------------------
            # Act
            # ----------------------------------------------------------------
            with allure.step("Execute ``write_compound_attr``"):
                write_compound_attr(parent, test_name, test_value, test_unit)

            # ----------------------------------------------------------------
            # Assert – internal calls
            # ----------------------------------------------------------------
            with allure.step("Assert ``_dtype_for_value`` was called once"):
                mock_dtype.assert_called_once_with(test_value)

            with allure.step("Assert ``json.dumps`` was **not** called"):
                mock_json_dump.assert_not_called()

            with allure.step("Capture the created compound array"):
                # ``attrs.create`` receives (name, data).  Grab the data argument.
                create_call: call = parent.attrs.create.call_args
                assert create_call is not None, "attrs.create was never called"
                created_name, created_data = create_call.args

                # Attach the raw numpy void for visual inspection.
                allure.attach(
                    str(created_data),
                    name="Created compound (numpy.void)",
                    attachment_type=allure.attachment_type.TEXT,
                )

                # Verify the name argument
                assert created_name == test_name

                # Verify dtype fields
                expected_dtype = np.dtype(
                    [("value", np.dtype(np.int64)), ("unit", h5py.string_dtype())]
                )
                assert created_data.dtype == expected_dtype

                # Verify stored values
                assert created_data["value"] == test_value
                assert created_data["unit"] == test_unit

    @allure.title("Non‑native value triggers JSON encoding")
    @allure.description(
        """
        If ``_dtype_for_value`` raises ``TypeError`` the function must:
        * encode the value with ``json.dumps``,
        * use a string dtype for the ``value`` field,
        * create the compound attribute with the JSON string and the unit.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    def test_write_falls_back_to_json(self):
        # -------------------------------------------------------------------
        # Arrange
        # -------------------------------------------------------------------
        parent = MagicMock(name="h5group")
        parent.attrs = MagicMock(name="attrs")

        test_name = "metadata"
        complex_value = {"a": 1, "b": [2, 3]}  # not natively storable
        test_unit = "unitless"

        # Force the dtype helper to raise TypeError
        with patch(
                "converters.utils.compound_attribute_helper._dtype_for_value",
                side_effect=TypeError("unsupported type"),
        ) as mock_dtype, patch(
            "converters.utils.compound_attribute_helper.json.dumps",
            return_value=json.dumps(complex_value, ensure_ascii=False),
        ) as mock_json:
            # ----------------------------------------------------------------
            # Act
            # ----------------------------------------------------------------
            with allure.step("Execute ``write_compound_attr`` with a complex dict"):
                write_compound_attr(parent, test_name, complex_value, test_unit)

            # ----------------------------------------------------------------
            # Assert – internal workflow
            # ----------------------------------------------------------------
            with allure.step("``_dtype_for_value`` must raise → JSON path taken"):
                mock_dtype.assert_called_once_with(complex_value)

            with allure.step("``json.dumps`` must be called once"):
                mock_json.assert_called_once_with(
                    complex_value, ensure_ascii=False
                )
                json_payload = mock_json.return_value
                allure.attach(
                    json_payload,
                    name="JSON payload sent to HDF5",
                    attachment_type=allure.attachment_type.JSON,
                )

            with allure.step("Validate the created compound data"):
                create_args = parent.attrs.create.call_args
                assert create_args is not None
                created_name, created_data = create_args.args
                assert created_name == test_name

                # The dtype for ``value`` must now be a variable‑length string
                expected_dtype = np.dtype(
                    [
                        ("value", h5py.string_dtype()),
                        ("unit", h5py.string_dtype()),
                    ]
                )
                assert created_data.dtype == expected_dtype
                assert created_data["value"] == json_payload
                assert created_data["unit"] == test_unit

    @allure.title("Unit argument can be a ``Unit`` enum or plain string")
    @allure.description(
        """
        The helper must transparently accept both a ``Unit`` enum instance and a
        plain string.  The test verifies that the string stored in the compound
        attribute is always the *value* attribute of the enum.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    def test_write_accepts_enum_and_string_units(self):
        from converters.utils.units import Unit  # Imported locally to avoid circular imports

        parent = MagicMock(name="h5group")
        parent.attrs = MagicMock(name="attrs")

        test_name = "voltage"
        test_value = 101325
        enum_unit = Unit("microvolts")  # pretend Enum with .value == "Pa"
        plain_unit = "microvolts"

        # Helper that runs the write for a given ``unit`` and returns the stored unit
        def _run_write(unit: Union[str, Unit]) -> str:
            with patch(
                    "converters.utils.compound_attribute_helper._dtype_for_value",
                    return_value=np.dtype(np.int64),
            ):
                write_compound_attr(parent, test_name, test_value, unit)

            created_name, created_data = parent.attrs.create.call_args.args
            assert created_name == test_name
            return created_data["unit"]

        with allure.step("Write using a ``Unit`` enum"):
            stored_from_enum = _run_write(enum_unit)

        with allure.step("Write using a plain string"):
            stored_from_str = _run_write(plain_unit)

        with allure.step("Both approaches must store the same textual unit"):
            assert stored_from_enum == plain_unit
            assert stored_from_str == plain_unit


@allure.feature("HDF5 compound attribute utilities")
@allure.story("Reading a compound attribute")
class TestReadCompoundAttr:
    """All reads are mocked – we never hit the filesystem."""

    @allure.title("Read regular compound attribute and decode unit")
    @allure.description(
        """
        The attribute is stored as a structured scalar with ``value`` and
        ``unit`` fields.  The test checks that:
        * the ``_maybe_item`` helper is invoked for each field,
        * byte strings are decoded,
        * the unit string is converted to a ``Unit`` enum.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    def test_read_compound_with_valid_unit(self):
        from converters.utils.units import Unit  # local import

        # -------------------------------------------------------------------
        # Arrange – construct a fake compound attribute
        # -------------------------------------------------------------------
        raw_value = 9.81
        raw_unit = "microamperes"

        compound_dtype = np.dtype(
            [("value", np.dtype(np.float64)), ("unit", h5py.string_dtype())]
        )
        compound = np.empty((), dtype=compound_dtype)
        compound["value"] = raw_value
        compound["unit"] = raw_unit

        parent = MagicMock(name="h5group")
        parent.attrs = MagicMock(name="attrs")
        parent.attrs.__getitem__.return_value = compound

        # Patch ``_maybe_item`` so we can assert that it was called for each field
        with patch(
                "converters.utils.compound_attribute_helper._maybe_item",
                side_effect=lambda x: x.item() if hasattr(x, "item") else x,
        ) as mock_maybe_item:
            with allure.step("Execute ``read_compound_attr``"):
                with patch(
                        "converters.utils.compound_attribute_helper.isinstance",
                        side_effect=lambda obj, cls, _builtin=isinstance: True if cls is np.void else _builtin(obj, cls)
                ):
                    value, unit = read_compound_attr(parent, "gravity")

            # ----------------------------------------------------------------
            # Assert – helper usage
            # ----------------------------------------------------------------
            with allure.step("``_maybe_item`` must be called twice (value, unit)"):
                assert mock_maybe_item.call_count == 2
                mock_maybe_item.assert_has_calls(
                    [call(compound["value"]), call(compound["unit"])]
                )

            with allure.step("Returned value must be the original float"):
                assert value == raw_value

            with allure.step("Returned unit must be a ``Unit`` enum instance"):
                assert isinstance(unit, Unit)
                assert unit.value == raw_unit

            # ----------------------------------------------------------------
            # Attach the extracted data for the report
            # ----------------------------------------------------------------
            allure.attach(
                json.dumps({"value": value, "unit": unit.value}),
                name="Extracted payload",
                attachment_type=allure.attachment_type.JSON,
            )

    @allure.title("Read attribute where unit is missing (empty string)")
    @allure.severity(allure.severity_level.MINOR)
    def test_read_compound_without_unit(self):
        # -------------------------------------------------------------------
        # Arrange – a compound where ``unit`` is an empty string
        # -------------------------------------------------------------------
        raw_value = "some text"
        raw_unit = ""  # denotes “no unit”

        compound_dtype = np.dtype(
            [("value", h5py.string_dtype()), ("unit", h5py.string_dtype())]
        )
        compound = np.empty((), dtype=compound_dtype)
        compound["value"] = raw_value
        compound["unit"] = raw_unit

        parent = MagicMock(name="h5group")
        parent.attrs = MagicMock(name="attrs")
        parent.attrs.__getitem__.return_value = compound

        # -------------------------------------------------------------------
        # Act
        # -------------------------------------------------------------------
        with allure.step("Call ``read_compound_attr``"):
            with patch(
                    "converters.utils.compound_attribute_helper.isinstance",
                    side_effect=lambda obj, cls, _builtin=isinstance: True if cls is np.void else _builtin(obj, cls)
            ):
                value, unit = read_compound_attr(parent, "note")

        # -------------------------------------------------------------------
        # Assert
        # -------------------------------------------------------------------
        with allure.step("Value is returned unchanged"):
            assert value == raw_value

        with allure.step("Unit must be ``None`` when the stored string is empty"):
            assert unit is None

        allure.attach(
            f"value: {value!r}\nunit: {unit}",
            name="Result of read",
            attachment_type=allure.attachment_type.TEXT,
        )

    @allure.title("Invalid unit string raises ``ValueError``")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_read_invalid_unit_raises(self):
        from converters.utils.units import Unit  # noqa: F401  (just to ensure import works)

        raw_value = 1
        raw_unit = "not-a-real-unit"

        compound_dtype = np.dtype(
            [("value", np.dtype(np.int64)), ("unit", h5py.string_dtype())]
        )
        compound = np.empty((), dtype=compound_dtype)
        compound["value"] = raw_value
        compound["unit"] = raw_unit

        parent = MagicMock(name="h5group")
        parent.attrs = MagicMock(name="attrs")
        parent.attrs.__getitem__.return_value = compound

        with allure.step("Attempt to read the attribute with a bogus unit"):
            with patch(
                    "converters.utils.compound_attribute_helper.isinstance",
                    side_effect=lambda obj, cls, _builtin=isinstance: True if cls is np.void else _builtin(obj, cls)
            ):
                with pytest.raises(ValueError) as exc_info:
                    read_compound_attr(parent, "broken")

        # Attach the error for the report
        allure.attach(
            str(exc_info.value),
            name="Caught exception",
            attachment_type=allure.attachment_type.TEXT,
        )

        with allure.step("The message must mention the list of allowed units"):
            assert "Allowed units are" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Optional fixtures – they demonstrate the setup/teardown pattern requested.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_allure_environment():
    """
    Reset Allure (attach a short environment description) before each test.
    This also shows how ``setup_method`` / ``teardown_method`` can be replaced
    by a fixture when using pytest.
    """
    allure.attach(
        "Python {} | pytest {} | numpy {} | h5py {}".format(
            ".".join(map(str, sys.version_info[:3])),
            pytest.__version__,
            np.__version__,
            h5py.__version__,
        ),
        name="Test environment",
        attachment_type=allure.attachment_type.TEXT,
    )
    yield  # test runs here
    # No explicit teardown needed – placeholder for future cleanup logic.
