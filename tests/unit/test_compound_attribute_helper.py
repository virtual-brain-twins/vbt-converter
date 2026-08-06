import json
from unittest import mock
from unittest.mock import MagicMock

import allure
import pytest

from converters.utils.compound_attribute_helper import write_compound_attr, read_compound_attr
from converters.utils.units import Unit


# ----------------------------------------------------------------------
# Helper – a tiny “pretty” wrapper around a structured‑array‑like dict.
# ----------------------------------------------------------------------
def _make_fake_compound(type_name: str, serialized: str, unit_str: str = ""):
    """
    Produce a fake compound attribute that mimics the ``np.empty((), dtype=_COMPOUND_DTYPE)``
    object returned by H5PY.  The object behaves like a NumPy structured array
    when accessed by the ``[]`` operator.
    """
    return {
        "type": type_name,
        "value": serialized,
        "unit": unit_str,
    }


# ----------------------------------------------------------------------
# The actual test class
# ----------------------------------------------------------------------
@allure.feature("HDF5 Compound Attribute Utilities")
class TestCompoundAttributeUtils:
    """
    A pytest‑style test class that validates the public helpers
    :func:`write_compound_attr` and :func:`read_compound_attr`.

    The tests **mock** the internal encoder/decoder functions as well
    as the H5PY ``attrs`` mapping so that we can:

    * verify proper delegation,
    * assert the exact payload that is written,
    * and guarantee that the decoder receives the expected arguments.
    """

    # ------------------------------------------------------------------
    # Optional fixture‑style set‑up / tear‑down – creates fresh mocks for
    # every test method.
    # ------------------------------------------------------------------
    def setup_method(self):
        """Create fresh MagicMock objects for each test case."""
        self.parent_group = MagicMock(name="h5py.Group")
        # ``attrs`` is itself a mapping‑like object on an H5PY group.
        self.parent_group.attrs = MagicMock(name="attrs")

    def teardown_method(self):
        """Release references – not strictly required but helps CPython GC."""
        self.parent_group = None

    # ------------------------------------------------------------------
    # WRITE TESTS
    # ------------------------------------------------------------------
    @allure.title("Write a compound attribute – normal flow")
    @allure.description(
        """
        Verify that **write_compound_attr**:

        1. Calls the internal ``_encode_value`` with the original Python value.
        2. Builds a compound NumPy‑structured object with the correct fields.
        3. Calls ``parent.attrs.create`` with the attribute name and the compound
           payload.

        The test uses a ``MagicMock`` for the parent group, therefore no real
        HDF5 file is touched.
        """
    )
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.story("Create a new attribute on a group")
    @mock.patch("converters.utils.compound_attribute_helper._encode_value")
    def test_write_compound_attribute_calls_encoder_and_attrs_create(
            self,
            mock_encode,
    ):
        # ------------------------------------------------------------------
        # Arrange – test data & mock behaviours
        # ------------------------------------------------------------------
        test_value = {"temperature": 23.5, "unit": "C"}  # any JSON‑serialisable object
        expected_type = "json"
        expected_serial = json.dumps(test_value)

        # ``_encode_value`` should return the type‑identifier and serialised string.
        mock_encode.return_value = (expected_type, expected_serial)

        attr_name = "sensor_data"

        # ------------------------------------------------------------------
        # Act – call the function under test
        # ------------------------------------------------------------------
        with allure.step("Calling write_compound_attr"):
            write_compound_attr(
                parent=self.parent_group,
                name=attr_name,
                value=test_value,
                unit=None,  # no unit for this case
            )

        # ------------------------------------------------------------------
        # Assert – encoder called exactly once with the original value
        # ------------------------------------------------------------------
        with allure.step("Asserting _encode_value was invoked correctly"):
            mock_encode.assert_called_once_with(test_value)

        # ------------------------------------------------------------------
        # Assert – attrs.create called with the expected name and payload
        # ------------------------------------------------------------------
        with allure.step("Asserting attrs.create received the right payload"):
            # Capture the numpy‑structured array that was handed to `create`.
            # It is a zero‑dimensional numpy array – we can convert it to a dict
            # for easier inspection.
            assert self.parent_group.attrs.create.called, "attrs.create was never called"
            call_args, call_kwargs = self.parent_group.attrs.create.call_args

            # Arguments: (name, compound)
            created_name, compound_array = call_args
            assert created_name == attr_name, "Attribute name mismatch"

            # Convert the structured array into a normal dict for readability.
            compound_dict = {
                "type": compound_array["type"].item(),
                "value": compound_array["value"].item(),
                "unit": compound_array["unit"].item(),
            }

            # Attach the captured payload to the Allure report for visibility.
            allure.attach(
                json.dumps(compound_dict, indent=2),
                name="compound_payload_written",
                attachment_type=allure.attachment_type.JSON,
            )

            # Verify each field.
            assert compound_dict["type"] == expected_type, "Stored type identifier mismatch"
            assert compound_dict["value"] == expected_serial, "Stored serialized value mismatch"
            assert compound_dict["unit"] == "", "Unit field should be empty when None is passed"

    @allure.title("Write a compound attribute – with a Unit")
    @allure.description(
        """
        The same flow as the previous test, this time passing a
        ``converters.utils.units.Unit`` enum value.  The unit string must be
        stored in the ``unit`` field of the compound attribute.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.story("Create a new attribute on a group – with unit")
    @mock.patch("converters.utils.compound_attribute_helper._encode_value")
    def test_write_attribute_includes_unit(
            self,
            mock_encode,
    ):
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        from converters.utils.units import Unit  # real enum – lightweight import

        test_value = 42
        mock_encode.return_value = ("int", json.dumps(test_value))

        attr_name = "answer"
        unit = Unit.MICROMETER

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        with allure.step("Calling write_compound_attr with a Unit"):
            write_compound_attr(
                parent=self.parent_group,
                name=attr_name,
                value=test_value,
                unit=unit,
            )

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        mock_encode.assert_called_once_with(test_value)

        # Capture the created payload.
        created_name, compound_array = self.parent_group.attrs.create.call_args[0]

        compound_dict = {
            "type": compound_array["type"].item(),
            "value": compound_array["value"].item(),
            "unit": compound_array["unit"].item(),
        }

        allure.attach(
            json.dumps(compound_dict, indent=2),
            name="compound_payload_with_unit",
            attachment_type=allure.attachment_type.JSON,
        )

        assert created_name == attr_name
        assert compound_dict["unit"] == unit.value

    # ------------------------------------------------------------------
    # READ TESTS
    # ------------------------------------------------------------------
    @allure.title("Read a compound attribute – normal flow")
    @allure.description(
        """
        Verify that **read_compound_attr**:

        1. Retrieves the raw compound attribute from ``parent.attrs``.
        2. Decodes the ``type`` and ``value`` fields using the internal
           ``_decode_value`` helper.
        3. Constructs a ``Unit`` enum when a non‑empty unit string is present.

        All external interactions (attrs access and the decoder) are mocked.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    @allure.story("Read back a previously written attribute")
    @mock.patch("converters.utils.compound_attribute_helper._decode_value")
    @mock.patch("converters.utils.compound_attribute_helper.Unit")
    def test_read_compound_attribute_calls_decoder_and_unit_lookup(
            self,
            mock_unit_enum,
            mock_decode,
    ):
        # ------------------------------------------------------------------
        # Arrange – fake stored compound attribute
        # ------------------------------------------------------------------
        stored_type = "int"
        stored_serial = json.dumps(123)
        stored_unit = Unit.MICROMETER.value

        fake_compound = _make_fake_compound(
            type_name=stored_type,
            serialized=stored_serial,
            unit_str=stored_unit,
        )

        # ``attrs.__getitem__`` is what the implementation uses.
        self.parent_group.attrs.__getitem__.return_value = fake_compound

        # Mock the decoder to simply return the original integer.
        mock_decode.return_value = 123

        # Mock the Unit enum – we want it to return a sentinel object.
        unit_instance = MagicMock(name="UnitInstance")
        mock_unit_enum.return_value = unit_instance

        attr_name = "mass"

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        with allure.step("Calling read_compound_attr"):
            value, unit = read_compound_attr(parent=self.parent_group, name=attr_name)

        # ------------------------------------------------------------------
        # Assert – attrs.__getitem__ called with the correct name
        # ------------------------------------------------------------------
        with allure.step("Asserting attrs.__getitem__ was invoked"):
            self.parent_group.attrs.__getitem__.assert_called_once_with(attr_name)

        # ------------------------------------------------------------------
        # Assert – decoder called with the stored type/value strings
        # ------------------------------------------------------------------
        with allure.step("Asserting _decode_value received the right arguments"):
            mock_decode.assert_called_once_with(stored_type, stored_serial)
            assert value == 123, "Returned value does not match decoder output"

        # ------------------------------------------------------------------
        # Assert – Unit enum lookup performed correctly
        # ------------------------------------------------------------------
        with allure.step("Asserting Unit enum lookup"):
            mock_unit_enum.assert_called_once_with(stored_unit)
            assert unit is unit_instance, "Returned unit is not the enum instance"

        # ------------------------------------------------------------------
        # Attach the decoded result for the report
        # ------------------------------------------------------------------
        allure.attach(
            json.dumps({"value": value, "unit": stored_unit}, indent=2),
            name="decoded_result",
            attachment_type=allure.attachment_type.JSON,
        )

    @allure.title("Read a compound attribute – missing unit")
    @allure.description(
        """
        When the stored ``unit`` field is empty the function should return
        ``unit=None``.  This test ensures the empty‑string branch works.
        """
    )
    @allure.severity(allure.severity_level.MINOR)
    @allure.story("Read back an attribute without a unit")
    @mock.patch("converters.utils.compound_attribute_helper._decode_value")
    @mock.patch("converters.utils.units.Unit")
    def test_read_attribute_without_unit_returns_none(
            self,
            mock_unit_enum,
            mock_decode,
    ):
        # ------------------------------------------------------------------
        # Arrange – compound with an empty unit string
        # ------------------------------------------------------------------
        stored_type = "str"
        stored_serial = "hello world"
        fake_compound = _make_fake_compound(
            type_name=stored_type,
            serialized=stored_serial,
            unit_str="",  # empty -> no unit
        )
        self.parent_group.attrs.__getitem__.return_value = fake_compound

        mock_decode.return_value = stored_serial  # decoder just returns the string

        attr_name = "greeting"

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        with allure.step("Calling read_compound_attr for a unit‑less attribute"):
            value, unit = read_compound_attr(parent=self.parent_group, name=attr_name)

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        self.parent_group.attrs.__getitem__.assert_called_once_with(attr_name)
        mock_decode.assert_called_once_with(stored_type, stored_serial)
        mock_unit_enum.assert_not_called()  # Unit enum should never be touched
        assert unit is None, "Expected unit to be None when stored string is empty"
        assert value == stored_serial

        allure.attach(
            f"value: {value!r}\nunit: {unit}",
            name="read_without_unit",
            attachment_type=allure.attachment_type.TEXT,
        )

    @allure.title("Read a compound attribute – unknown unit raises")
    @allure.description(
        """
        The decoder should raise a ``ValueError`` when the stored unit string
        cannot be resolved to a member of the ``Unit`` enum.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.story("Error handling – unknown unit")
    @mock.patch("converters.utils.compound_attribute_helper._decode_value")
    @mock.patch("converters.utils.units.Unit")
    def test_read_attribute_unknown_unit_raises_value_error(
            self,
            mock_unit_enum,
            mock_decode,
    ):
        # ------------------------------------------------------------------
        # Arrange – simulate a bad unit string
        # ------------------------------------------------------------------
        stored_type = "float"
        stored_serial = json.dumps(3.14)
        bad_unit_str = "lightyears"

        fake_compound = _make_fake_compound(
            type_name=stored_type,
            serialized=stored_serial,
            unit_str=bad_unit_str,
        )
        self.parent_group.attrs.__getitem__.return_value = fake_compound

        # decoder works fine
        mock_decode.return_value = 3.14

        # make the enum raise ValueError on lookup
        mock_unit_enum.side_effect = ValueError("invalid enum value")

        attr_name = "distance"

        # ------------------------------------------------------------------
        # Act / Assert – the function must re‑raise a ValueError with a helpful message
        # ------------------------------------------------------------------
        with allure.step("Calling read_compound_attr and expecting a ValueError"):
            with pytest.raises(ValueError) as excinfo:
                read_compound_attr(parent=self.parent_group, name=attr_name)

        # Verify that the original exception was wrapped (the test only cares that
        # a ValueError bubbles up, not the exact message).
        assert "not a recognised unit" in str(excinfo.value)

        # ------------------------------------------------------------------
        # Attach the error message for the Allure report
        # ------------------------------------------------------------------
        allure.attach(
            str(excinfo.value),
            name="error_message",
            attachment_type=allure.attachment_type.TEXT,
        )
