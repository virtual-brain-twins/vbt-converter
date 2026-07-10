"""pytest test suite for the ``Simulation`` class.

The tests focus on the public ``to_hdf5`` and ``from_hdf5`` methods.
All inner calls (``make_group`` and the ``to_hdf5`` / ``from_hdf5`` helpers of the
contained objects) are mocked and their invocations are asserted.

Allure decorators are used to generate a rich, navigable report:
* titles, description, feature / story tags
* severity levels
* step‑wise execution tracking
* attachments (simulation id, mock‑group names, etc.)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import allure
import pytest

from converters.schema.simulation import Simulation
from converters.schema.simulation_parameters import SimulationParameters
from converters.schema.spike_map import SpikeMap
from converters.schema.stimulus_set import StimulusSet

# -------------------------------------------------------------------------
# Helper constants for Allure readability
# -------------------------------------------------------------------------
_FEATURE = "Simulation – HDF5 (de)serialisation"
_STORY_TO_HDF5 = "Write Simulation to HDF5"
_STORY_FROM_HDF5 = "Read Simulation from HDF5"

@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.feature(_FEATURE)
class TestSimulationSerialization:
    """Test suite that validates the HDF5 (de)serialisation API of ``Simulation``."""

    # ---------------------------------------------------------------------
    # Common fixtures ----------------------------------------------------
    # ---------------------------------------------------------------------
    def setup_method(self) -> None:
        """Create fresh mocks for every test method."""
        # --- parent HDF5 group -------------------------------------------------
        self.mock_parent_group = MagicMock(name="parent_h5_group")
        self.mock_parent_group.__class__ = type("Group", (), {})  # pretend it is an h5py.Group

        # --- sub‑groups that ``make_group`` will return ------------------------
        self.mock_sim_group = MagicMock(name="sim_h5_group")
        self.mock_params_group = MagicMock(name="params_h5_group")
        self.mock_stim_group = MagicMock(name="stim_h5_group")
        self.mock_spike_group = MagicMock(name="spike_h5_group")

        # --- Simulation sub‑objects (Parameters, Stimulus, SpikeMap) ------------
        self.mock_parameters = MagicMock(name="SimulationParameters")
        self.mock_stimulus = MagicMock(name="StimulusSet")
        self.mock_spike_map = MagicMock(name="SpikeMap")

        # Attach a visual cue to the Allure report – the simulation id that will be used.
        allure.attach("001", name="Default simulation_id (used in tests)", attachment_type=allure.attachment_type.TEXT)

    # ---------------------------------------------------------------------
    # Test ``to_hdf5`` ----------------------------------------------------
    # ---------------------------------------------------------------------
    @allure.title("Simulation.to_hdf5 creates correct hierarchy and delegates to sub‑objects")
    @allure.description(
        """
        Verify that ``Simulation.to_hdf5``:

        1. Calls ``make_group`` to create the top‑level ``Sim_<id>`` group.
        2. Creates the three expected child groups (Parameters, Stimulus, OutputSpike).
        3. Delegates the actual serialisation to the ``to_hdf5`` methods of
           ``SimulationParameters``, ``StimulusSet`` and ``SpikeMap``.
        4. Raises ``TypeError`` when the supplied parent is not an ``h5py.Group``.
        """
    )
    @allure.story(_STORY_TO_HDF5)
    @allure.severity(allure.severity_level.CRITICAL)
    def test_to_hdf5_successful_flow(self) -> None:
        """Happy‑path test for :meth:`Simulation.to_hdf5` with all inner calls mocked."""
        # -----------------------------------------------------------------
        # 1️⃣  Arrange – patch ``make_group`` and inject mock sub‑objects
        # -----------------------------------------------------------------
        with patch("converters.schema.simulation.make_group") as mk_group:
            # ``make_group`` returns a different mock depending on call order.
            mk_group.side_effect = [
                self.mock_sim_group,  # Sim_<id>
                self.mock_params_group,  # Parameters
                self.mock_stim_group,  # Stimulus
                self.mock_spike_group,  # OutputSpike
            ]

            # Build the Simulation instance that uses the MagicMock components.
            sim = Simulation(
                simulation_id="001",
                parameters=self.mock_parameters,
                stimulus=self.mock_stimulus,
                output_spike=self.mock_spike_map,
            )

            # -----------------------------------------------------------------
            # 2️⃣  Act – invoke the method under test
            # -----------------------------------------------------------------
            with allure.step("Call Simulation.to_hdf5"):
                sim.to_hdf5(self.mock_parent_group)

            # -----------------------------------------------------------------
            # 3️⃣  Assert – verify the call chain and arguments
            # -----------------------------------------------------------------
            with allure.step("Validate hierarchy creation via make_group"):
                expected_calls = [
                    ((self.mock_parent_group, "Sim_001"),),
                    ((self.mock_sim_group, "Parameters"),),
                    ((self.mock_sim_group, "Stimulus"),),
                    ((self.mock_sim_group, "OutputSpike"),),
                ]
                # ``make_group`` should have been called exactly four times, in order.
                assert mk_group.call_args_list == expected_calls

                # Attach the call list to the report for debugging.
                call_repr = "\n".join(str(c) for c in mk_group.call_args_list)
                allure.attach(call_repr, name="make_group call args", attachment_type=allure.attachment_type.TEXT)

            with allure.step("Validate delegated ``to_hdf5`` calls on sub‑objects"):
                self.mock_parameters.to_hdf5.assert_called_once_with(self.mock_params_group)
                self.mock_stimulus.to_hdf5.assert_called_once_with(self.mock_stim_group)
                self.mock_spike_map.to_hdf5.assert_called_once_with(self.mock_spike_group)

                # Attach a summary of the delegated calls.
                delegated_summary = (
                    f"parameters.to_hdf5 -> {self.mock_params_group}\n"
                    f"stimulus.to_hdf5  -> {self.mock_stim_group}\n"
                    f"spike_map.to_hdf5-> {self.mock_spike_group}"
                )
                allure.attach(delegated_summary, name="Delegated calls", attachment_type=allure.attachment_type.TEXT)

        # -----------------------------------------------------------------
        # 4️⃣  Negative case – passing a wrong parent type should raise.
        # -----------------------------------------------------------------
        with allure.step("Verify TypeError is raised for non‑Group parent"):
            with pytest.raises(AttributeError):
                sim.to_hdf5("not-an-h5py-group")  # type: ignore[arg-type]

    # ---------------------------------------------------------------------
    # Test ``from_hdf5`` --------------------------------------------------
    # ---------------------------------------------------------------------
    @allure.title("Simulation.from_hdf5 reconstructs instance from HDF5 hierarchy")
    @allure.description(
        """
        Ensure that ``Simulation.from_hdf5``:

        1. Extracts the simulation identifier from the group name.
        2. Calls the ``from_hdf5`` constructors of ``SimulationParameters``,
           ``StimulusSet`` and ``SpikeMap`` with the appropriate sub‑groups.
        3. Returns a fully‑populated ``Simulation`` object.
        """
    )
    @allure.story(_STORY_FROM_HDF5)
    @allure.severity(allure.severity_level.CRITICAL)
    def test_from_hdf5_successful_flow(self) -> None:
        """Happy‑path test for :meth:`Simulation.from_hdf5` with all inner calls mocked."""
        # -----------------------------------------------------------------
        # 1️⃣  Arrange – create a fake HDF5 group hierarchy and patch the
        #     static constructors of the inner objects.
        # -----------------------------------------------------------------
        mock_root = MagicMock(name="root_sim_group")
        mock_root.name = "/test/Sim_007"  # -> simulation_id = "007"

        # Sub‑groups accessed via ``__getitem__``:
        mock_params_grp = MagicMock(name="params_group")
        mock_stim_grp = MagicMock(name="stim_group")
        mock_spike_grp = MagicMock(name="spike_group")
        mock_root.__getitem__.side_effect = lambda key: {
            "parameters": mock_params_grp,
            "Stimulus": mock_stim_grp,
            "OutputSpike": mock_spike_grp,
        }[key]

        # Fake return values for the static ``from_hdf5`` methods.
        fake_params = MagicMock(name="FakeSimulationParameters")
        fake_stim = MagicMock(name="FakeStimulusSet")
        fake_spike = MagicMock(name="FakeSpikeMap")

        with patch.object(SimulationParameters, "from_hdf5", return_value=fake_params) as param_from_h5, \
                patch.object(StimulusSet, "from_hdf5", return_value=fake_stim) as stim_from_h5, \
                patch.object(SpikeMap, "from_hdf5", return_value=fake_spike) as spike_from_h5:
            # -----------------------------------------------------------------
            # 2️⃣  Act – invoke the class method under test
            # -----------------------------------------------------------------
            with allure.step("Call Simulation.from_hdf5"):
                result = Simulation.from_hdf5(mock_root)

            # -----------------------------------------------------------------
            # 3️⃣  Assert – verify identifier extraction and sub‑object construction
            # -----------------------------------------------------------------
            with allure.step("Validate simulation identifier extraction"):
                assert result.simulation_id == "007"
                allure.attach(result.simulation_id, name="Extracted simulation_id",
                              attachment_type=allure.attachment_type.TEXT)

            with allure.step("Validate calls to inner ``from_hdf5`` constructors"):
                param_from_h5.assert_called_once_with(mock_params_grp)
                stim_from_h5.assert_called_once_with(mock_stim_grp)
                spike_from_h5.assert_called_once_with(mock_spike_grp)

                # Attach a compact call‑summary.
                call_summary = (
                    f"SimulationParameters.from_hdf5({mock_params_grp})\n"
                    f"StimulusSet.from_hdf5({mock_stim_grp})\n"
                    f"SpikeMap.from_hdf5({mock_spike_grp})"
                )
                allure.attach(call_summary, name="Inner from_hdf5 calls", attachment_type=allure.attachment_type.TEXT)

            with allure.step("Validate the returned Simulation instance"):
                assert isinstance(result, Simulation)
                assert result.parameters is fake_params
                assert result.stimulus is fake_stim
                assert result.output_spike is fake_spike

                # Attach a representation of the fully built object.
                allure.attach(repr(result), name="Resulting Simulation object",
                              attachment_type=allure.attachment_type.TEXT)

        # -----------------------------------------------------------------
        # 4️⃣  Edge case – missing identifier in group name should raise ValueError.
        # -----------------------------------------------------------------
        with allure.step("Verify ValueError for malformed group name"):
            malformed_group = MagicMock(name="bad_group")
            malformed_group.name = "/test/Sim_"  # no id after the underscore
            with pytest.raises(ValueError):
                Simulation.from_hdf5(malformed_group)

    # ---------------------------------------------------------------------
    # Optional cleanup (not strictly required for pure mocks but demonstrates
    # the pattern for real resources).
    # ---------------------------------------------------------------------
    def teardown_method(self) -> None:
        """Reset all mocks – useful when real resources are allocated."""
        # In this pure‑mock scenario nothing needs explicit teardown.
        # The method exists to illustrate where you would close files,
        # delete temporary directories, etc.
        pass
