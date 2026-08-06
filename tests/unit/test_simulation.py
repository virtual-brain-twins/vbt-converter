"""
Pytest test suite for :class:`converters.schema.simulation.Simulation`.

All internal interactions are mocked – no real HDF5 files are created.
The suite is heavily annotated with Allure to produce a rich, searchable
HTML report.
"""

from __future__ import annotations

from unittest import mock

import allure
import h5py

from converters.schema.configuration import Configuration
# ----------------------------------------------------------------------
# Import the objects under test – adjust the import path to match your repo
# ----------------------------------------------------------------------
from converters.schema.simulation import Simulation
from converters.schema.simulation_parameters import SimulationParameters
from converters.schema.spike_map import SpikeMap
from converters.schema.stimuli_group import StimuliGroup

# ----------------------------------------------------------------------
# Helper constants for readability / report consistency
# ----------------------------------------------------------------------
_SIM_GROUP_NAME_FMT = "Sim_{sim_id}"
_ROOT_SUBGROUPS = ("Configuration", "Parameters", "Stimuli", "OutputSpikes")


@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for Simulation")
@allure.description("Unit tests for the `Simulation` class, mocking all I/O and verifying the public API.")
@allure.feature("Simulation – HDF5 (de)serialization")
class TestSimulationHDF5:
    """
    Test suite that verifies the behaviour of
    :meth:`Simulation.to_hdf5` and :meth:`Simulation.from_hdf5`.

    All *inner* calls are mocked and their usage is asserted, allowing the
    tests to run in pure memory while still guaranteeing that the public
    API orchestrates its collaborators correctly.
    """

    # ------------------------------------------------------------------
    # Optional fixtures – executed before/after each test method
    # ------------------------------------------------------------------
    def setup_method(self, method):
        """
        Prepare a fresh ``Simulation`` instance and a dummy ``h5py.Group``
        that will be used as the parent container for the *to_hdf5* test.
        """
        self.parent_group = mock.Mock(spec=h5py.Group, name="parent_group")
        # Use a deterministic id so the expected group names are obvious
        self.simulation = Simulation(simulation_id="42")

    def teardown_method(self, method):
        """Placeholder kept for symmetry – nothing to clean up right now."""
        pass

    # ------------------------------------------------------------------
    # Test: Serialization
    # ------------------------------------------------------------------
    @allure.story("Serialize a Simulation object")
    @allure.title("Simulation.to_hdf5 creates groups & delegates correctly")
    @allure.description(
        """
        Validates that ``Simulation.to_hdf5``:

        1️⃣ Calls ``make_group`` **five** times (root + four sub‑groups) with the
           correct parent object and group names.  
        2️⃣ Invokes the ``to_hdf5`` method of each component exactly once,
           passing the freshly created subgroup.  
        3️⃣ Does **not** touch the real filesystem (everything is mocked).

        The workflow is fully exercised in memory; any exception raised by a
        child would simply propagate – this is out of scope for the happy‑path
        test but the call‑graph is still verified.
        """
    )
    @allure.severity(allure.severity_level.NORMAL)
    def test_to_hdf5_calls_inner_methods(self):
        """Validate the internal workflow of ``Simulation.to_hdf5``."""

        # ------------------------------------------------------------------
        # 1️⃣  Patch everything that would touch disk or invoke real code
        # ------------------------------------------------------------------
        with mock.patch(
                "converters.schema.simulation.make_group", autospec=True
        ) as mk_group, \
                mock.patch.object(Configuration, "to_hdf5", autospec=True) as cfg_to_h5, \
                mock.patch.object(SimulationParameters, "to_hdf5", autospec=True) as params_to_h5, \
                mock.patch.object(StimuliGroup, "to_hdf5", autospec=True) as stim_to_h5, \
                mock.patch.object(SpikeMap, "to_hdf5", autospec=True) as spike_to_h5:
            # ------------------------------------------------------------------
            # 2️⃣  Prepare deterministic fake group objects for each ``make_group`` call
            # ------------------------------------------------------------------
            fake_root_grp = mock.MagicMock(spec=h5py.Group, name="Sim_42")
            fake_cfg_grp = mock.MagicMock(spec=h5py.Group, name="Configuration")
            fake_params_grp = mock.MagicMock(spec=h5py.Group, name="Parameters")
            fake_stim_grp = mock.MagicMock(spec=h5py.Group, name="Stimuli")
            fake_spike_grp = mock.MagicMock(spec=h5py.Group, name="OutputSpikes")

            mk_group.side_effect = [
                fake_root_grp,
                fake_cfg_grp,
                fake_params_grp,
                fake_stim_grp,
                fake_spike_grp,
            ]

            # ------------------------------------------------------------------
            # 3️⃣  Execute the method under test – wrapped in an Allure step
            # ------------------------------------------------------------------
            with allure.step("Call Simulation.to_hdf5"):
                self.simulation.to_hdf5(self.parent_group)

            # ------------------------------------------------------------------
            # 4️⃣  Assertions (each block lives in its own Allure step)
            # ------------------------------------------------------------------
            # ── make_group invocation validation ─────────────────────────────
            with allure.step("Validate ``make_group`` call sequence"):
                expected_calls = [
                    mock.call(self.parent_group, _SIM_GROUP_NAME_FMT.format(sim_id="42")),
                    mock.call(fake_root_grp, "Configuration"),
                    mock.call(fake_root_grp, "Parameters"),
                    mock.call(fake_root_grp, "Stimuli"),
                    mock.call(fake_root_grp, "OutputSpikes"),
                ]
                assert mk_group.mock_calls == expected_calls, \
                    "Unexpected make_group call order or arguments"

                # Attach the raw call list for debugging
                allure.attach(
                    "\n".join(str(c) for c in mk_group.mock_calls),
                    name="make_group_calls",
                    attachment_type=allure.attachment_type.TEXT,
                )

            # ── component ``to_hdf5`` validation ───────────────────────────────
            with allure.step("Validate each component's ``to_hdf5`` call"):
                cfg_to_h5.assert_called_once_with(self.simulation.configuration, fake_cfg_grp)
                params_to_h5.assert_called_once_with(self.simulation.parameters, fake_params_grp)
                stim_to_h5.assert_called_once_with(self.simulation.stimulus, fake_stim_grp)
                spike_to_h5.assert_called_once_with(self.simulation.output_spike, fake_spike_grp)

                # Human‑readable summary for the report
                calls_summary = "\n".join(
                    [
                        f"Configuration.to_hdf5 → {cfg_to_h5.call_args}",
                        f"SimulationParameters.to_hdf5 → {params_to_h5.call_args}",
                        f"StimuliGroup.to_hdf5 → {stim_to_h5.call_args}",
                        f"SpikeMap.to_hdf5 → {spike_to_h5.call_args}",
                    ]
                )
                allure.attach(
                    calls_summary,
                    name="component_to_hdf5_calls",
                    attachment_type=allure.attachment_type.TEXT,
                )

    # ------------------------------------------------------------------
    # Test: Deserialization
    # ------------------------------------------------------------------
    @allure.story("Deserialize a Simulation object")
    @allure.title("Simulation.from_hdf5 rebuilds object & calls inner from_hdf5")
    @allure.description(
        """
        Confirms that ``Simulation.from_hdf5``:

        1️⃣ Extracts the simulation identifier from the group name (``Sim_<id>``).  
        2️⃣ Calls the ``from_hdf5`` **class‑method** of each child object exactly once,
           feeding the matching subgroup.  
        3️⃣ Returns a fully‑populated ``Simulation`` instance.

        All HDF5 access is mocked, therefore the test runs in pure memory.
        """
    )
    @allure.severity(allure.severity_level.CRITICAL)
    def test_from_hdf5_calls_inner_methods(self):
        """Validate the internal workflow of ``Simulation.from_hdf5``."""

        # ------------------------------------------------------------------
        # 1️⃣  Mock the HDF5 hierarchy that ``from_hdf5`` expects
        # ------------------------------------------------------------------
        #   /some/path/Sim_99
        #       ├─ Configuration
        #       ├─ Parameters
        #       ├─ Stimuli
        #       └─ OutputSpikes
        # ------------------------------------------------------------------
        mock_root_grp = mock.MagicMock(spec=h5py.Group, name="Sim_99")
        mock_root_grp.name = "/some/path/Sim_99"

        mock_cfg_grp = mock.MagicMock(name="Configuration")
        mock_params_grp = mock.MagicMock(name="Parameters")
        mock_stim_grp = mock.MagicMock(name="Stimuli")
        mock_spike_grp = mock.MagicMock(name="OutputSpikes")

        # ``group['sub_name']`` returns the appropriate subgroup
        mock_root_grp.__getitem__.side_effect = lambda key: {
            "Configuration": mock_cfg_grp,
            "Parameters": mock_params_grp,
            "Stimuli": mock_stim_grp,
            "OutputSpikes": mock_spike_grp,
        }[key]

        # ------------------------------------------------------------------
        # 2️⃣  Patch the child ``from_hdf5`` class‑methods
        # ------------------------------------------------------------------
        with mock.patch.object(
                Configuration,
                "from_hdf5",
                autospec=True,
                return_value=Configuration(),
        ) as cfg_from, \
                mock.patch.object(
                    SimulationParameters,
                    "from_hdf5",
                    autospec=True,
                    return_value=SimulationParameters(),
                ) as params_from, \
                mock.patch.object(
                    StimuliGroup,
                    "from_hdf5",
                    autospec=True,
                    return_value=StimuliGroup(),
                ) as stim_from, \
                mock.patch.object(
                    SpikeMap,
                    "from_hdf5",
                    autospec=True,
                    return_value=SpikeMap(),
                ) as spike_from:
            # ------------------------------------------------------------------
            # 3️⃣  Execute the method under test
            # ------------------------------------------------------------------
            with allure.step("Call Simulation.from_hdf5"):
                result = Simulation.from_hdf5(mock_root_grp)

            # ------------------------------------------------------------------
            # 4️⃣  Verify identifier extraction
            # ------------------------------------------------------------------
            with allure.step("Check that the simulation identifier is parsed"):
                assert result.simulation_id == "99", "Identifier extraction failed"
                allure.attach(
                    result.simulation_id,
                    name="extracted_simulation_id",
                    attachment_type=allure.attachment_type.TEXT,
                )

            # ------------------------------------------------------------------
            # 5️⃣  Validate inner ``from_hdf5`` calls
            # ------------------------------------------------------------------
            with allure.step("Assert each child ``from_hdf5`` was invoked once"):
                cfg_from.assert_called_once_with(mock_cfg_grp)
                params_from.assert_called_once_with(mock_params_grp)
                stim_from.assert_called_once_with(mock_stim_grp)
                spike_from.assert_called_once_with(mock_spike_grp)

                # Attach a readable call‑summary
                calls_summary = "\n".join(
                    [
                        f"Configuration.from_hdf5 → {cfg_from.call_args}",
                        f"SimulationParameters.from_hdf5 → {params_from.call_args}",
                        f"StimuliGroup.from_hdf5 → {stim_from.call_args}",
                        f"SpikeMap.from_hdf5 → {spike_from.call_args}",
                    ]
                )
                allure.attach(
                    calls_summary,
                    name="inner_from_hdf5_calls",
                    attachment_type=allure.attachment_type.TEXT,
                )

            # ------------------------------------------------------------------
            # 6️⃣  Final sanity checks on the returned object
            # ------------------------------------------------------------------
            with allure.step("Validate the returned ``Simulation`` instance"):
                assert isinstance(result, Simulation), "Result is not a Simulation"
                assert isinstance(result.configuration, Configuration)
                assert isinstance(result.parameters, SimulationParameters)
                assert isinstance(result.stimulus, StimuliGroup)
                assert isinstance(result.output_spike, SpikeMap)

                # Full object dump for debugging purposes
                allure.attach(
                    repr(result),
                    name="Simulation_repr",
                    attachment_type=allure.attachment_type.TEXT,
                )
