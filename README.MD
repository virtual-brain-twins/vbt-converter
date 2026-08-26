# vbt‑converter

**Virtual‑Brain‑Twin (VBT) data‑conversion utilities** – This is a framework that reads the raw protocol results,
potential consisting out of multiple individual simulations, and converts it into a standardized format. As standard
that can then be processed by downstream analysis pipelines.

The library is pure‑Python, depends only on a small scientific stack, and is organized as a reusable package that can be
vendored as a Git sub‑module.

---  

## Table of contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Add the repository as a sub‑module](#add-as-submodule)
4. [Installation (Python environment)](#installation)
5. [Typical workflow](#workflow)
    * [Step 1 – Build a master HDF5 bundle](#step1-build-bundle)
    * [Step 2 – Export a protocol‑specific artefact](#step2-export)
6. [HDF5 bundle layout (schema 1.0)](#bundle-layout)
    * [Common root structure](#common-root)
    * [AMU protocol specifics](#amu-layout)
    * [Basal‑Ganglia protocol specifics](#bg-layout)
7. [Extending the package (new protocol)](#extending)
8. [Design Documentation](#design-doc)
9. [Creation or update of MinIO cache entries](#minio-cache)
10. [Running the test‑suite (optional)](#testing)
11. [License & contributing](#license)

---  

<a name="overview"></a>

## 1. Overview

| Script                                                   | Purpose                                                                                                                                                                  | Main entry points                                                            |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `converters/generic_converter/convert_any_to_hdf5.py`    | Walk an **AMU directory tree** **or** a **Basal‑Ganglia CSV** and write a *single* master HDF5 file (`master.h5`) along with the raw_bundle.zip.                         | `python -m converters.generic_converter.convert_any_to_hdf5 …`               |
| `converters/generic_converter/convert_hdf5_to_format.py` | Read the master HDF5 and, **per protocol**, produce the downstream artefacts (`*.dsr.npz` for AMU, `*.npz` for Basal‑Ganglia).                                           | `python -m converters.generic_converter.convert_hdf5_to_format … <protocol>` |
| `converters/factory/*_factory.py`                        | Low‑level factories that assemble a :class:`~converters.schema.master_bundle.MasterBundle` from raw input.                                                               |
| `converters/exporters/*_exporter.py`                     | Concrete exporters that implement the per‑simulation conversion logic for a given protocol.                                                                              |
| `converters/schema/*`                                    | Dataclasses that capture the new HDF5 schema (`Simulation`, `SpikeMap`, `StimuliGroup`, …) and implement :class:`~converters.schema.hdf5_serializable.HDF5Serializable`. |
| `converters/utils/*`                                     | Helper utilities – logging, dataset creation, ZIP archiving, CSV handling, **compound‑attribute handling with optional units**, etc.                                     |

All scripts use only the packages listed in `requirements.txt` (NumPy, h5py, PyYAML, tqdm, …).

---  

<a name="prerequisites"></a>

## 2. Prerequisites

| Requirement      | Details                                                                                                                                                                                                                                                                                                                                                                       |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Python**       | 3.9+ (the library is type‑annotated and tested on 3.9 – 3.12).                                                                                                                                                                                                                                                                                                                |
| **Dependencies** | Install from `requirements.txt`.                                                                                                                                                                                                                                                                                                                                              |
| **Input data**   | <ul><li>**AMU** – a root directory where each immediate sub‑folder is a separate simulation run containing `spike_times/*.npz`, `config.yaml`, `record_info/record_neurons.data`, …</li><li>**Basal‑Ganglia** – a *tab‑separated* CSV generated by the original Snudda pipeline; the CSV references per‑run stimulus JSON files and Snudda output‑spike HDF5 files.</li></ul> |
| **Optional**     | `zstandard` Python package – enables ZSTD compression when creating the optional raw‑data ZIP archive (fallback to DEFLATE is automatic).                                                                                                                                                                                                                                     |

---  

<a name="add-as-submodule"></a>

## 3. Add the repository as a sub‑module

If you have an existing analysis repository, vendor the converter as a sub‑module:

```bash
# Inside your main repository
git submodule add https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter.git external/vbt-converter
git submodule update --init --recursive   # fetch the sub‑module contents
```

After cloning you can execute the CLI scripts directly:

```bash
python external/vbt-converter/converters/generic_converter/convert_any_to_hdf5.py …
python external/vbt-converter/converters/generic_converter/convert_hdf5_to_format.py …
```

*Tip*: Pin the sub‑module to a stable tag/branch (e.g. `-b main`) and update with `git submodule update --remote` when a
new version is released.

---  

<a name="installation"></a>

## 4. Installation (Python environment)

From the sub‑module root:

```bash
# (optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install the required packages
pip install -r requirements.txt
```

The helper script `setup_environment.sh` (used by the CI) performs the same steps.

---  

<a name="workflow"></a>

## 5. Typical workflow

The conversion is a **two‑stage pipeline**:

<a name="step1-build-bundle"></a>

### Step 1 – Build the master HDF5 bundle

```bash
# ---- AMU --------------------------------------------------------------
python -m converters.generic_converter.convert_any_to_hdf5 \
    -i /path/to/amu_root                \
    -o /tmp/master.h5                    \
    --protocol amu                       \
    --log-level INFO                     # optional (DEBUG, INFO, …)
    # optional arguments introduced in v1.0
    --protocol-version 1.0.0 \
    --protocol-info-path /path/to/protocol_info.pdf

# ---- Basal‑Ganglia ----------------------------------------------------
python -m converters.generic_converter.convert_any_to_hdf5 \
    -i /path/to/run_list.csv             \
    -o /tmp/master.h5                    \
    --protocol basal-ganglia             \
    --log-level INFO \
    --protocol-version 1.0.0 \
    --protocol-info-path /path/to/protocol_info.pdf
```

* **`--protocol`** – either `amu` or `basal-ganglia` (exact strings used by the CLI).
* **`--no-raw-bundle`** – disables the optional embedding of the original raw data as a ZIP archive (`raw_bundle.zip`).
* **`--protocol-version`** – version string that will be stored in the *Protocol* group of the bundle.
* **`--protocol-info-path`** – path to a PDF (or any binary) containing the protocol description; if omitted the script
  looks for a file named `protocol_info.pdf` inside the input directory.
* The script creates the output directory if it does not exist, writes the attribute `format_version = "1.0"` to the
  root, and builds the *StandardFormat* group (`/StandardFormat`) according to the new schema (see below).

<a name="step2-export"></a>

### Step 2 – Export a protocol‑specific artefact

```bash
# ---- AMU (produces a single compressed .npz in DS‑R format) ----------
python -m converters.generic_converter.convert_hdf5_to_format \
    -i /tmp/master.h5 \
    -o ./out_amu      \
    -b 0.001          \   # bin size (seconds) – required by AMUExporter
    amu

# ---- Basal‑Ganglia (one .npz per simulation) -------------------------
python -m converters.generic_converter.convert_hdf5_to_format \
    -i /tmp/master.h5 \
    -o ./out_bg \
    basal-ganglia
```

* The **protocol name** is the *sub‑command* (`amu` or `basal-ganglia`).
* Protocol‑specific arguments are added automatically – e.g. `--bin-size` (`-b`) for the AMU exporter.
* Output files are written into the directory given with `-o`.
* A log file `<script_name>.log` is created in the directory defined by the environment variable `VBT_JOB_LOGS`
  (default: the current working directory) and a copy of all messages is also sent to `stderr`.

---  

<a name="bundle-layout"></a>

## 6. HDF5 bundle layout (schema 1.0)

The master file follows a **well‑defined hierarchy** that is versioned (`format_version = "1.0"`). All protocol‑specific
data lives under the top‑level group **`StandardFormat`**; optional raw data, protocol information and
external‑dependency information are stored in separate groups.

```
├─ format_version (attr) = "1.0"
├─ StandardFormat/
│   ├─ Global/                ← global experiment metadata (attributes)
│   ├─ Configuration/         ← optional simulation‑wide configuration (compound attributes)
│   └─ Simulations/
│       ├─ Sim_<id_1>/        ← first simulation
│       │   ├─ Configuration/ ← per‑simulation configuration (compound KV container)
│       │   ├─ Parameters/    ← scalar key/value pairs (SimulationParameters)
│       │   ├─ Stimuli/       ← StimuliGroup (population‑wise piece‑wise constants)
│       │   └─ OutputSpikes/  ← SpikeMap (Populations → Neurons → times/indices)
│       ├─ Sim_<id_2>/ …
│       └─ …
├─ RawBundle/                 ← optional; contains the raw zip archive *as a dataset* (`raw_bundle.zip`)
├─ Protocol/                  ← optional; stores protocol version and binary documentation (PDF, etc.)
└─ ExternalDependencies/      ← **removed** – this group is no longer part of the schema
```

### Common root structure

| Path                                   | Meaning                                                                                                                                                                                                                                       |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/format_version`                      | String attribute – current schema version (`"1.0"`).                                                                                                                                                                                          |
| `/StandardFormat/Global`               | Attributes holding free‑form global metadata (`experiment_info`, **`parameter_simulation_mapping`**, arbitrary key/value pairs).                                                                                                              |
| `/StandardFormat/Configuration`        | **New (v1.0)** – a group that contains arbitrary key‑value pairs stored as **compound attributes** (value + optional unit). Typically used for global configuration that applies to all simulations.                                          |
| `/StandardFormat/Simulations/Sim_<id>` | One group per simulation. The `<id>` is taken from the source (CSV `sim_info_file` stem for Basal‑Ganglia, directory name for AMU).                                                                                                           |
| `/RawBundle`                           | If `--no-raw-bundle` is **not** used, a **dataset** `raw_bundle.zip` stores the raw zip archive *inside* the HDF5 file (compressed chunk‑wise) together with a SHA‑256 checksum attribute.                                                    |
| `/Protocol`                            | **New (v1.0)** – stores protocol meta‑information. Contains the **attribute** `Version` (UTF‑8 string) and a **dataset** `Content` holding the binary protocol description (e.g. a PDF). The `Content` dataset **must be stored compressed**. |

### AMU protocol specifics (`/StandardFormat/Simulations/Sim_<id>`)

| Sub‑group                                                                                                                                                                                     | Content                                                                                                                     | HDF5 representation                                                                                                                                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Parameters`                                                                                                                                                                                  | All scalar simulation parameters – the three scaling factors plus any additional metadata (e.g. `t_sim`, `record_neurons`). | Stored as **attributes** on the group (`SimulationParameters`).                                                                                                                                                                     |
| `Stimuli`                                                                                                                                                                                     | A *single* uniform stimulus (same for every population) – piece‑wise‑constant.                                              | One `StimuliGroup` named `"default"` for each population; each contains a `Stimulus` (datasets `Limits`, `Values`, `MaxValue`).                                                                                                     |
| `OutputSpikes`                                                                                                                                                                                | Spike trains for all populations defined in :data:`AMU_POPULATIONS` (extracted from the sweep directory data).              | `SpikeMap → Population → Neuron → times` (datasets `times`). The optional `indices` dataset is present when the original NPZ included an `indices` array; otherwise a dummy neuron per spike is created during bundle construction. |
| **Optional** `record_neurons` (binary payload) is stored as an attribute inside the `Configuration` group by the factory – later accessed through `Configuration.kv_pairs["record_neurons"]`. |

The **AMU exporter** (`AMUExporter`) reads the above sections, discretises spikes with the user‑provided bin size,
discretises the stimulus, normalises firing‑rates, stimulus and parameters, and writes a **single compressed NumPy
archive** named `amu.npz` (default output suffix `.npz`) with the keys required by the downstream DS‑R pipeline:

```
x          → normalised firing‑rate matrix (n_points, n_time_bins, n_populations)
u          → normalised stimulus (n_points, n_time_bins, 1)
theta      → normalised parameter matrix (n_points, n_params)
names_x    → list of population names (AMU_POPULATIONS)
names_theta→ list of short parameter names (["numneur", "synstr", "nmdaeff"])
dt         → bin size (seconds)
mu_x, sd_x → mean / std used for normalisation of x
mu_u, sd_u → mean / std used for normalisation of u
mu_theta, sd_theta → mean / std used for normalisation of theta
```

### Basal‑Ganglia protocol specifics (`/StandardFormat/Simulations/Sim_<id>`)

| Sub‑group                                      | Content                                                                                          | HDF5 representation                                                                                                                                                               |
|------------------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Parameters`                                   | Simulation meta‑data (network state, duration, DA level, input conductance, …).                  | Stored as attributes (string‑encoded or JSON‑encoded) in the `Parameters` group.                                                                                                  |
| `Stimuli`                                      | Full stimulus definition – per‑population piece‑wise constant or generic sub‑stimuli.            | One `StimuliGroup` per population; each sub‑stimulus stores a `Stimulus` (datasets `Limits`, `Values`, `MaxValue`) and any scalar attributes (e.g. generator type, cluster size). |
| `OutputSpikes`                                 | Direct copy of the original Snudda spike‑output HDF5 hierarchy (`meta_data`, `time`, `neurons`). | The factory reads the classic Snudda layout and builds a `SpikeMap` comprised of `Population` and `Neuron` objects (datasets `times`).                                            |
| **Optional** `RawBundle` behaves like for AMU. |

The **Basal‑Ganglia exporter** (`BasalGangliaExporter`) writes **one `.npz` per simulation** containing:

- `metadata` – a JSON‑encoded string of all scalar meta‑data (attributes from `Parameters`).
- `<pop>_times` – raw spike timestamps for each population.
- `<pop>_indices` – optional neuron IDs (present only if the original data stored them).

---  

<a name="extending"></a>

## 7. Extending the package (new protocol)

Adding a new protocol is straightforward and does not require touching the existing code. Add a new factory and
exporter, register them in the dictionaries `PROTOCOL_BUILDERS` and `PROTOCOL_EXPORTERS`, and the generic driver will
automatically add a new sub‑command to the CLI.

- Create a new factory that produces a :class:`~converters.schema.master_bundle.MasterBundle` from the raw data. Follow
  the pattern of `AMUFactory` or `BasalGangliaFactory`.
- Register the factory in `PROTOCOL_BUILDERS` (see `protocol_registry.py`).

Adding support for a new downstream format only requires a few well‑defined steps:

1. **Create a subclass of** :class:`converters.exporters.protocol_exporter.ProtocolExporter`.
    * Implement
        * `extract_one_simulation(self, sim_id, sim_group)` → return a plain `dict` (or any structure) that contains the
          data you need for the final artefact.
        * `export_one_simulation(self, sim_id, sim_group)` → write the artefact to ``self.out_dir`` (use ``self.log``
          for messaging).
    * Optionally override ``protocol_name`` and ``output_suffix``.
    * Add a ``@classmethod configure_parser(cls, parser)`` that declares any extra CLI arguments (e.g. `--threshold`).

2. **Register the exporter** in `converters/generic_converter/convert_hdf5_to_format.py` by adding an entry to the
   module‑level dictionary `PROTOCOL_EXPORTERS`:

   ```python
   from converters.exporters.my_new_exporter import MyNewExporter
   PROTOCOL_EXPORTERS["my-protocol"] = MyNewExporter
   ```

3. **Add a builder** if the new protocol also needs to be *created* from raw data.
    * Implement a factory function that returns a :class:`~converters.schema.master_bundle.MasterBundle`.
    * Expose it via a concrete subclass of :class:`converters.factory.protocol_registry.ProtocolBuilder` and add the
      builder to `PROTOCOL_BUILDERS`.

The generic driver automatically:

* Adds a sub‑command with the name you used in `PROTOCOL_EXPORTERS`.
* Injects only the arguments that the exporter’s ``__init__`` signature expects (so adding new CLI options never breaks
  existing code).
* Handles logging, input validation and HDF5 opening.

---  

<a name="design-doc"></a>

## 8. Design Documentation

Take a look at the generated UML diagrams, data‑flow diagrams and sequence diagrams in `docs/` for a visual overview of
the architecture, data flow and execution sequences. See [docs/Readme.md](docs/Readme.md) for more details.

The detailed specification of the HDF5 schema is in [docs/specification/Virtual Brain Twin Inter Protocol Data Standard Specification and HDF5 Specification_v1.0.docx](docs/specification/Virtual%20Brain%20Twin%20Inter%20Protocol%20Data%20Standard%20Specification%20and%20HDF5%20Specification_v1.0.docx).

---  

<a name="minio-cache"></a>

## 9. Creation or update of MinIO cache entries

If you need to create a new cache entry **or** update an existing one that is used by the integration tests, run the job
**`update_or_create_new_cache_job`** manually in a pipeline.  
Set the following CI variables when you trigger the job:

| Variable         | Description                                                                 | Example value                                                                           |
|------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `CI_CACHE_KEY`   | The cache key that GitLab uses for the MinIO bucket.                        | `vbt-converter-test-cache` (AMU cache key) / `vbt-kth-bg-protocol-cache` (BG cache key) |
| `CI_CACHE_DIR`   | Directory inside the job where the cache will be written.                   | `${CI_PROJECT_DIR}/tests/test_data/`                                                    |
| `MINIO_SRC_PATH` | Path (relative to the bucket *ci‑temp‑cache*) of the new ground‑truth file. | `VBT_AMU_Protocol/ground_truth/master_v1.h5`                                            |
| `LOCAL_DEST_DIR` | Full destination path inside the job where the file will be copied to.      | `${CI_PROJECT_DIR}/tests/test_data/ground_truth/test_convert_output.h5`                 |

### What the job does? (Example case: AMU protocol)

The job copies the object located at https://vbt-minio.fz-juelich.de/console/browser/ci-temp-cache/$MINIO_SRC_PATH to
the local cache location defined by `LOCAL_DEST_DIR`. When the job finishes successfully, the ground‑truth file in the
cache (`tests/test_data/ground_truth/test_convert_output.h5`) is refreshed, and subsequent integration tests will use
the updated data.

### How to run it

1. Create a new pipeline (e.g., from **CI/CD → Pipelines → Run pipeline**).
2. In the pipeline UI locate the job **`update_or_create_new_cache_job`** and click **▶ Play**.
3. In the “Run job” dialog, add the four variables above with the values you need.
4. Start the job and wait for it to complete.

After the job succeeds, the MinIO cache entry identified by `CI_CACHE_KEY` contains the new data and is ready for the
integration tests.

---  

<a name="testing"></a>

## 10. Running the test‑suite (optional)

The library includes a test‑suite (unit and integration) that can be run with `pytest`. The tests are located in
`tests/` and cover the schema, factories, exporters and CLI scripts. The test can be run via GitLab CI/CD or locally. To
run the tests locally, ensure you have `pytest` installed in your environment and execute:

```bash
pytest tests/
```

---  

<a name="license"></a>

## 11. License & contributing

*The license is still to be decided.*  
Contributions are welcome:

* Fork the repository.
* Create a feature branch.
* Ensure the test suite passes (`pytest`).
* Open a merge request against the `main` branch.