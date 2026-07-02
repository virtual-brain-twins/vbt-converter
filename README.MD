# vbt‑converter  

**Virtual‑Brain‑Twin (VBT) data‑conversion utilities** – a tiny library that  
*collects simulation results* into a **master HDF5 bundle** and then *exports* that bundle to the protocol‑specific format required by downstream analysis pipelines (currently **AMU** and **Basal‑Ganglia**).

---

## Table of contents
1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Add the repository as a submodule](#add-as‑submodule)  
4. [Installation (Python environment)](#installation)  
5. [Typical workflow](#workflow)  
   - [Step 1 – Build the master HDF5](#step1-build-master-hdf5)  
   - [Step 2 – Export to a protocol format](#step2-export-protocol)  
6. [HDF5 bundle layout](#hdf5‑layout)  
   - [AMU protocol](#amu‑layout)  
   - [Basal‑Ganglia protocol](#basal‑ganglia‑layout)  
7. [Extending the exporter (new protocol)](#extending)  
8. [Running the test‑suite (optional)](#testing)  
9. [License & contributing](#license)  

---

<a name="overview"></a>
## 1. Overview  

| Script | Purpose | Main entry points |
|--------|---------|-------------------|
| `converters/generic_converter/convert_any_to_hdf5.py` | Walk a **Basal‑Ganglia CSV** or an **AMU directory tree** and write a *single* master HDF5 file (`master.h5`). | `python convert_any_to_hdf5.py …` |
| `converters/generic_converter/convert_hdf5_to_format.py` | Read the master HDF5 and, **per protocol**, produce the downstream artefacts (`*.basal.npz` or `*.dsr.npz`). | `python convert_hdf5_to_format.py -i master.h5 -o out_dir amu` (or `basal-ganglia`) |

Both scripts are completely self‑contained – they only rely on the Python packages listed in `requirements.txt` (NumPy, h5py, pandas, PyYAML, tqdm, Allure‑pytest‑plugin, …).

---

<a name="prerequisites"></a>
## 2. Prerequisites  

* Python 3.9 + (the CI uses the `venv` flavour).  
* The packages in `requirements.txt`.  
* For **Basal‑Ganglia** the input CSV must follow the VBT reference layout (see `bg_protocol_helper.py`).  
* For **AMU** each simulation directory must contain the files produced by the original AMU pipeline (`spike_times/*.npz`, `record_info/record_neurons.data`, `metrics/avg_firing_rates.csv`, etc.).

---

<a name="add-as-submodule"></a>
## 3. Add the repository as a submodule  

If you already have a Git project that needs the converters, add it as a submodule:

```bash
# Inside your main repository
git submodule add https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter.git external/vbt-converter
git submodule update --init --recursive   # fetch the submodule contents
```

After the submodule is cloned you can call the scripts directly:

```bash
python external/vbt-converter/converters/generic_converter/convert_any_to_hdf5.py …
python external/vbt-converter/converters/generic_converter/convert_hdf5_to_format.py …
```

*Tip*: keep the submodule on a stable tag/branch (e.g. `-b main`) and update it with `git submodule update --remote` when a new version is released.

---

<a name="installation"></a>
## 4. Installation (Python environment)

From the submodule root:

```bash
# 1) Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install the library dependencies
pip install -r requirements.txt
```

The `setup_environment.sh` script used by the CI performs the same steps automatically.

---

<a name="workflow"></a>
## 5. Typical workflow  

The conversion is a **two‑step pipeline**:

<a name="step1-build-master-hdf5"></a>
### Step 1 – Build the master HDF5  

```bash
# AMU directory tree
python converters/generic_converter/convert_any_to_hdf5.py \
    -i /path/to/amu_root \
    -o /tmp/master.h5 \
    --protocol amu \
    --log-level INFO

# Basal‑Ganglia CSV + simulation root
python converters/generic_converter/convert_any_to_hdf5.py \
    -i /path/to/run_list.csv \
    -o /tmp/master.h5 \
    --protocol basal \
    --log-level INFO
```

* `--no‑raw‑bundle` disables the optional zipped copy of the original data.  
* `--log-level` can be `DEBUG`, `INFO`, …  

The script creates the output directory if missing, writes a global attribute `format_version = "1.0"` and, for each discovered simulation, creates a `sim_<id>` group containing the various *sections* (see [bundle layout](#hdf5‑layout)).

<a name="step2-export-protocol"></a>
### Step 2 – Export the protocol‑specific artefacts  

```bash
# Export AMU DS‑R .npz files
python converters/generic_converter/convert_hdf5_to_format.py \
    -i /tmp/master.h5 \
    -o ./out_amu \
    -b 0.001      # bin size (seconds) required by the AMU exporter
    amu

# Export Basal‑Ganglia per‑simulation .npz files
python converters/generic_converter/convert_hdf5_to_format.py \
    -i /tmp/master.h5 \
    -o ./out_bg \
    basal-ganglia
```

The command line follows the conventional pattern:

```
convert_hdf5_to_format.py -i <master.h5> -o <out_dir> [--log-level <lvl>] <protocol>
```

The **protocol name** is the sub‑command (`amu` or `basal-ganglia`).  
Additional protocol‑specific arguments are added automatically (e.g. `--bin-size` for AMU).

---

<a name="hdf5-layout"></a>
## 6. HDF5 bundle layout  

The master file is a **flat collection of groups** – one per simulation – under the root.  
All groups share the same *section* names, but the concrete layout depends on the protocol.

```
/
├─ format_version (attr) = "1.0"
├─ sim_<id_1>/
│   ├─ metadata/
│   ├─ spike_trains/          (AMU)   OR   output_spikes/ (Basal)
│   ├─ record_neurons/        (AMU only)
│   ├─ metrics/               (AMU only)
│   ├─ stimulus/              (both)
│   ├─ logs/                  (Basal only)
│   └─ raw_bundle/            (optional zip)
└─ sim_<id_N>/ …
```

<a name="amu-layout"></a>
### AMU protocol  

| Section (group) | Content | HDF5 representation |
|-----------------|---------|----------------------|
| `metadata` | All scalar simulation parameters (e.g. `scale_num_neurons`, `t_sim`) | Attributes on the group |
| `spike_trains` | One sub‑group per population (`L2/3E`, `L4PV`, …) containing `indices` (optional) and `times` datasets | `float64` (seconds) and `int64` (neuron ids) |
| `record_neurons` | Mapping `label → int` (e.g. `n_cells`) | Structured dataset `record_neurons` with fields `label` (UTF‑8) and `value` (`int64`) |
| `metrics` | Average firing rates per population (CSV in the reference pipeline) | Structured dataset `Avg Firing Rate Hz` (`label`, `value`) |
| `stimulus` | Piece‑wise‑constant external stimulus | Attributes: `type = "piecewise_constant"`; Datasets `limits` (int ms) and `values` (float array) |
| `raw_bundle` (optional) | Zip archive of the original simulation directory | Binary blob stored via `DataSection.store_external_path` (external HDF5 link if the target is HDF5). |

All datasets are **compressed** where appropriate (gzip → level 6).  
The exporter (`AMUExporter`) reads the `metadata`, `spike_trains`, `record_neurons`, and `stimulus` sections to compute firing‑rate matrices, discretise the stimulus, normalise the data and finally write a single compressed NumPy archive named `amu.dsr.npz`.

<a name="basal-ganglia-layout"></a>
### Basal‑Ganglia protocol  

| Section (group) | Content | HDF5 representation |
|-----------------|---------|----------------------|
| `metadata` | Simulation meta‑data (network state, duration, DA level, …) | Attributes on the group |
| `output_spikes` | Direct copy of the original Snudda `output_spikes.h5` file hierarchy (`/neurons/<gid>/spikes`, `/time`, `/meta_data/…`) | Copied *as‑is* using `h5py.Group.copy` (lazy, no data copy). |
| `aux_files` | Three standard log files (`error.log`, `info.log`, `warning.log`) | Binary blobs (small files) or external links for larger files. |
| `raw_bundle` (optional) | Zip archive of the whole simulation directory | Same mechanism as for AMU. |

---

<a name="extending"></a>
## 7. Extending the exporter (new protocol)

Adding a custom protocol is straightforward:

1. **Create a subclass** of `converters/exporters/protocol_exporter.py.ProtocolExporter`.  
2. Implement `extract_one_simulation` (logic that returns a dictionary of NumPy arrays) and `export_one_simulation` (writes the final artefact).  
3. Register the class in `convert_hdf5_to_format.py` by adding a new entry to the `PROTOCOL_EXPORTERS` dictionary.  
4. Optionally add a `configure_parser` class‑method to expose protocol‑specific CLI arguments.

The generic driver automatically discovers the sub‑command name and forwards all non‑global arguments that match the exporter’s `__init__` signature.

---

<a name="testing"></a>
## 8. Running the test‑suite (optional)

The repository ships a **pytest‑based integration test harness** that validates the whole CI pipeline (see `ci/template_repo/vbt_protocol_template_tests`).  
To run it locally:

```bash
# Activate the venv where the requirements are installed
source .venv/bin/activate

# The test harness expects the environment variables used by the CI.
export VBT_ENTRY_SCRIPT=$(pwd)/converters/generic_converter/convert_any_to_hdf5.py
export VBT_ENTRY_ARGS="--protocol amu"
export VBT_INPUT_PATH=/path/to/amu_root
export VBT_OUTPUT_PATH=$(pwd)/tmp_out
export VBT_RUN_TIMEOUT=300

# Run the generic test suite
pytest -s ci/template_repo/vbt_protocol_template_tests
```

All artefacts are attached to an Allure report (`allure-results`) which can be generated locally with `allure serve allure-results`.

---

<a name="license"></a>
## 9. License & contributing  

* To be decided.  
* Contributions are welcome – please open a merge request against the `main` branch and ensure that the test suite passes (`pytest`).  

---