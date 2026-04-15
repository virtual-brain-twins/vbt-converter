# Virtual Brain‑Twin Converter  

> *Standardising simulation output for the Virtual Brain‑Twin ecosystem*  

[![GitLab pipeline status](https://gitlab.com/your‑group/virtual‑brain‑twin‑converter/badges/main/pipeline.svg)](https://gitlab.com/your‑group/virtual‑brain‑twin‑converter/-/pipelines)


### Background  

Simulations of large‑scale cortical micro‑circuits generate many heterogeneous artefacts:  

* `config.yaml` – model configuration  
* `spike_times/*.npz` – per‑layer spike raster  
* `record_info/record_neurons.data` – mapping population → neuron count  
* `metrics/avg_firing_rates.csv` – population‑wise firing statistics  
* `random_input_*.npz` – piece‑wise constant stimulus files  

Because each research group may name files differently or store them in slightly different hierarchies, downstream analyses quickly become brittle.  

**`convert_direc_to_hdf.py`** ingests a *directory* that follows the above layout (or a subset of it) and writes a **single, self‑describing HDF5 file** that conforms to a community‑agreed schema (see `docs/data_schema.md`).  

The resulting HDF5 can be read directly by Python, MATLAB, Julia, or any language that supports the HDF5 API – the data are ready for further processing, archiving or sharing.

---