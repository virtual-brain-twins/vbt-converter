# Design Documentation

This **docs/Readme.md** is a quick‑start guide to the generated documentation for **vbt‑converter**.  
It provides a visual overview of the project’s architecture, data flow and execution sequences.

---

## High‑level Class UML diagrams

| Diagram      | Rendered view                                                                    |
|--------------|----------------------------------------------------------------------------------|
| **Classes**  | <img src="./classes_vbt-converter.svg" alt="Classes UML diagram" width="800"/>   |
| **Packages** | <img src="./packages_vbt-converter.svg" alt="Packages UML diagram" width="800"/> |

---

## Data‑flow diagrams (DFD)

| Level                    | Diagram                                                      |
|--------------------------|--------------------------------------------------------------|
| **Level 0 – High‑level** | <img src="./dfd_level_0.svg" alt="DFD level 0" width="800"/> |
| **Level 1 – Detailed**   | <img src="./dfd_level_1.svg" alt="DFD level 1" width="800"/> |

These diagrams illustrate how data moves through the converter, from raw input directories to the final
protocol‑specific exports.

---

## Sequence diagrams

| Use‑case                                      | Diagram                                                                                                   |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Directory → HDF5** (parameter‑sweep import) | <img src="./sd_directory_to_hdf5.svg" alt="Sequence diagram: directory to HDF5" width="800"/>             |
| **HDF5 → Protocol format** (export)           | <img src="./sd_hdf5_to_protocol_format.svg" alt="Sequence diagram: HDF5 to protocol format" width="800"/> |

These sequences detail the step‑by‑step interaction of the CLI commands and the internal conversion pipeline.

---

## Component‑Interaction Overview

| Diagram                    | Rendered view                                                                              |
|----------------------------|--------------------------------------------------------------------------------------------|
| **Component interactions** | <img src="./component_interactions.svg" alt="Component interaction overview" width="800"/> |

The component‑interaction view ties together the builders, exporters, factories and helper modules that make up the
whole system.