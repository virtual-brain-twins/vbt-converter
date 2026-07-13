Install the dependencies for the documentation generation:
- sudo apt update && sudo apt install -y graphviz plantuml
- pip install pylint graphviz

Generate the documentation:

Generate the UML diagrams for the converters module:
- cd <path_to_your_project>/docs
- pyreverse -o html -p vbt-converter converters/
- pyreverse -o svg -p vbt-converter converters/

Generate the UML diagrams for the core module:

Data flow diagrams:
- cd <path_to_your_project>/docs
High level DFD:
- plantuml -tsvg dfd_level_0.puml
Detailed DFD:
- plantuml -tsvg dfd_level_1.puml

Sequence diagrams:
SD convert_any_to_hdf5.py
- plantuml -tsvg sd_directory_to_hdf5.puml
SD convert_hdf5_to_format.py
- plantuml -tsvg sd_hdf5_to_protocol_format.puml

Component‑Interaction Overview:
- plantuml -tsvg component_interactions.puml