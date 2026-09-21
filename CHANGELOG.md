# Changelog

All notable changes to this project will be documented in this file.
## [1.1.0](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/compare/v1.0.0...v1.1.0) (2026-09-21)
### Release Summary

*Release engineering & tooling*  
- Added **`update‑changelog.sh`** – a dedicated script that reads `.release‑summary.txt`, extracts the version from `package.json`, safely escapes special characters and injects the summary into `CHANGELOG.md` using a robust `awk`‑based approach.  
- Re‑implemented the `release:summary` npm script to call the new script; removed the obsolete `STDVER` variable and simplified version‑tag handling (creates a clean `v1.0.0` tag on first release).  
- Updated release checklist, added a full changelog, release script, and package configuration to the repository.

*Documentation*  
- Split the README into **User‑Guide** and **Developer‑Guide** sections, improving navigation.  
- Added Markdown links to UML, data‑flow, and sequence diagrams; URL‑encoded HDF5 spec reference.  
- Clarified installation steps (sub‑module cloning, environment setup, version pinning) and expanded quick‑start examples with optional metadata flags and raw‑bundle handling.  
- Introduced a “Verifying an HDF5 file” snippet and refreshed licensing, contribution guidelines, and release instructions.

*Metadata handling*  
- Introduced a configurable **`metadata`** argument for `build_kth_master_bundle` and `build_uva_master_bundle`.  
- Global metadata (author, creation date, software, version, UUID) now defaults to values from `MetaDataBuilder` and can be overridden via new `--meta-<key>` CLI options in `convert_any_to_hdf5.py`.  
- Updated CI to fetch the MinIO client from a configurable `${CI_MC_URL}`.

*Converters & simulation parameters*  
- Removed hard‑coded parameter lists; parameters are now derived dynamically from the configuration section.  
- Added `DEFAULT_UVA_PARAMETERS` and optional `parameters` arguments to both UVA and KTH bundle builders.  
- Extended `protocol_registry` to forward `**extra_kwargs` (e.g., `parameters`) to the appropriate builder.  
- New `--parameters` CLI flag allows users to specify a custom parameter set at runtime.  
- Implemented `keys()` / `iter_keys()` in `CompoundKVContainer` for case‑insensitive, sorted key retrieval.  
- Adjusted integration tests to use the new flag.

*Protocol renaming & CI updates*  
- Renamed all AMU / Basal‑Ganglia references to **UVA / KTH** across code, docs, CI pipelines, and Allure reports.  
- Updated cache key names, input‑path variables, and wrapper script references to match the new protocol names.  
- Fixed CI configuration for the KTH and UVA pipelines.

*Other*  
- Minor code formatting, doc‑string refinements, and log‑message updates for better readability.  

### New Features

* broaden changelog header parsing to support bracketed and linked versions ([0dd2780](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/0dd27802b7e22df3e91219d9efe9f487e10c0c1a))
* **converters:** make simulation parameters configurable via CLI and factory API ([f4ed85f](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/f4ed85f909fc8944308987668cd74091066c700a))

### Documentation

* **readme:** clarify design documentation links and update HDF5 spec reference ([a55757d](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/a55757db3d70e7df5727ed9a986ef67f85a0c985))
* **readme:** restructure README into user‑guide and developer‑guide sections ([dca51dd](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/dca51ddfe823251177f5f084e2996402dcbd7e2d))

### Code Refactoring

* group commit types into sections and broaden changelog header matching ([a12d446](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/a12d446b2098dcf047d53348b3544e72ad26213f))
* **metadata:** add configurable global metadata via CLI and propagate to factories ([fd330a3](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/fd330a35db31d7b796ec2256153b0e4e0f662595))

### Chores

* **release:** enhance release workflow and documentation ([2581a8c](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/2581a8c1737af9236343dc1eae53c9acf840e8b4))

## 1.0.0 (2026-09-17)

### Release Summary
Version 1.0.0 of the VBT converter including HDF5 standard (1.0)

The vbt-converter release introduces a pure-Python, reusable conversion framework for Virtual Brain Twin simulation data that standardizes heterogeneous protocol outputs into a versioned HDF5 master bundle using schema 1.0. The core of the release is the creation of a single master.h5 file with a clearly defined HDF5 hierarchy: a root format_version = "1.0" attribute, a top-level StandardFormat group containing global metadata, optional simulation-wide configuration stored as compound attributes, and per-simulation groups under Simulations/Sim_ that organize scalar parameters, population-wise stimuli, and spike-train output in a consistent structure. Protocol-specific raw inputs, such as UVA directory trees or KTH Snudda CSV references, are converted into this common layout through protocol factories, while optional raw data can be embedded as a compressed RawBundle dataset and protocol metadata, including version and binary documentation, can be stored in a dedicated Protocol group. The schema also removes the former ExternalDependencies group and establishes a stable basis for downstream export tools to generate protocol-specific artifacts, such as .npz files, from the standardized HDF5 bundle.

### New Features

* add CI job for sweep‑directory conversion to DSR and enhance wrapper to produce NPZ with intermediate HDF5 ([66c1411](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/66c1411bb5e4e306cb339f19fa73e07a769b9c06))
* add CI variable for template submodule path ([3edbb45](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/3edbb457b42de545b48974e6ca0bfe0633872326))
* add CI wrapper for VBT converter and simplify job configuration ([9ced5c3](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/9ced5c3554a0a12a9d8cec908dc7269ea40e0690))
* add configurable HDF5 artifact handling and HTML link generation ([8a2fd13](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/8a2fd13128e204472b73c78c0ba49b90b09ef4fe))
* add CSV‑based metadata handling and path relativization for bundle creation ([df73e7d](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/df73e7d6de66afd038b9cc6ed79d5746d3d4e643))
* add dedicated HTML templates for HDF5 and NPZ comparison reports ([513542c](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/513542c8c87e80c57fd38c40c9fc746d4b592e24))
* add dedicated KTH BG protocol CI pipeline with setup, test, and cleanup stages ([72c7382](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/72c73826ac54f7658a590e4d1259083bff144f4f))
* add dynamic population handling and strict master‑bundle validation ([391e409](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/391e4091031f682056117057c78ad7820e38df30))
* add dynamic test name suffix and fix HDF5 input path ([8d46ccb](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/8d46ccb72afbc0564bdd2e660ce7be25f89731a3))
* add fast parallel conversion from sweep directory to NPZ/HDF5 and improve existing converters ([fdcd31b](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/fdcd31becffe611a934d5f1ec4216989d4b9d99f))
* add full AMU & Basal‑Ganglia bundle support with new spike readers and sections ([906bb7a](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/906bb7a058413cf3a405f14b67394b4271adf8c1))
* add ground‑truth validation and manual cache‑update CI job ([04a5fd2](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/04a5fd2d342ba55ae8828d9a8c9aefcd7536b788))
* add HDF5 diff validation and HTML report generation in CI ([822120e](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/822120ef219b18694ed28a1b2ab9f86699f78d78))
* add HDF5‑to‑DSR conversion stage and improve CI testing ([0f2cadb](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/0f2cadbac558e2a576eeb330d5b25f79f783b9ef))
* add interactive HDF5 tree visualization and standardize exporter output suffix ([6bc0224](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/6bc0224a219a3aec28e7a364542480dd87b21e11))
* add interactive release summary and migrate to commit-and-tag-version ([effb3cb](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/effb3cbd1dcdefae8694664ea7125e2a76ca2767))
* add master HDF5 bundler and replace sweep‑to‑NPZ conversion script (Initial test commit, actual logic will be adapted later) ([0183492](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/0183492cf0606922f9ecbe0c6593cd3a751642a6))
* add NPZ comparison and unified HTML report ([5f1c22d](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/5f1c22dbc5c926c3800a1540eb1f4f4eea05ed47))
* add optional is_path_absolute flag to link spec handling ([a607aa5](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/a607aa5def284e2423f54cc417837b9fdab40bd1))
* add optional ZSTD compression with safe fallback ([b249b22](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/b249b22ea8995548e1a5dfd965a77dd4248a82e5))
* add publish_report stage and configure vbt_publish_job for integration pipelines ([cb338bc](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/cb338bc195249182e42792f4e9c72cf7a199edfd))
* Add reusable MinIO cache fetch template with configurable defaults ([75aac79](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/75aac79a4733a78ab8cdb092e0da5edcae807777))
* add structured logging and detailed h5diff status handling ([bfeb2b1](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/bfeb2b1fff064b84e6e7bb786b3f534033a64bc7))
* add unified HTML link helper and refactor wrappers to use it ([7ee34dc](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/7ee34dc0c937390a6672a25deb804dd5bbe9fa93))
* add validation and configurable paths to update_cache_job ([f6bc9be](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/f6bc9be189021e4ea186ce84b76e310f4abbbe2f))
* add VBT converter scripts, test suite, CI pipeline, and documentation ([bfb629d](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/bfb629d7eaca8648ab3c81d3a8b09f3a457d5d01))
* attach input sweep directory as CI artifact and improve Allure links ([5f675e4](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/5f675e40f4c998ffd9ff1cb6f0c3ed5cfca72264))
* clean up temporary zip file and add unit‑aware attribute formatting ([0321701](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/032170192adaed9ae17c0963a4bfc0d819810b39))
* enable cross‑pipeline artifact dependencies for VBT publish workflow ([0abbb4e](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/0abbb4ee9f99589ebbc5627f506cfac395c6b715))
* enhance HDF5 HTML viewer collapse behavior and correct protocol argument ([b8a8546](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/b8a85460f1c103bca334e08c9101f74243f10598))
* enhance logging and script organization for child artifact download ([d3b6445](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/d3b6445f5282b09173725f42a16afaa72fc20cce))
* enrich unit_tests CI job with Allure reporting metadata ([00171e5](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/00171e5350fd4f01f0fdc3aa852945989de05d0b))
* expand project dependencies for testing and CI ([63b1314](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/63b1314c659718ed4a8e37acc3c0dc58591870c2))
* extend link helper to support absolute URLs and update integration scripts ([0d2ea2a](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/0d2ea2abcb083734e6ca519136e8d211a0f80269))
* externalize raw bundles and large auxiliary files ([fcd6824](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/fcd68244fe4c914a1b5616c77c8c19c6f291ad04))
* implement full AMU bundle conversion, export pipeline, and CLI framework ([d4b65fe](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/d4b65fee49b5357fc5cf9e529752f59532282a38))
* implement VBT converter for HDF5 data handling and update CI configuration ([89f54a0](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/89f54a005bd6bebae17bb68e0094ad1705743a38))
* integrate child artifact retrieval into publish pipeline ([7d37416](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/7d37416909b2a8dcf6a7595c405db5321a64ffd0))
* overhaul conversion pipeline and adopt new HDF5 schema ([2d49a1a](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/2d49a1ab4941d827e1d32dc6b9bcc83cd779f873))
* refactor schema to use compound attributes with units and add protocol support ([9815594](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/9815594f2877f1b0905cb369b68abb1052de4ad8))
* rename cache job and document CI cache variables ([397f787](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/397f787c76c038184b8b9b6c4151e61d9c816528))
* restructure CI paths and expose converted NPZ artifact in validation report ([0c49946](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/0c49946fabd1867a70b1efb49dbf7198a5043c31))
* run conversion script in CI and refactor test import ([fb38099](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/fb380995e935ccdd2c8077b7fc7918c5a763c782))
* switch integration test to generic HDF5 converter ([31cf60d](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/31cf60d629ffd94ddef9046ebe53fb2d75dcc0a4))
* switch VBT entry script to PS‑to‑NPZ converter ([3e24955](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/3e249558449071c6ad5ee354059ef7cbf4cf569d))

### Bug Fixes

* add missing hdf5-tools package to CI dependencies ([8f22d8b](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/8f22d8bea44c3c8596a0e8b6e6b6ca9dd925c8d9))
* added missing input & output paths ([83f4fa3](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/83f4fa384b57acfb028c843b38b636aa611fd42f))
* adjust MinIO copy to respect destination type ([3924ab8](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/3924ab8f4973f3bf301cb8938ab6b40508c4a7c1))
* Align CLI flag naming with underlying converter script ([bbb1d92](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/bbb1d92495522c4307708fa763802904fdfd1a75))
* clean up update_cache_job CI script formatting ([d74bd68](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/d74bd6832ac9e8e4ace9a12b554a28673c52dcb4))
* correct argparse handling for protocol options ([6cc59ab](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/6cc59abb1ed5eed925fefc9622bdbf0a83a29d90))
* correct artifact link path ([6e6e8c7](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/6e6e8c750fc66c1b256217fea502068e00cfc24f))
* correct artifact URLs and add directory‑link support ([f0b4f4b](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/f0b4f4b1f20ad6e50db1f0c6e0003caa4f3f57ad))
* correct before_script syntax and clean up commented variables in CI config ([59e87f5](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/59e87f531c3fafe67c151954e055ce6f191d3e61))
* correct default bin size for HDF5‑to‑DSR conversion ([fb46aa9](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/fb46aa900210eeda06ef8adff09525d0323b3745))
* correct external link target and enhance documentation & utilities ([d841af9](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/d841af94dd4e97f24b48fef35b0e191772ce5d9a))
* correct HDF5 dataset keys for stimulus and spikes ([7a7bd65](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/7a7bd6510fcafb6f8541e3158a6440b26dd54b71))
* correct HDF5 metadata group and ensure lock initialization for intermediate files ([dfc2ec4](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/dfc2ec422cac6b2fd48e015cfcacc8a963eb9275))
* correct helper function imports in integration test wrappers ([620fbc1](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/620fbc1e95746b4121bf01b6bf99eae22fc302c8))
* correct HTML template path for VBT converter report ([205dbb6](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/205dbb632f590fa29fb23f55b55fe645b3e92567))
* correct input directory handling for VBT test job ([29a5539](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/29a55392c73fefd902dc6a71830cadc9e48df922))
* correct input directory paths for converter tests ([e50c72e](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/e50c72e975451e6541ab560eed29ab4b87dc7dba))
* correct VBT_SUBMODULE_PATH interpolation in CI pipeline configs ([e607613](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/e6076131846bc1455867a97534497ea2db1ed193))
* correct VBT_SUBMODULE_PATH interpolation in CI pipeline configs ([2439f72](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/2439f722d89d13bcb0680f26d3243cd86897c555))
* ensure destination folder creation handles file paths and missing directories ([a46b80d](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/a46b80ddc8ad12c90f43561707dfcd96c1c46ea3))
* export CI environment variables explicitly for setup and cleanup jobs ([0303664](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/0303664979b15c40ffb73c265b9717b241863a1e))
* improve CI variable validation in update_cache_job ([8ffd83b](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/8ffd83bd461b2a570b6be65f4fe4c400af982bc7))
* improve handling of parameter sweep directory movement in CI wrapper ([fff4c2a](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/fff4c2abe171e964f933106f3c5487dc172441a0))
* improve HDF5 conversion wrapper ([7a8c501](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/7a8c501cb0ad0002a1e0dec0e6545dbc2dc56d4b))
* migrate CI artifact handling to attachments directory and enhance link generation ([6338e4c](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/6338e4c107cc9712dc168d8f2ac3b876ff4e2ef9))
* pass parent pipeline ID to downstream integration pipelines ([170eaf7](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/170eaf7b84e81d4ce7ba6b42f527a760b2fee34f))
* remove stray ellipsis in CI echo command to resolve YAML parsing error ([82ef3be](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/82ef3beb2716e2887807f198ef7425e2c5803aed))
* resolve output HDF5 path and ensure destination directory exists ([61eb066](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/61eb0669d7835293716a434af3730e21cbc56a5a))
* separate CLI arguments in conversion wrapper ([be431c9](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/be431c96012cae1f4add4f4940e7c636253f15a1))
* tighten CI variable validation and clean up job defaults ([aea1365](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/aea13654bc51e2b673c0f58aff6849474453e5e2))
* update simulation serialization tests to match new HDF5 group names ([3d76f31](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/3d76f31ab0f05124b8a3be66e0ea606b06461ff7))

### Documentation

* add HTML conversion wrapper docs and expose them to Allure via CI variables ([8c91c81](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/8c91c813d48dfa917ca107c358e50f4e3a82ae06))
* add MinIO cache guide and tidy README formatting ([e5ccffe](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/e5ccffe4ceeac4492d547eda93a09d323153657d))
* clarify builder requirements and simplify AMUBuilder description ([de7545f](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/de7545f7af7ca64d29855498524f5503c04fef09))
* clarify library purpose and CI description ([71c220b](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/71c220bd8e37f56fd093d2bc62385f126be0fe58))
* clean up README formatting ([32b0212](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/32b02122c2ea1f0ca0f6717d451d470d69dddd1d))
* committed the first version of HDF5 schema specification. ([05e4ba2](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/05e4ba2e16a7ba5740f84e2062813164a72c5b0a))
* replace minimal README with comprehensive documentation and usage guide ([5d22414](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/5d224146a1ac3eb5346c18e4515a8f41b7f3cecb))
* simplify diagram rendering in README ([d20fe2d](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/d20fe2df6c24d0f6eddfce62736ca09dfd95e3cc))
* update VBT‑Converter Generic HDF5 Specification (v1.0) ([fb1c013](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/fb1c01340edfccf51b16b9da96ad285e470b5bdd))
* update VBT‑Converter Generic HDF5 Specification (v1.0) ([dc665cd](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/dc665cdd787f700455aeb4da2bb689cd7acad644))
* updated the uml diagrams as per the latest changes in schema ([ab2d66f](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/ab2d66f00a060516a7b3f3876c80a22a083952f2))
* updated the uml diagrams as perefactor: migrate compound attribute imports to helper and expand test suite ([d8bc35e](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/d8bc35eb0cee87e52558bc5733cf2528b94e6047))

### Code Refactoring

* align HDF5 schema naming and enrich documentation ([ff149cd](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/ff149cda9ba4afb502e653224f7320e0e7b33b4d))
* delegate unit test execution to shared runner via VBT_TEST_PATHS variable ([5eb9fa6](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/5eb9fa69e4eff634ea5facf9bab469dad4a1401e))
* enhance HDF5 conversion robustness and extend CI with unit‑test job ([be4cb14](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/be4cb1485b08c34bf87be69e7b3b8b8ad024a7a6))
* externalize population list for stimulus building ([c9608d8](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/c9608d819dd6c85b5161c713efc98aac794c9721))
* improve arithmetic evaluation, unify stimulus config parsing, and fix raw bundle handling ([90143d5](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/90143d58b61de7eec57ca20f7286ef99491184f6))
* improve CI HTML template rendering ([c610ecb](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/c610ecb2de0d5567de90db2932d710ce1f4943e4))
* modularize CI by extracting integration tests into triggered pipelines ([ec033eb](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/ec033eb49d8ab7ce936a9f9ce69e451dcc7ee637))
* postpone input cleanup and use VBT_INPUT_PATH env variable ([34c0ab3](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/34c0ab3625579e84bf96c78b31b407ae60bc04f8))
* rename cache job to create_new_cache ([8198873](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/81988731917cf78d90730d75838aeb51cd97914c))
* rename VBT BG test identifiers in CI config ([81d3032](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/81d3032e2fe8dcedfe24430d3586e458b42f0e38))
* replace pipeline‑variable needs with direct job dependencies and make them optional ([897c486](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/897c4862cbc04364ffcded82cc6cadd2bf398a34))
* replace zip helper, clean bundle interface, and tighten test discovery ([dc37b87](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/dc37b87fdab18428459a354d3faff2e61db4798d))
* simplify Allure metadata handling and unify CI job names ([d1c5340](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/d1c534011d2682a118c86823ab708bcd3eaeabb8))
* update integration wrappers to new generic converter CLI and prune unused CI test ([f38e633](https://gitlab.jsc.fz-juelich.de/virtualbraintwin/vbt-converter/commit/f38e6333f929a211768ca17a6ba9421a53046c66))
