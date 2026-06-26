import logging
import os
import shutil
from pathlib import Path
from typing import Mapping, Dict

import h5py

from converters.interfaces.data_section import DataSection
from converters.utils.common import configure_logging

logger = configure_logging(
    module_name=__name__,
    log_file_name=Path(__file__).with_suffix(".log").name,
    level=logging.INFO,
)


class AuxiliaryFileSection(DataSection):
    """
    Keeps an *exact copy* of arbitrary text‑based files.  Small files (<8MiB)
    are stored directly as a binary dataset; larger files are
    stored as external references within the output folder – still fully self‑contained.
    """
    MAX_INLINE = 8 * 1024 * 1024  # 8MiB

    def __init__(self, mapping: Mapping[str, Path] | None = None, output_dir: Path = Path(".")) -> None:
        """
        ``mapping``: logical-name → absolute path of the source file.
        The class does **not** read the files at construction time – that is
        deferred to ``to_hdf5`` so that you can reuse the same object for
        multiple bundles if you wish.
        """
        self.output_dir: Path = output_dir
        self.mapping: Dict[str, Path] = {k: Path(v) for k, v in (mapping or {}).items()}

    def to_hdf5(self, grp: h5py.Group) -> None:
        """
        Store every entry of ``self.mapping`` in *grp*.

        * If the source file is larger than the threshold (env var
          ``VBT_AUX_LINK_THRESHOLD`` or the default 8MiB) we copy the file
          under ``self.output_dir`` and create an *external HDF5 link*.
        * Otherwise we read the whole file into memory and store the raw
          bytes as a binary dataset.

        For every dataset we set two mandatory attributes:

        * ``original_path`` – the absolute path the file came from.
        * ``is_external``   – ``True`` when the dataset is an external link,
          ``False`` when the payload is stored inline.
        """

        # Helper: decide whether a file should become an external link.
        def _make_external(src: Path) -> bool:
            # The threshold can be tuned by the user through an env‑var.
            # ``int`` will raise if the variable is malformed – we fall back
            # to the class constant in that case.
            try:
                thresh = int(os.getenv("VBT_AUX_LINK_THRESHOLD", str(self.MAX_INLINE)))
            except ValueError:
                thresh = self.MAX_INLINE
            return src.stat().st_size > thresh

        for logical_name, src_path in self.mapping.items():
            # Validate source
            if not src_path.is_file():
                logger.warning("Aux file %s does not exist – skipping.", src_path)
                continue

            try:
                # Decide on external vs. inline storage
                if _make_external(src_path):
                    # External link
                    # Destination is a *mirrored* tree under ``output_dir`` that
                    # preserves the absolute path hierarchy (except the root‑/
                    # drive letter).  This guarantees a deterministic layout
                    # independent of the current working directory.
                    rel_to_root = src_path.relative_to(src_path.anchor)
                    dst = self.output_dir / rel_to_root
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst)

                    # Create an *ExternalLink* that points to the copied file.
                    # The link lives inside *grp* under the logical name.
                    ext_link = h5py.ExternalLink(str(dst), "")
                    grp[logical_name] = ext_link

                    # Mark the dataset as external.
                    dset = grp[logical_name]
                    dset.attrs["is_external"] = True
                    dset.attrs["external_path"] = str(dst.relative_to(self.output_dir))
                else:
                    # Inline (binary) storage
                    data = src_path.read_bytes()
                    # Store exactly the bytes, but compress them to save space.  We do not use a variable‑length string
                    self._store_binary_blob(grp, logical_name, data, compress=True)
                    dset = grp[logical_name]
                    dset.attrs["is_external"] = False

                # Common attributes (original location)
                dset.attrs["original_path"] = str(src_path)

            except Exception as exc:
                logger.error(
                    "Failed to store auxiliary file %s as %s: %s",
                    src_path,
                    logical_name,
                    exc,
                    exc_info=True,
                )
                # Continue with the next file
                continue

    @classmethod
    def from_group(cls, grp: h5py.Group) -> "AuxiliaryFileSection":
        """
        Re‑create the section *without* trying to rebuild the original files.
        It only records the *names* of the datasets (which are the logical
        identifiers).  If you need the original bytes you can read
        ``grp[name][()]`` yourself.
        """
        mapping: Dict[str, Path] = {}
        for dset_name in grp:
            # We store the path as a string attribute, but we do not recreate the actual file.
            original = grp[dset_name].attrs.get("original_path", "")
            mapping[dset_name] = Path(original) if original else Path()
        return cls(mapping)

    def from_hdf5(self, grp: h5py.Group) -> None:
        self.mapping = self.from_group(grp).mapping
