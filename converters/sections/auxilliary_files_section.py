import logging
import os
import shutil
from pathlib import Path
from typing import Mapping, Dict

import h5py

from converters.interfaces.data_section import DataSection
from converters.utils.common import configure_logging

logger = configure_logging(
    __name__,
    log_file_name=Path(__file__).with_suffix(".log").name,
    level=logging.INFO,
)

class AuxiliaryFileSection(DataSection):
    """
    Keeps an *exact copy* of arbitrary text‑based files.  Small files (<8 MiB)
    are stored directly as a variable‑length string dataset; larger files are
    stored as external references within th output folder – still fully self‑contained.
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

    # ------------------------------------------------------------------
    def to_hdf5(self, grp: h5py.Group) -> None:
        for logical_name, src_path in self.mapping.items():
            if not src_path.is_file():
                logger.warning("Aux file %s does not exist – skipping.", src_path)
                continue

            # Decide if we keep the file *in* the bundle or as a link.
            # A simple heuristic – files larger than 8MiB become external links.
            # The threshold is exposed via an environment variable so that a
            # user can tune it without touching the code.
            ext_thresh = int(os.getenv("VBT_AUX_LINK_THRESHOLD", str(self.MAX_INLINE)))
            make_link = src_path.stat().st_size > ext_thresh

            if make_link:
                # Fast, zero‑copy reference
                dst = Path(self.output_dir) / Path(src_path).relative_to(Path(src_path).anchor)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst) if Path(src_path).is_file() else shutil.copytree(src_path, dst,
                                                                                             dirs_exist_ok=True)
                DataSection.store_external_path(grp, logical_name, Path(dst).relative_to(self.output_dir), make_link=True)
            else:
                # Keep small files inline (current behaviour)
                data = src_path.read_bytes()
                if len(data) <= self.MAX_INLINE:
                    try:
                        txt = data.decode("utf-8")
                        self._store_text_blob(grp, logical_name, txt, compress=True)
                    except UnicodeDecodeError:
                        self._store_binary_blob(grp, logical_name, data, compress=True)
                else:   # fall‑back for binary >MAX_INLINE
                    self._store_binary_blob(grp, logical_name, data, compress=True)

            grp[logical_name].attrs["original_path"] = str(src_path)

    # ------------------------------------------------------------------
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