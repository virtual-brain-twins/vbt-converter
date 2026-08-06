"""
A small helper that stores a *zip* (or any binary) file inside an HDF5
container.  The implementation is deliberately generic – it can store
any ``Path`` that points to a file on disk, but it is primarily used for
the original ``.zip`` bundles that the VBT converter works with.

Key features
------------
*   **Chunk‑wise streaming** – the file is never fully materialised in RAM.
*   **Correct SHA‑256 checksum** – calculated on‑the‑fly while streaming.
*   **Robust error handling** – friendly messages for missing files,
    existing datasets, permission problems, etc.
*   **Optional compression** – you can ask for gzip/lzf/zstd compression if you
    ever need it (default is *no* compression because the zip itself is
    already compressed).
*   **Round‑trip support** – ``from_hdf5`` creates a lightweight proxy
    object that can either expose the raw bytes or write them back to a
    temporary file.

The class follows the same ``HDF5Serializable`` contract used throughout
the project, i.e. it implements ``to_hdf5`` and ``from_hdf5`` class‑methods.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import make_group

# Constants
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB – a good compromise for SSD/HDD


# Helper: incremental SHA‑256
def _incremental_sha256(file_path: Path) -> str:
    """
    Compute the SHA‑256 checksum of *file_path* by reading it in
    ``CHUNK_SIZE`` blocks.  This avoids loading the whole file into memory.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


# RawBundleGroup implementation
@dataclass
class RawBundleGroup(HDF5Serializable):
    """
    Store the original raw ``.zip`` archive **inside** the HDF5 file.

    Parameters
    ----------
    zip_path:
        Path to the zip file that will be streamed into the HDF5 dataset.
        The path is stored *as‑is* – it is not copied elsewhere.
    checksum:
        Optional SHA‑256 checksum.  If ``None`` a checksum is calculated
        while writing the file to HDF5.  The checksum is also stored as an
        attribute on the dataset for later verification.
    compression:
        HDF5 compression filter to use (e.g. ``'gzip'``, ``'lzf'`` or
        ``'szip'``).  The default ``None`` means “no compression”.  This
        argument is exposed mainly for testing; in production the zip is
        already compressed so we keep the data uncompressed for speed.
    """

    zip_path: Path = field(default_factory=Path)
    checksum: Optional[str] = None
    compression: Optional[Union[str, Tuple[str, int]]] = None  # e.g. ("gzip", 4)

    # Private utilities
    def _validate_inputs(self) -> None:
        """Raise a clear error if the supplied zip_path is not usable."""
        if not isinstance(self.zip_path, Path):
            raise TypeError(f"zip_path must be a pathlib.Path, got {type(self.zip_path)!r}")

        if not self.zip_path.is_file():
            raise FileNotFoundError(f"File not found: {self.zip_path}")

        if not os.access(self.zip_path, os.R_OK):
            raise PermissionError(f"Cannot read file: {self.zip_path}")

    # Public API – serialization
    def to_hdf5(self, root: h5py.File) -> None:
        """
        Write the zip file to ``root`` under the group ``RawBundle`` and
        dataset ``raw_bundle.zip``.

        The dataset is created with ``maxshape=(None,)`` (i.e. extendable)
        and a chunk size of ``CHUNK_SIZE`` bytes.  Data is streamed in
        chunks to keep the RAM footprint small.

        A SHA‑256 checksum is either taken from ``self.checksum`` (if
        already supplied) or computed on‑the‑fly while the file is being
        written.

        The following attributes are set on the dataset:

        * ``format``      – ``'zip'`` (kept for backwards compatibility)
        * ``checksum``    – the SHA‑256 checksum as a hex string
        * ``original_name`` – the name of the source file (useful for debugging)
        * ``original_size`` – size of the source file in bytes
        """
        # Basic sanity checks
        self._validate_inputs()

        # Prepare / create the group and dataset
        raw_grp = make_group(root, "RawBundle")  # creates if missing, re‑uses otherwise

        dset_name = "raw_bundle.zip"

        # If the dataset already exists we *replace* it – this keeps the
        # class idempotent (you can call ``to_hdf5`` twice on the same
        # object without blowing up).  H5PY does not allow in‑place resize
        # to zero, so we delete first.
        if dset_name in raw_grp:
            # Deleting the old dataset frees the underlying storage.
            del raw_grp[dset_name]

        # H5PY expects the shape tuple; ``(0,)`` creates an empty extensible
        # dataset ready for chunked appends.
        dset = raw_grp.create_dataset(
            dset_name,
            shape=(0,),
            maxshape=(None,),
            dtype="uint8",
            chunks=(CHUNK_SIZE,),
            compression=self.compression,
        )

        # Stream the file into the dataset *and* compute the checksum.
        sha256 = hashlib.sha256()
        total_written = 0

        with self.zip_path.open("rb") as src:
            while True:
                chunk = src.read(CHUNK_SIZE)
                if not chunk:
                    break

                # Update checksum
                sha256.update(chunk)

                # Resize dataset – resize expects a *tuple*.
                new_len = total_written + len(chunk)
                dset.resize((new_len,))

                # Write the new bytes.
                dset[total_written:new_len] = np.frombuffer(chunk, dtype="uint8")

                total_written = new_len

        # Store attributes
        derived_checksum = sha256.hexdigest()
        final_checksum = self.checksum or derived_checksum
        dset.attrs["format"] = "zip"
        dset.attrs["checksum"] = final_checksum
        dset.attrs["original_name"] = self.zip_path.name
        dset.attrs["original_size"] = self.zip_path.stat().st_size

        # Keep the computed checksum on the instance for later use.
        self.checksum = final_checksum

    # Public API – deserialization
    @classmethod
    def from_hdf5(cls, root: h5py.File) -> "RawBundleGroup":
        """
        Re‑create a ``RawBundleGroup`` object from an HDF5 file.

        The returned object does **not** contain a real on‑disk file.
        Instead, ``zip_path`` points to a *virtual* location that signals the
        dataset is embedded in the HDF5 file.

        If you need the raw bytes you can either:

        * read them directly via ``dset[()]`` (returns a ``bytes`` object)
        * or call ``extract_to(path)`` which writes the contents to a
          temporary file and returns the path.

        Returns
        -------
        RawBundleGroup
            Instance with ``zip_path`` set to a placeholder and the stored
            checksum attached.
        """
        if "RawBundle" not in root:
            raise KeyError("Group 'RawBundle' not found in the supplied HDF5 file.")

        raw_grp = root["RawBundle"]
        if "raw_bundle.zip" not in raw_grp:
            raise KeyError("Dataset 'raw_bundle.zip' not found in group 'RawBundle'.")

        dset = raw_grp["raw_bundle.zip"]
        checksum = dset.attrs.get("checksum")

        # We create a *placeholder* Path that identifies the data lives inside
        # the HDF5 file.  It is never meant to be opened directly.
        placeholder = Path(f"<embedded:{root.filename}>")

        # Create the instance – ``zip_path`` is a dummy; the real bytes are
        # inside the HDF5 dataset.
        instance = cls(zip_path=placeholder, checksum=checksum)

        # Store a reference to the dataset so that helper methods can access
        # it without having to re‑lookup the same objects repeatedly.
        instance._h5_dataset = dset  # type: ignore[attr-defined]

        return instance

    # Helper – extract the embedded zip to a *real* file
    def extract_to(self, destination: Union[Path, str]) -> Path:
        """
        Write the embedded ``raw_bundle.zip`` to ``destination`` (which can be
        a directory or a full file path).  The method works for objects that
        have been created via ``from_hdf5`` as well as for fresh instances
        that have just written their data.

        Parameters
        ----------
        destination :
            If a directory is passed, the original file name (as stored in the
            ``original_name`` attribute) is used.  If a full file path is
            passed, the data is written there verbatim.

        Returns
        -------
        pathlib.Path
            The absolute path of the written file.
        """
        # Resolve the final path
        dest_path = Path(destination).expanduser().resolve()
        if dest_path.is_dir():
            # Use the original name if we have it, else fall back to a generic one.
            original_name = getattr(self, "_h5_dataset", None)
            if original_name is not None:
                original_name = original_name.attrs.get("original_name", "raw_bundle.zip")
            else:
                original_name = self.zip_path.name or "raw_bundle.zip"
            dest_path = dest_path / original_name

        # Ensure parent directory exists.
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Grab the dataset (either stored on the instance or re‑lookup it)
        if hasattr(self, "_h5_dataset"):  # instance was built via from_hdf5
            dset = self._h5_dataset  # type: ignore[attr-defined]
        else:
            # Normal (non‑embedded) case – open the original file and copy it.
            # This is a fast-path that avoids reading the whole dataset into RAM.
            # We simply copy the source file.
            with self.zip_path.open("rb") as src, dest_path.open("wb") as dst:
                while chunk := src.read(CHUNK_SIZE):
                    dst.write(chunk)
            return dest_path

        # Write the dataset to disk in a memory‑friendly way.
        with dest_path.open("wb") as out_f:
            total = dset.shape[0]
            # Iterate over the dataset in chunks – ``dset[i:j]`` returns a NumPy array.
            for start in range(0, total, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, total)
                out_f.write(dset[start:end].tobytes())

        return dest_path

    # Optional: representation helpers
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(zip_path={self.zip_path!s}, "
            f"checksum={self.checksum!r}, compression={self.compression!r})"
        )
