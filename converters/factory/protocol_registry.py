"""Provides a small registry of protocol‑specific ``ProtocolBuilder`` objects that
construct :class:`~converters.schema.master_bundle.MasterBundle` instances from
raw input data.  Each concrete builder delegates the heavy lifting to a factory
function (e.g. :func:`converters.factory.amu_factory.build_amu_master_bundle`).

Typical usage
-------------
>>> from pathlib import Path
>>> builder = PROTOCOL_BUILDERS["amu"]
>>> bundle = builder(
...     input_path=Path("/data/raw"),
...     embed_raw=True,
...     embed_raw_path=Path("/data/raw_embedded"),
...     output_dir=Path("/data/processed"),
... )
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from converters.factory.amu_factory import build_amu_master_bundle
from converters.factory.basal_ganglia_factory import build_bg_master_bundle
from converters.schema.master_bundle import MasterBundle


class ProtocolBuilder(ABC):
    """Abstract base class for building protocol‑specific master bundles.

    Sub‑classes must implement ``__call__`` which returns a fully populated
    :class:`~converters.schema.master_bundle.MasterBundle` instance.
    """

    @abstractmethod
    def __call__(
            self,
            input_path: Path,
            embed_raw: bool,
            embed_raw_path: Path,
            output_dir: Path,
    ) -> MasterBundle:
        """Build a :class:`~converters.schema.master_bundle.MasterBundle`.

        Args:
            input_path: Path to the raw input data directory or file.
            embed_raw: If ``True`` the raw data will be embedded inside the bundle.
            embed_raw_path: Destination where the raw data should be copied when
                ``embed_raw`` is ``True``.
            output_dir: Directory where the builder may write auxiliary output
                files (e.g., intermediate JSON, logs).

        Returns:
            An instance of :class:`~converters.schema.master_bundle.MasterBundle`
            representing the processed protocol data.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
            PermissionError: If the process lacks read/write permission for any
                of the supplied paths.
            ValueError: If the combination of arguments is invalid (for example,
                ``embed_raw`` is ``True`` but ``embed_raw_path`` is not writable).
            RuntimeError: For any unexpected errors raised by concrete builder
                implementations.
        """
        raise NotImplementedError


class AMUBuilder(ProtocolBuilder):
    """Builder for the *Acute Myeloid Leukemia* (AMU) protocol.

    This implementation forwards all arguments to
    :func:`converters.factory.amu_factory.build_amu_master_bundle`.
    """

    def __call__(
            self,
            input_path: Path,
            embed_raw: bool,
            embed_raw_path: Path,
            output_dir: Path,
    ) -> MasterBundle:
        """Create an AMU ``MasterBundle`` from raw input data.

        The method simply proxies its arguments to
        :func:`converters.factory.amu_factory.build_amu_master_bundle`.

        Args:
            input_path: Path to the raw AMU input data.
            embed_raw: Whether to embed the raw data inside the bundle.
            embed_raw_path: Target path for the embedded raw data.
            output_dir: Directory where the builder may emit auxiliary files.

        Returns:
            A populated ``MasterBundle`` representing the AMU protocol.

        Raises:
            FileNotFoundError, PermissionError, ValueError, RuntimeError:
                Propagated from :func:`build_amu_master_bundle` if the underlying
                factory encounters an error.
        """
        return build_amu_master_bundle(
            input_path,
            embed_raw=embed_raw,
            embed_raw_path=embed_raw_path,
            output_dir=output_dir,
        )


class BasalGangliaBuilder(ProtocolBuilder):
    """Builder for the *Basal Ganglia* protocol.

    This implementation forwards all arguments to
    :func:`converters.factory.basal_ganglia_factory.build_bg_master_bundle`.
    """

    def __call__(
            self,
            input_path: Path,
            embed_raw: bool,
            embed_raw_path: Path,
            output_dir: Path,
    ) -> MasterBundle:
        """Create a Basal Ganglia ``MasterBundle`` from raw input data.

        The method simply proxies its arguments to
        :func:`converters.factory.basal_ganglia_factory.build_bg_master_bundle`.

        Args:
            input_path: Path to the raw Basal Ganglia input data.
            embed_raw: Whether to embed the raw data inside the bundle.
            embed_raw_path: Target path for the embedded raw data.
            output_dir: Directory where the builder may emit auxiliary files.

        Returns:
            A populated ``MasterBundle`` representing the Basal Ganglia protocol.

        Raises:
            FileNotFoundError, PermissionError, ValueError, RuntimeError:
                Propagated from :func:`build_bg_master_bundle` if the underlying
                factory encounters an error.
        """
        return build_bg_master_bundle(
            input_path,
            embed_raw=embed_raw,
            embed_raw_path=embed_raw_path,
            output_dir=output_dir,
        )


# Mapping from protocol identifier (as used by the CLI / UI) to the concrete
# ``ProtocolBuilder`` that knows how to produce a ``MasterBundle`` for that
# protocol.
PROTOCOL_BUILDERS: Dict[str, ProtocolBuilder] = {
    "amu": AMUBuilder(),
    "basal-ganglia": BasalGangliaBuilder(),
}
