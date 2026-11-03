# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
"""Provides a small registry of protocol‑specific ``ProtocolBuilder`` objects that
construct :class:`~converters.schema.master_bundle.MasterBundle` instances from
raw input data.  Each concrete builder delegates the heavy lifting to a factory
function (e.g. :func:`converters.factory.uva_factory.build_uva_master_bundle`).

Typical usage
-------------
>>> from pathlib import Path
>>> builder = PROTOCOL_BUILDERS["uva"]  # or "kth"
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

from converters.factory.kth_factory import build_kth_master_bundle
from converters.factory.uva_factory import build_uva_master_bundle
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
            protocol_version: str,
            protocol_info_path: Path,
            **extra_kwargs: object
    ) -> MasterBundle:
        """Build a :class:`~converters.schema.master_bundle.MasterBundle`.

        Args:
            input_path: Path to the raw input data directory or file.
            embed_raw: If ``True`` the raw data will be embedded inside the bundle.
            embed_raw_path: Destination where the raw data should be copied when
                ``embed_raw`` is ``True``.
            output_dir: Directory where the builder may write auxiliary output
                files (e.g., intermediate JSON, logs).
            protocol_version: The version of the protocol to be incorporated.
            protocol_info_path: Path to a PDF file containing protocol information.
            **extra_kwargs: Additional keyword arguments for the protocol builder. e.g. a ``parameters`` list

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


class UVABuilder(ProtocolBuilder):
    """Builder for the UVA protocol.

    This implementation forwards all arguments to
    :func:`converters.factory.uva_factory.build_uva_master_bundle`.
    """

    def __call__(
            self,
            input_path: Path,
            embed_raw: bool,
            embed_raw_path: Path,
            output_dir: Path,
            protocol_version: str,
            protocol_info_path: Path,
            **extra_kwargs: object,
    ) -> MasterBundle:
        """Create an UVA ``MasterBundle`` from raw input data.

        The method simply proxies its arguments to
        :func:`converters.factory.uva_factory.build_uva_master_bundle`.

        Args:
            input_path: Path to the raw UVA input data.
            embed_raw: Whether to embed the raw data inside the bundle.
            embed_raw_path: Target path for the embedded raw data.
            output_dir: Directory where the builder may emit auxiliary files.
            protocol_version: The version of the protocol to be incorporated.
            protocol_info_path: Path to a PDF file containing protocol information.
            **extra_kwargs: Additional keyword arguments to pass to the factory function.

        Returns:
            A populated ``MasterBundle`` representing the UVA protocol.

        Raises:
            FileNotFoundError, PermissionError, ValueError, RuntimeError:
                Propagated from :func:`build_uva_master_bundle` if the underlying
                factory encounters an error.
        """
        return build_uva_master_bundle(
            input_path,
            embed_raw=embed_raw,
            embed_raw_path=embed_raw_path,
            output_dir=output_dir,
            protocol_version=protocol_version,
            protocol_info_path=protocol_info_path,
            **extra_kwargs
        )


class KTHBuilder(ProtocolBuilder):
    """Builder for the *KTH* protocol.

    This implementation forwards all arguments to
    :func:`converters.factory.kth_factory.build_kth_master_bundle`.
    """

    def __call__(
            self,
            input_path: Path,
            embed_raw: bool,
            embed_raw_path: Path,
            output_dir: Path,
            protocol_version: str,
            protocol_info_path: Path,
            **extra_kwargs: object
    ) -> MasterBundle:
        """Create a KTH ``MasterBundle`` from raw input data.

        The method simply proxies its arguments to
        :func:`converters.factory.kth_factory.build_kth_master_bundle`.

        Args:
            input_path: Path to the raw KTH input data.
            embed_raw: Whether to embed the raw data inside the bundle.
            embed_raw_path: Target path for the embedded raw data.
            output_dir: Directory where the builder may emit auxiliary files.
            protocol_version: The version of the protocol to be incorporated.
            protocol_info_path: Path to a PDF file containing protocol information.
            **extra_kwargs: Additional keyword arguments to pass to the factory function.

        Returns:
            A populated ``MasterBundle`` representing the KTH protocol.

        Raises:
            FileNotFoundError, PermissionError, ValueError, RuntimeError:
                Propagated from :func:`build_kth_master_bundle` if the underlying
                factory encounters an error.
        """
        return build_kth_master_bundle(
            input_path,
            embed_raw=embed_raw,
            embed_raw_path=embed_raw_path,
            output_dir=output_dir,
            protocol_version=protocol_version,
            protocol_info_path=protocol_info_path,
            **extra_kwargs
        )


# Mapping from protocol identifier (as used by the CLI / UI) to the concrete
# ``ProtocolBuilder`` that knows how to produce a ``MasterBundle`` for that
# protocol.
PROTOCOL_BUILDERS: Dict[str, ProtocolBuilder] = {
    "uva": UVABuilder(),
    "kth": KTHBuilder(),
}
