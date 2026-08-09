# SPDX-License-Identifier: Apache-2.0
"""Typed public exceptions and stable machine-readable error codes."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar


class CodaraScanError(Exception):
    """Base class for expected CodaraScan failures."""

    code: ClassVar[str] = "codarascan_error"

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = MappingProxyType(dict(context or {}))


class ConfigurationError(CodaraScanError, ValueError):
    code = "configuration_error"


class UnsupportedFormatError(ConfigurationError):
    code = "unsupported_format"


class InvalidImageError(CodaraScanError, ValueError):
    code = "invalid_image"


class DocumentError(CodaraScanError):
    code = "document_error"


class DocumentRenderError(DocumentError):
    code = "document_render_error"


class PageProcessingError(DocumentError):
    code = "page_processing_error"


class NativeBackendError(CodaraScanError):
    code = "native_backend_error"


class SerializationError(CodaraScanError):
    code = "serialization_error"


class ProtocolError(CodaraScanError):
    code = "protocol_error"


class InternalProcessingError(CodaraScanError):
    code = "internal_processing_error"
