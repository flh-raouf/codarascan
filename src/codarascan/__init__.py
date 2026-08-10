# SPDX-License-Identifier: Apache-2.0
"""CodaraScan public API."""

__version__ = "0.1.2"

from .core.contracts import Roi
from .documents import DocumentInput, DocumentStream, ErrorPolicy, WorkerCount
from .engines.common.native.loader import NativeFallbackWarning
from .errors import (
    CodaraScanError,
    ConfigurationError,
    DocumentError,
    DocumentRenderError,
    InternalProcessingError,
    InvalidImageError,
    NativeBackendError,
    PageProcessingError,
    ProtocolError,
    SerializationError,
    UnsupportedFormatError,
)
from .formats import FormatInfo, FormatKind, supported_formats
from .models import (
    DecodedSymbolResult,
    DocumentResult,
    ErrorDetail,
    ImageResult,
    PageResult,
    Point,
    ScanMetadata,
    SymbolKind,
    SymbolResult,
    SymbolStatus,
)
from .scanner import ScanMode, Scanner, SymbolGroup, iter_document, scan_document, scan_image
from .serialization import to_json

ROI = Roi

__all__ = [
    "CodaraScanError",
    "ConfigurationError",
    "DecodedSymbolResult",
    "DocumentError",
    "DocumentInput",
    "DocumentRenderError",
    "DocumentResult",
    "DocumentStream",
    "ErrorDetail",
    "ErrorPolicy",
    "FormatInfo",
    "FormatKind",
    "ImageResult",
    "InternalProcessingError",
    "InvalidImageError",
    "NativeBackendError",
    "NativeFallbackWarning",
    "PageProcessingError",
    "PageResult",
    "Point",
    "ProtocolError",
    "ROI",
    "Roi",
    "ScanMetadata",
    "ScanMode",
    "Scanner",
    "SerializationError",
    "SymbolGroup",
    "SymbolKind",
    "SymbolResult",
    "SymbolStatus",
    "UnsupportedFormatError",
    "WorkerCount",
    "__version__",
    "iter_document",
    "scan_document",
    "scan_image",
    "supported_formats",
    "to_json",
]
