"""Runtime path configuration for the external Codara application."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def codara_backend_root() -> Path | None:
    """Expose Codara's backend packages when ``CODARA_BACKEND_ROOT`` is set.

    The repository does not assume a sibling checkout or a machine-specific
    absolute path. A Codara integration run must opt in explicitly with the
    backend directory, for example ``/workspace/codara-app/apps/backend``.
    """

    value = os.environ.get("CODARA_BACKEND_ROOT")
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CODARA_BACKEND_ROOT is not a directory: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
