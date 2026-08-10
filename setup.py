# SPDX-License-Identifier: Apache-2.0
"""PEP 517 build hook for CodaraScan's optional-at-runtime native accelerator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "codarascan" / "engines" / "common" / "native" / "sttg_native.cpp"
SOURCE_RELATIVE = SOURCE.relative_to(ROOT).as_posix()

if sys.platform == "win32":
    compile_args = ["/O2", "/std:c++17", "/DNDEBUG"]
else:
    compile_args = ["-O3", "-g0", "-std=c++17", "-DNDEBUG"]

extensions = []
if os.environ.get("CODARASCAN_BUILD_NATIVE", "1") != "0":
    extensions.append(
        Extension(
            "codarascan._sttg_native",
            sources=[SOURCE_RELATIVE],
            include_dirs=[numpy.get_include()],
            language="c++",
            extra_compile_args=compile_args,
            # Source installs remain functional through the observable Python
            # fallback when no compiler is available. Official wheel jobs set
            # CODARASCAN_REQUIRE_NATIVE=1 and fail instead.
            optional=os.environ.get("CODARASCAN_REQUIRE_NATIVE") != "1",
        )
    )

setup(ext_modules=extensions)
