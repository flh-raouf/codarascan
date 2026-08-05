"""Build the optional local STTG native proposal extension."""

from __future__ import annotations

from pathlib import Path

import numpy
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="barcode-detection-sttg-native",
    version="1.0.0",
    packages=[],
    py_modules=[],
    ext_modules=[
        Extension(
            "_sttg_native",
            sources=[str(ROOT / "sttg_native.cpp")],
            include_dirs=[numpy.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-DNDEBUG"],
        )
    ],
)
