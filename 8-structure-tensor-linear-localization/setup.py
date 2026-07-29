#!/usr/bin/env python3
"""Build the optional fused native STTG proposal kernel in-place."""

from __future__ import annotations

from pathlib import Path

import numpy
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="sttg-native",
    version="0.1.0",
    ext_modules=[
        Extension(
            "_sttg_native",
            [str(ROOT / "native" / "sttg_native.cpp")],
            include_dirs=[numpy.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-DNDEBUG"],
        )
    ],
)
