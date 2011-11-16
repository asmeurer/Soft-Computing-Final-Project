#!/usr/bin/env python

import os

from Cython.Compiler.Main import compile

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext

source_root = os.path.dirname(__file__)

compiled_modules = [
    "neuralnet",
    "generate_patterns",
]

extensions = []

for module in compiled_modules:
    source_file = os.path.join(source_root, *module.split('.')) + ".py"

    print("Compiling module %s ..." % module)
    result = compile(source_file)

    if result.c_file is None:
        raise RuntimeError("failed to compile %s" % module)

    extensions.append(
        Extension(module, sources=[str(result.c_file)],
            extra_compile_args=['-O2', '-Wall'],
        )
    )

setup(
    name        = "NeuralNet",
    packages    = [
        "neuralnet",
        "generate_patterns",
    ],
    cmdclass    = {
        "build_ext": build_ext
    },
    ext_modules = extensions
)

