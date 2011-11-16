"""Helper module for cooperation with Cython. """

import cython

def cythonized(intspecs='', floatspecs=''):
    arg_types = {}

    for spec in intspecs.split(' '):
        arg_types[spec] = cython.int
    for spec in floatspecs.split(' '):
        arg_types[spec] = cython.float

    return cython.locals(**arg_types)
