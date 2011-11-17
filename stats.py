#!/usr/bin/env python
"""
Compute stastics from the outputs of the neural network.

This uses numpy to compute the mean and standard deviation.
"""

import argparse
import sys

from numpy import mean, std

parser = argparse.ArgumentParser(description="This computes stastics "
    "from the outputs of the neural network.")

parser.add_argument('filename', metavar='FILE', type=str,
                    help="Name of the file of outputs.")

args = parser.parse_args()

def main(args):
    outputs = getoutputs(args.filename)

    # Create a dictionary of dictionaries for each data attribute
    # that has a list of the corresponding attributes.
    # For example, bydict['NODES'][(30, 30, 30, 1)]['accuracy'] will give you
    # a list of all the accuracies for outputs with NODES = [30, 30, 30, 1]
    bydict = {}
    for key in outputs[0]:
        bydict[key] = {}
    for outputdict in outputs:
        for key1 in outputdict:
            for key2 in outputdict:
                if key2 == key1:
                    continue
                o = outputs[key1]
                if isinstance(o, list):
                    o = tuple(o)
                if o not in bydict[key1]:
                    bydict[key1] = {key2: outputdict[key2]}
                else:
                    if outputdict[key2] not in bydict[key1][o]:
                        bydict[key1][o] = [outputdict[key2]]
                    else:
                        bydict[key1][o].append(outputdict[key2])


def getoutputs(filename):
    with open(filename) as file:
        outputtxt = file.read()

    outputs = []

    for line in outputtxt.split('\n'):
        if not line:
            # Handle blanke lines at the end of the file
            continue

        outputs.append(eval(line))

    return outputs

if __name__ == "__main__":
    main(args)
    sys.exit(0)

