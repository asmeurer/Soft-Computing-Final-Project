#! /usr/bin/env python

"""
This generates the subset problems for input to the neural network.
"""

import argparse
from random import randint

# TODO: Refactor these into command line arguments:

setsize = 10
number_of_sets = 100

parser = argparse.ArgumentParser(description="This generates the subset "
    "problems for input to the neural network.")

parser.add_argument('-s', '--seed', metavar='N', type=int, nargs=1,
                   help='The random seed used to generate the output.')

args = parser.parse_args()


def main():
    for i in range(number_of_sets):
        print [(-1)**randint(0, 1)*randint(1, 10) for i in range(setsize)]
    return

if __name__ == "__main__":
    main()
