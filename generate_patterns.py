#! /usr/bin/env python

"""
This generates subset problems for input to the neural network.
"""

import argparse
from random import randint, seed
from itertools import combinations, combinations_with_replacement

# TODO: Refactor these into command line arguments:

parser = argparse.ArgumentParser(description="This generates subset "
    "problems for input to the neural network.")

parser.add_argument('filename', metavar='FILE', type=str,
                    help="Name of the file to train the network with.")
parser.add_argument('-s', '--seed', metavar='N', type=int,
                    help="The random seed used to generate the output.")
parser.add_argument('-n', '--number', metavar='N', type=int, default=100,
                    help="The number of sets to generate.")

args = parser.parse_args()

setrange = 500
setsize = 10
number_of_sets = args.number

seed(args.seed)

def subsets(seq, k=None, repetition=False):
    """
    Generates all k-subsets (combinations) from an n-element set, seq.

    Copyright (c) 2011 SymPy Development Team

    A k-subset of an n-element set is any subset of length exactly k. The
    number of k-subsets of an n-element set is given by binomial(n, k),
    whereas there are 2**n subsets all together. If k is None then all
    2**n subsets will be returned from shortest to longest.

    Examples:
        >>> from sympy.utilities.iterables import subsets

    subsets(seq, k) will return the n!/k!/(n - k)! k-subsets (combinations)
    without repetition, i.e. once an item has been removed, it can no
    longer be "taken":
        >>> list(subsets([1, 2], 2))
        [(1, 2)]
        >>> list(subsets([1, 2]))
        [(), (1,), (2,), (1, 2)]
        >>> list(subsets([1, 2, 3], 2))
        [(1, 2), (1, 3), (2, 3)]


    subsets(seq, k, repetition=True) will return the (n - 1 + k)!/k!/(n - 1)!
    combinations *with* repetition:
        >>> list(subsets([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:
        >>> list(subsets([0, 1], 3, repetition=False))
        []
        >>> list(subsets([0, 1], 3, repetition=True))
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
       """
    if k is None:
        for k in range(len(seq) + 1):
            for i in subsets(seq, k, repetition):
                yield i
    else:
        if not repetition:
            for i in combinations(seq, k):
                yield i
        else:
            for i in combinations_with_replacement(seq, k):
                yield i

def main():
    sets = []
    has_subsets = []
    with open(args.filename, 'w') as file:
        for i in range(number_of_sets):
            a = [(-1)**randint(0, 1)*randint(1, setrange) for i in range(setsize)]
            sets.append(a)
            has_subset = any(sum(i) == 0 for i in subsets(a) if i)
            has_subsets.append(has_subset)
            file.write("%s, %s\n" % (tuple(a), has_subset))
        # print "%d/%d" % (sum(int(i) for i in has_subsets), len(has_subsets))

if __name__ == "__main__":
    main()
