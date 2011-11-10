#! /usr/bin/env python

"""
Project 1. Aaron Meurer.

This is a multilayer perceptron neural network.  The project is to experiment
with using it to solve the subset sum problem, which is NP-complete.

"""
from __future__ import division
import sys
import random
import math
import argparse

NODES = [10, 4, 1]
eps = 0.1
eta = 1

parser = argparse.ArgumentParser(description="Runs the neural network.")

parser.add_argument('-s', '--seed', metavar='N', type=int,
                   help='The random seed used to generate the initial weights.')

args = parser.parse_args()

seed = args.seed

def main():
    patterns = get_problems()
    # Each weight corresponds to the inputs of the node.
    W = init_weights(patternlength=len(patterns[0][0]), seed=seed)
    while objective(patterns) < eps:
        for pattern in patterns:
            return W

def get_problems():
    with open("patterns", 'r') as file:
        problemstxt = file.read()

    problems = []

    for line in problemstxt.split('\n'):
        if not line:
            # Handle blank lines at the end of the file
            continue

        set, has_subsets = eval(line)
        problems.append((set, int(has_subsets)))

    return problems

def init_weights(patternlength, seed=None, max=1):
    random.seed(seed)

    W = []
    _NODES = [patternlength] + NODES
    for i in xrange(len(NODES)):
        w = []
        for j in xrange(NODES[i]):
            w.append([random.random()*max for k in xrange(_NODES[i])])
        W.append(w)

    return W

def objective(patterns):
    pass

def error(pattern):
    pass

def sigmoid(h, k=1):
    """
    Return 1/(1 + exp(-k*h)).
    """
    return 1/(1 + math.exp(-k*h))

def sigmoiddiff(fh, k=1):
    """
    Return d/dh(f(h)) in terms of f(h), where f(h) = 1/(1 + exp(-k*h)).
    """
    return k*fh*(1 - fh)

def activations_and_outputs(pattern, W):
    A, O = [], []
    input = pattern[0]
    for layer in W:
        a, o = [], []
        for node in layer:
            a.append(activation(input, node))
        A.append(a)
        o = [sigmoid(i) for i in a]
        O.append(o)
        input = o

    return A, O

def activation(input, node):
    assert len(node) == len(input)
    return sum(wij*opi for wij, opi in zip(node, input))

def errorsignals(pattern, W, O):
    D = []
    for i, layer in reversed(enumerate(W)):
        d = []
        for j, node in enumerate(layer):
            o = O[i][j]
            if i == len(NODES) - 1:
                # Outer layer
                dnew.append(sigmoiddiff(o)*(pattern[1] - o))
            else:
                for d in D[-1]:
                    assert len(d) == len(W[i + 1])
                    dnew.append(sigmioddiff(o)*sum(dd*wj for dd, w in
                        zip(d, W[i + 1])))
        D.append(dnew)

    D = list(reversed(D))
    return D

if __name__ == "__main__":
    main()
    sys.exit(0)
