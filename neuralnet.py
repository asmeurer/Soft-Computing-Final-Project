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

NODES = [11, 4, 1]
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
            H, O = activations_and_outputs(pattern, W)
            D = errorsignals(pattern, W, H, O)
            W = adaptweights(W, D, O)
            return

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

def sigmoid(h, k=1/20):
    """
    Return 1/(1 + exp(-k*h)).

    Note, it instead calculates (1 + tanh(k*h/2))/2, which is equivalent, but
    doesn't have problems computing negative numbers with large absolute value.

    Note that it still has the problem where large numbers are rounded to 1.0.
    I think the solution for this is to use smaller values for k.
    """
    return (1 + math.tanh(k*h/2))/2

def sigmoiddiff(fh, k=1/20): # Make sure k is the same here as above
    """
    Return d/dh(f(h)) in terms of f(h), where f(h) = 1/(1 + exp(-k*h)).
    """
    return k*fh*(1 - fh)

def activations_and_outputs(pattern, W):
    H, O = [], []
    input = pattern[0]
    for layer in W:
        h, o = [], []
        for node in layer:
            h.append(activation(input, node))
        H.append(h)
        o = [sigmoid(i) for i in h]
        O.append(o)
        input = o

    return H, O

def activation(input, node):
    assert len(node) == len(input)
    return sum(wij*opi for wij, opi in zip(node, input))

def errorsignals(pattern, W, H, O):
    # Outer layer
    D = [[sigmoiddiff(o)*(pattern[1] - o) for o in O[-1]]]

    for i, layer in reversed(list(enumerate(W))):
        if i == len(W) - 1:
            continue
        d = []
        for j, node in enumerate(layer):
            a = H[i][j]
            outputws = [W[i + 1][k][j] for k in xrange(len(W[i + 1]))]
            # XXX: This is wrong
            assert len(D[-1]) == len(outputws)
            d.append(sigmoiddiff(o)*sum(dd*w for dd, w in
                zip(D[-1], outputws)))
        D.append(d)

    D = list(reversed(D))
    return D

def adaptweights(W, D, O):
    newW = []
    for wlayer, dlayer, olayer in zip(W, D, O):
        neww = []
        for wnode, d, o in zip(wlayer, dlayer, olayer):
            neww.append([w + eta*d*o for w in wnode])
        newW.append(neww)

    return newW

if __name__ == "__main__":
    main()
    sys.exit(0)
