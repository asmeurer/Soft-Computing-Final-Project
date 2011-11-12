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

NODES = [10, 10, 1]
global eta
eta = 1
eps = 0.1

parser = argparse.ArgumentParser(description="Runs the neural network.")

parser.add_argument('-s', '--seed', metavar='N', type=int,
                   help="The random seed used to generate the initial weights.")
parser.add_argument('filename', type=str, help="Name of the file to "
                    "run the network on (defaults to 'patterns'")

args = parser.parse_args()

seed = args.seed
filename = args.filename

def main():
    global eta

    a = 1
    b = 0.01
    B = [[1 for i in xrange(j)] for j in NODES]
    patterns = get_problems(filename)
    # Each weight corresponds to the inputs of the node.
    oldW = init_weights(patternlength=len(patterns[0][0]), max=0)
    W = init_weights(patternlength=len(patterns[0][0]), seed=seed, max=1)

    obj = None
    oldobj = None
    outputs = {}
    epochs = 0
    while True:
        epochs += 1
        for pattern in patterns:
            H, O = activations_and_outputs(pattern, W, B)
            outputs[pattern] = O
            D = errorsignals(pattern, W, H, O)
            oldW, (W, B) = W, adaptweights(W, D, pattern, O, oldW, B, eta=eta)
        oldoldobj, oldobj, obj = oldobj, obj, objective(patterns, outputs)
        if True:
            if oldobj:
                if obj - oldobj < 0:
                    eta += a
                elif obj - oldobj > 0:
                    eta -= b*eta
        print obj, eta, epochs
        if obj < eps:
            print B, W
            print "Converged in %d epochs." % epochs
            break

def get_problems(filename=None):
    if not filename:
        filename = "patterns"
    with open(filename, 'r') as file:
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
            w.append([random.random()*max*(-1)**random.randint(0, 1)
                for k in xrange(_NODES[i])])
        W.append(w)

    return W

def objective(patterns, outputs):
    return sum(error(pattern, outputs[pattern]) for pattern in patterns)

def error(pattern, O):
    return (pattern[1] - O[-1][0])**2/2

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

def activations_and_outputs(pattern, W, B):
    H, O = [], []
    input = pattern[0]
    for layer, blayer in zip(W, B):
        h, o = [], []
        for node, bias in zip(layer, blayer):
            h.append(activation(input, node, bias))
        H.append(h)
        o = [sigmoid(i) for i in h]
        O.append(o)
        input = o

    return H, O

def activation(input, node, bias=0):
    assert len(node) == len(input)
    return sum(wij*opi for wij, opi in zip(node, input)) + bias

# XXX: H is not needed here.
def errorsignals(pattern, W, H, O):
    # Outer layer
    D = [[sigmoiddiff(o)*(pattern[1] - o) for o in O[-1]]]

    for i, layer in reversed(list(enumerate(W))):
        if i == len(W) - 1:
            continue
        d = []
        for j, node in enumerate(layer):
            o = O[i][j]
            outputws = [W[i + 1][k][j] for k in xrange(len(W[i + 1]))]
            assert len(D[-1]) == len(outputws)
            d.append(sigmoiddiff(o)*sum(dd*w for dd, w in
                zip(D[-1], outputws)))
        D.append(d)

    D = list(reversed(D))
    return D

def inputOs(pattern, O):
    """
    Return a the outputs as inputs (including the pattern).
    """
    # TODO: restructure the loops in adaptweights so that we don't need
    # to create this huge redundant list
    iO = [[pattern[0] for i in xrange(NODES[0])]]
    for n, layer in zip(xrange(len(NODES[:-1])), O):
        # zip() will automatically skip the last output
        iO.append([[layer[i] for i in xrange(NODES[n])] for j in xrange(NODES[n + 1])])
    return iO

# XXX: Is this true any more
# alpha should be in [0, 1)
def adaptweights(W, D, pattern, O, oldW, B, eta=1, alpha=0.9):
    newW = []
    newB = []
    iO = inputOs(pattern, O)

    for wlayer, dlayer, olayer, wolayer, blayer in zip(W, D, iO, oldW, B):
        neww = []
        newb = []
        for wnode, d, onode, wonode, bias in zip(wlayer, dlayer, olayer, wolayer, blayer):
            neww.append([w + eta*d*o + alpha*(w - wo)
                for w, wo, o in zip(wnode, wonode, onode)])
            newb.append(bias + eta*d)
        newW.append(neww)
        newB.append(newb)

    return newW, newB

if __name__ == "__main__":
    main()
    sys.exit(0)
