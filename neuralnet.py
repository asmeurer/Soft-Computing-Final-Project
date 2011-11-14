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

# XXX: If you change the default number of layers, you have to update the help
#      text for -N below.
DEFAULT_NODES = [30, 20, 20, 1]

parser = argparse.ArgumentParser(description="Runs the neural network.")

parser.add_argument('-s', '--seed', metavar='N', type=int,
                    help="The random seed used to generate the initial weights.")
parser.add_argument('filename', metavar='FILE', type=str,
                    help="Name of the file to train the network with.")
parser.add_argument('-w', '--weights', metavar='FILE', type=str, default=None,
                    help="Weights file to initialize weights to (defaults "
                    "random initial weights).")
parser.add_argument('-t', '--test', metavar='FILE', type=str,
                    help="File of patterns to test the neural network on.")
parser.add_argument('-n', '--no-train', dest='notrain', action='store_true',
                    help="Do not attempt to train the network.")
parser.add_argument('-N', '--nodes', metavar='N', type=int, nargs='+',
                    help="The number of nodes to use per layer. This will be "
                    "created automatically if a weight file is provided. "
                    "Defaults to %d %d %d %d." % tuple(DEFAULT_NODES))
parser.add_argument('-e', '--eps', metavar='N', type=float, default=1,
                    help="The epsilon value to check convergence against. "
                    "Defaults to 1.")
parser.add_argument('-a', '--alpha', metavar='N', type=float, default=0.9,
                    help="The alpha value to use when adjusting the weights.")

args = parser.parse_args()

def main(args):
    seed = args.seed
    filename = args.filename
    global weights
    weights = args.weights
    test = args.test
    notrain = args.notrain
    nodes = args.nodes
    eps = args.eps
    alpha = args.alpha

    if not 0 <= alpha < 1:
        # TODO: Use ArgumentError
        raise argparse.ArgumentError(alpha, "alpha must be in [0, 1).")

    if nodes:
        if any(i <= 0 for i in nodes):
            raise ValueError("Nodes must all be positive.")
        if nodes[-1] != 1:
            raise NotImplementedError("Only one output node is supported.")
        NODES = nodes
    else:
        NODES = DEFAULT_NODES

    eta = 1
    a = 1
    b = 0.01
    patterns = get_problems(filename)
    patternlength = len(patterns[0][0])

    # Each weight corresponds to the inputs of the node.
    if not weights:
        B = [[1 for i in xrange(j)] for j in NODES]
        oldW = init_weights(patternlength, NODES, seed=seed, max=0)
        W = init_weights(patternlength, NODES, seed=seed, max=1)
    else:
        # Note, we don't check consistency among W, oldW, and B
        with open(weights) as file:
            vars = eval(file.read())

        B = vars['B']
        W = vars['W']
        oldW = vars['oldW']
        eta = vars['eta']
        if nodes:
            if NODES != [len(i) for i in B]:
                raise ArgumentError("The number of nodes given and the weight "
                    "file do not match.")
        else:
            NODES = [len(i) for i in B]

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
            oldW, (W, B) = W, adaptweights(W, D, pattern, O, oldW, B, NODES, eta, alpha)
        oldoldobj, oldobj, obj = oldobj, obj, objective(patterns, outputs)
        if True:
            if oldobj and oldoldobj:
                if obj - oldobj < 0 and oldobj < oldoldobj:
                    eta += a
                elif obj - oldobj > 0:
                    eta -= b*eta
        print obj, eta, epochs
        if obj < eps and all(round(outputs[pattern][-1][0]) == pattern[1]
            for pattern in patterns) or epochs == 1000:

            with open("weights.py", 'w') as file:
                file.write("{\n")
                file.write("'B': %s,\n\n" % B)
                file.write("'W': %s,\n\n" % W)
                file.write("'oldW': %s,\n\n" % oldW)
                file.write("'eta': %s,\n" % eta)
                file.write("}\n")
            for pattern in patterns:
                print pattern[1], outputs[pattern][-1][0],
                if round(outputs[pattern][-1][0]) != pattern[1]:
                    print False
                else:
                    print
            if obj < eps:
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

def init_weights(patternlength, NODES, seed=None, max=1):
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

def inputOs(pattern, O, NODES):
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
def adaptweights(W, D, pattern, O, oldW, B, NODES, eta, alpha):
    newW = []
    newB = []
    iO = inputOs(pattern, O, NODES)

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
    main(args)
    sys.exit(0)
