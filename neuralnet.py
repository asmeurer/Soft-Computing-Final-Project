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

NODES = [10, 4, 1]
eps = 0.1
eta = 1

def main():
    problems = get_problems()
    W = init_weights()
    while objective(patterns) < eps:
        for pattern in patterns:
            break


def get_problems():
    with open("problems", 'r') as file:
        problemstxt = file.read()

    problems = []

    for line in problemstxt.split('\n'):
        if not line:
            # Handle blank lines at the end of the file
            continue

        set, has_subsets = eval(line)
        problems.append((set, int(has_subsets)))

    return problems

def init_weights(seed=None, max=1):
    if seed:
        random.seed(seed)

    W = []
    for nodelen in NODES:
        W.append([random.random()*max for i in xrange(nodelen)])

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

def activation(pattern, W, layer, node):
    if layer == -1:
        return sum(wij*opi for wij, opi in zip(W[layer], pattern[0])
    return sum(wij*opi for wij, opi in zip(

def errorsignal(pattern, W, layer, node):
    if layer == len(NODES) - 1:
        # Outer layer
        o = output(pattern, W, layer, node)
        return sigmoiddiff(o)*(pattern[1] - o)

if __name__ == "__main__":
    main()
    sys.exit(0)
