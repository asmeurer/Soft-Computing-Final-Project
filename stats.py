#!/usr/bin/env python
"""
Compute statistics from the outputs of the neural network.

This uses numpy to compute the mean and standard deviation and scipy to
calculate the t-test for comparing the means of the accuracy and the
output per test pattern.
"""

from __future__ import division
import argparse
import sys

from numpy import mean, std
from scipy.stats import ttest_1samp

parser = argparse.ArgumentParser(description="This computes statistics "
    "from the outputs of the neural network.")

parser.add_argument('filename', metavar='FILE', type=str,
                    help="Name of the file of outputs.")

args = parser.parse_args()

def main(args):
    outputs = getoutputs(args.filename)

    # Create a dictionary of dictionaries for each data attribute
    # that has a list of the corresponding attributes.
    # For example, bydict['NODES'][(30, 30, 30, 1)]['accuracy'] will give you
    # a list of all the accuracies for outputs with NODES == [30, 30, 30, 1]
    bydict = {}
    for key in outputs[0]:
        bydict[key] = {}
    for outputdict in outputs:
        for key1 in outputdict:
            for key2 in outputdict:
                if key2 == key1:
                    continue
                o = outputdict[key1]
                if isinstance(o, list):
                    o = tuple(o)
                if o not in bydict[key1]:
                    bydict[key1][o] = {key2: [outputdict[key2]]}
                else:
                    if key2 not in bydict[key1][o]:
                        bydict[key1][o][key2] = [outputdict[key2]]
                    else:
                        bydict[key1][o][key2].append(outputdict[key2])

    ###################################################
    print "Stat 1: accuracy by NODES"
    print "-------------------------"
    print "NODES, mean accuracy, std accuracy, mean +/- 2*std"
    for NODES in sorted(bydict['NODES'].keys()):
        accuracies = bydict['NODES'][NODES]['accuracy']
        mean_accuracies = mean(accuracies)
        std_accuracies = std(accuracies)
        print NODES, mean_accuracies, std_accuracies,
        print [mean_accuracies - 2*std_accuracies, mean_accuracies + 2*std_accuracies]
    print

    ###################################################
    print "Stat 2: Average output of the test patterns"
    print "-------------------------------------------"
    mean_ones = []
    diffs = []
    for NODES in bydict['NODES']:
        for trueneg, falseneg, truepos, falsepos, accuracy in zip(
            bydict['NODES'][NODES]['trueneg'],
            bydict['NODES'][NODES]['falseneg'],
            bydict['NODES'][NODES]['truepos'],
            bydict['NODES'][NODES]['falsepos'],
            bydict['NODES'][NODES]['accuracy']):

                m = (truepos + falsepos)/(trueneg + falseneg + truepos + falsepos)
                mean_ones.append(m)
                diff = m - accuracy
                diffs.append(diff)

    mean_mean_ones = mean(mean_ones)
    std_mean_ones = std(mean_ones)
    t, prob = ttest_1samp(diffs, 0)
    print "mean ones, std, mean +/- 2*std, t, prob, mean diffs, std diffs"
    print mean_mean_ones, std_mean_ones,
    print [mean_mean_ones - 2*std_mean_ones, mean_mean_ones + 2*std_mean_ones],
    print t, prob, mean(diffs), std(diffs)
    print

    ###################################################
    print "Stat 3: convergence by NODES"
    print "---------------------------"
    print "NODES, mean convergences"
    for NODES in sorted(bydict['NODES'].keys()):
        convergences = map(int, bydict['NODES'][NODES]['converged'])
        mean_convergences = mean(convergences)
        print NODES, mean_convergences
    print

    ###################################################
    print "Stat 4: Epochs per NODES (for converged)"
    print "----------------------------------------"
    print "NODES, mean epochs, std epochs, mean +/- 2*std"
    for NODES in sorted(bydict['NODES'].keys()):
        epochs = []
        for i, converged in enumerate(bydict['NODES'][NODES]['converged']):
            if converged:
                epochs.append(bydict['NODES'][NODES]['epochs'][i])
        if not epochs:
            print NODES, "--", "--", "[--, --]"
            continue
        mean_epochs = mean(epochs)
        std_epochs = std(epochs)
        print NODES, mean_epochs, std_epochs,
        print [mean_epochs - 2*std_epochs, mean_epochs + 2*std_epochs]
    print

    return


def getoutputs(filename):
    with open(filename) as file:
        outputtxt = file.read()

    outputs = []

    for line in outputtxt.split('\n'):
        if not line:
            # Handle blank lines at the end of the file
            continue

        outputs.append(eval(line))

    return outputs

if __name__ == "__main__":
    main(args)
    sys.exit(0)

