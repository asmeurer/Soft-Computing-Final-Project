Neural Nework
=============

By Aaron Meurer.

This was the final project from my CSE 464 Soft Computing class.  It is a
neural network written in Python, which I attempted to use to solve the subset
sum problem.  See the report for more details.  In short, neural networks are
not suited to solve this problem (it is a NP-Complete problem), and I ended up
learning more about how to make neural networks and get them to converge to
input data than about solving this particular problem.

Info
====

The neuralnet.py file is the neural network.  To train it, run

    ./neuralnet.py patterns

where patterns is a file of patterns.  The patterns should be of the
format

    (1, 2, 3, 4), True
    (5, 6, 7, 8), False

Where the dimensionality of the tuple can be anything (but should be the
same for all patterns).  To generate patterns for the subset sum
problem, run

    ./generate_patterns.py patterns

To test the neural network on the file patterns1, run

    ./neuralnet.py patterns -t patterns1

After training, the weights are saved in the file weights.py.  You can
use this to test the network without retraining, but running

    ./neuralnet.py patterns -w weights.py -t patterns1

And it will start training from the patterns file with the weights in
the weights file.  If the weights have already converged, it will just
skip straight to training.

To do this automatically, run

    make

And the Makefile will generate two sets of patterns, train the network
on one of them, and test on the other.

There are more options to each of these functions.  Run

    ./neuralnet.py --help

and

    ./generate_patterns.py --help

to see them all.

Extensive data on the run of the neural network is saved to a file
output of the neural network is saved to the file `output` (new output
is appended so you can gather statistics on multiple runs)

To generate statistics on the outputs, run

    ./stats.py output

The files `output_presentation` and `output_report` are the outputs that I
used for the presentation and the report, respectively.

The files `or`, `xor`, and `two` are small test patterns used to test the
network.

The script `runall.sh` will run the neural network on eight different
architectures 50 times, as I did in the report.  This takes a very long
time to finish.
