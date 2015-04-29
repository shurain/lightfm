LightFM
=======

An implementation of Factorization Machines. Intended to be educational.

Currently supports naive feature vectorization using maps and feature hashing
a.k.a. hashing tricks.

Weights are optimized using stochastic gradient descent. It only looks at a
single training example and does not work in full-batch or mini-batch mode.

## Data

LightFM expects the following input format.

    target feature:weight feature:weight

First term is the target which is followed by features. An example is given
below.

    3.5 p54081:1 i843:1
    3 p49752:1 i324:1
    4.5 p60777:1 i1011:1

## Build

    make

## Test

    make test

## Execution

LightFM takes various command line arguments. You need to specify the path for
training data and test data.

    bin/lightfm -d <training data path> -t <test data path>

Other arguments can be inspected by `--help` command.

    bin/lightfm --help

## Reference

- Steffen Rendle (2010): [Factorization Machines](http://www.inf.uni-konstanz.de/~rendle/pdf/Rendle2010FM.pdf), in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia.
- Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). [Feature Hashing for Large Scale Multitask Learning](http://arxiv.org/pdf/0902.2206.pdf) (ICML)
