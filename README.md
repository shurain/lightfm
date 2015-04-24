LightFM
=======

An implementation of Factorization Machines. Intended to be educational.

Current implementation does not use any fancy tricks to speed up the prediction
and learning phase.

Currently, LightFM does not take any command line arguments. You have to modify
the source code if you want to change the behavior of the program in any way.

Weights are optimized using stochastic gradient descent. It only looks at a
single training example and does not work in full-batch or mini-batch mode.

## Data

LightFM follows LibSVM format.

    target feature:weight feature:weight

First term is the target which is followed by features. An example is given
below.

    3.5 54081:1 843:1
    3 49752:1 324:1
    4.5 60777:1 1011:1

## Build

    make

## Test

    make test

## Execution

Currently, LightFM does not take any command line arguments. You have to modify
the source code to change the input path.

    make run

or,

    bin/lightfm

## Reference

- Steffen Rendle (2010): [Factorization Machines](http://www.inf.uni-konstanz.de/~rendle/pdf/Rendle2010FM.pdf), in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia.
