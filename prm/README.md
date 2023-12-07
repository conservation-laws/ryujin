<img align="right" height="150" src="../doc/logo.png">

Parameter files
===============

This directory contains a collecting of parameter files ranging from
airfoils, to various published benchmark configurations to validation
configurations with test vectors.


Verification
------------

The `./verification` subdirectory contains parameter files and
<i>baseline</i> output vectors that document expected error and convergence
rates for various analytical solutions. All configurations compare the
simulation result at final time to a known analytic solution and record the
normalized L1, L2, and L\infty error norms (summed up over all components).

All test configurations should be run with double floating point precision.

Benchmarks
----------

The `./benchmarks` directory contains parameter files for well known and
popular <i>benchmark</i> configurations. These configurations typically do
not have an analytical solution, but the expected solution structure is
well known. They are thus usually compared in the <i>eyeball norm</i>.
