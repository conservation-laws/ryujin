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

All test configurations should be rund with double floating point
precision; a majority of test configurations is for 1D problems for which
it is best to recompile with the compile-time option `dim` set to `1`
(e.g., via `cmake -Ddim=1 .`, or changing the value via `ccmake`).

Benchmarks
----------

The `./benchmarks` directory contains parameter files for well known and
popular <i>benchmark</i> configurations. These configurations typically do
not have an analytical solution, but the expected result and structure of
solutions is well known. They are thus usually compared in the <i>eyeball
norm</i>.
