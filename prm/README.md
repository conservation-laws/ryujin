<img align="right" height="150" src="../doc/logo.png">

Parameter files
===============

This directory contains verification configurations with test vectors,
the parameter files for the numerical results of publications, and
benchmarks for the throughput.


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

The `./benchmarks` subdirectory contains parameter files to assess the
throughput, i.e., the number of updated degrees of freedom per second
(reported as "MQ/s" on the terminal and in the log file). The
configurations run no tasks besides the time integration and the logger,
and the timer granularity equals the final time, so that output does not
distort the measurement:

 - `./benchmarks/euler-mach3-cylinder-2d.prm`: Mach 3 flow past a
   cylinder in 2D (compressible Euler equations, mesh refinement 8)

Reproducibility
---------------

The `./reproducibility` directory contains parameter files for the
numerical results of publications with and about ryujin, one subdirectory
per equation module. Each subdirectory has a `README.md` with the
references, a list of the configurations and how they compare to the
published results. File names start with the year and the author initials
of the publication, followed by the section number:

 - `./reproducibility/euler`: Mach 3 flow past a cylinder, forward facing
   step, double Mach reflection, discontinuous Galerkin tests, adaptive
   mesh refinement (2018-GNPT, 2021-MK, 2024-GMPST, 2025-KMT, 2026-HKMT)
 - `./reproducibility/euler_aeos`: arbitrary equations of state
   (2023-CGMPT, 2026-HKMT)
 - `./reproducibility/euler_atmospheric`: well-balanced atmospheric flows
   (2026-FMT)
 - `./reproducibility/euler_poisson`: Euler-Poisson and magnetized
   Euler-Poisson equations (2023-MST, 2026-HMST)
 - `./reproducibility/navier_stokes`: compressible Navier-Stokes equations
   (2021-GMPT, 2022-GKMPT)
 - `./reproducibility/scalar_conservation`: greedy wave speed estimates
   (2024-GMPST)
 - `./reproducibility/shallow_water`: high-order explicit Runge-Kutta
   schemes, adaptive mesh refinement (2025-GMT, 2026-HKMT)
