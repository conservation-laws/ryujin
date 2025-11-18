<img align="right" height="150" src="doc/logo.png">

ryujin
======

Ryujin is a high-performance high-order collocation-type finite-element
solver for conservation equations such as the compressible Navier-Stokes
and Euler equations of gas dynamics. The solver is based on the [convex
limiting technique](https://doi.org/10.1137/17M1149961) to ensure
[invariant domain preservation](https://doi.org/10.1137/16M1074291) and
uses the finite element library [deal.II](https://github.com/dealii/dealii)
([website](https://www.dealii.org)) and the [vector class SIMD
library](https://github.com/vectorclass/version2). As such the solver
maintains important physical invariants and is guaranteed to be stable
without the use of ad-hoc tuning parameters.

Ryujin is freely available under the terms of the
[Apache License 2.0](https://spdx.org/licenses/Apache-2.0.html)
with [LLVM Exception](https://spdx.org/licenses/LLVM-exception.html).
Part of the contributed source code, third-party dependencies and header
libraries are covered by different open source licenses. For details
consult [COPYING.md](COPYING.md). Contributions to the ryujin source code
are governed by the [Developer Certificate of Origin version
1.1](https://developercertificate.org/); see
[CONTRIBUTING.md](CONTRIBUTING.md) for details.


Modules
-------

Ryujin features the following equation modules selectable by the following
parameter flags:
 - `equation = euler`, an optimized solver module for the
   [compressible Euler
   equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics))
   with polytropic equation of state.
 - `equation = euler aeos`, a generalized solver module for the
   compressible Euler equation with an [arbitrary or tabulated equation of
   state](https://en.wikipedia.org/wiki/Equation_of_state).
 - `equation = navier stokes`, an optimized solver module for the
   [compressible Navier-Stokes
   equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations)
   with polytropic equation of state,
   Newtonian fluid model, and Fourier's law for the heat flux.
 - `equation = shallow water`, a module for the [shallow water
   equations](https://en.wikipedia.org/wiki/Shallow_water_equations).
 - `equation = scalar conservation`, a module for scalar conservation
   equations with user-supplied flux. The module features a greedy
   wave-speed estimate to maintain an invariant domain, a generic indicator
   based on the entropy-viscosity commutator technique with a general,
   entropy-like function, and a customizable convex limiter.

Resources
---------

 - [Website](https://conservation-laws.org/)
 - [Installation instructions](./INSTALLATION.md)
 - [Usage instructions](./USAGE.md)
 - [Doxygen documentation](https://conservation-laws.org/ryujin/doxygen)

Videos
------

A number of simulation results can be found on [this youtube
channel](https://www.youtube.com/@matthiasmaier8956).

[<img src="https://img.youtube.com/vi/ig7R3yA7CtE/maxresdefault.jpg" width="400"/>](https://www.youtube.com/watch?v=ig7R3yA7CtE)
[<img src="https://img.youtube.com/vi/yM2rT3teakE/maxresdefault.jpg" width="400"/>](https://www.youtube.com/watch?v=yM2rT3teakE)
[<img src="https://img.youtube.com/vi/xIwJZlsXpZ4/0.jpg" width="400"/>](https://www.youtube.com/watch?v=xIwJZlsXpZ4)
[<img src="https://img.youtube.com/vi/pPP26zelb0M/0.jpg" width="400"/>](https://www.youtube.com/watch?v=pPP26zelb0M)
[<img src="https://img.youtube.com/vi/vBCRAF_c8m8/0.jpg" width="400"/>](https://www.youtube.com/watch?v=vBCRAF_c8m8)
[<img src="https://img.youtube.com/vi/xecIZylotSE/0.jpg" width="400"/>](https://www.youtube.com/watch?v=xecIZylotSE)

There original videos are available
[here](https://tamiko.43-1.org/developer/) ([license
details](https://tamiko.43-1.org/developer/COPYING.txt)).

References
----------

If you use this software for an academic publication please consider citing
some of the following references:

```
@article{ryujin-2021-1,
  author  = {Matthias Maier and Martin Kronbichler},
  title   = {Efficient parallel 3D computation of the compressible Euler equations with an invariant-domain preserving second-order finite-element scheme},
  doi     = {10.1145/3470637},
  url     = {https://arxiv.org/abs/2007.00094},
  journal = {ACM Transactions on Parallel Computing},
  year    = {2021},
  volume  = {8},
  number  = {3},
  pages   = {16:1-30}
}

@article{ryujin-2021-2,
  author  = {Jean-Luc Guermond and Matthias Maier and Bojan Popov and Ignacio Tomas},
  title   = {Second-order invariant domain preserving approximation of the compressible Navier--Stokes equations},
  doi     = {10.1016/j.cma.2020.113608},
  url     = {https://arxiv.org/abs/2009.06022},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year    = {2021},
  volume  = {375},
  number  = {1},
  pages   = {113608}
}

@article{ryujin-2021-3,
  author  = {Jean-Luc~Guermond and Martin Kronbichler and Matthias Maier and Bojan Popov and Ignacio Tomas},
  title   = {On the implementation of a robust and efficient finite element-based parallel solver for the compressible Navier-stokes equations},
  url     = {https://arxiv.org/abs/2106.02159},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year    = {2022},
  volume  = {389},
  pages   = {114250}
}

@article{ryujin-2023-4,
  author  = {Bennett Clayton and Jean-Luc Guermond and Matthias Maier and Bojan Popov and Tovar, Eric J.},
  title   = {Robust second-order approximation of the compressible Euler equations with an arbitrary equation of state},
  url     = {http://arxiv.org/abs/2207.12832},
  journal = {Journal of Computational Physics},
  pages   = {111926},
  year    = {2023}
}

@article{ryujin-2024-5,
  author = {Jean-Luc Guermond and Matthias Maier and Bojan Popov and Laura Saavedra and Ignacio Tomas},
  title = {First-Order Greedy Invariant-Domain Preserving Approximation for Hyperbolic Problems: Scalar Conservation Laws, and p-System},
  url = {https://arxiv.org/abs/2310.01713},
  journal = {Journal of Scientific Computing},
  year = {2024},
  volume = {100},
  number = {46},
  pages = {},
}

@article{ryujin-2025-6,
  author = {Jean-Luc Guermond and Matthias Maier and Tovar, Eric J.},
  title = {A high-order explicit Runge-Kutta approximation technique for the shallow water equations},
  url = {https://arxiv.org/abs/2403.17123},
  journal = {Computers \& Fluids},
  year = {2025},
  volume = {288},
  pages = {106493},
}

@article{ryujin-2025-7,
  author = {Martin Kronbichler and Matthias Maier and Ignacio Tomas},
  title = {Graph-based methods for hyperbolic systems of conservation laws using discontinuous space discretizations},
  url = {https://arxiv.org/abs/2402.04514},
  journal = {Communications in Computational Physics},
  year = {2025},
  volume = {38},
  pages = {74--108}
}

@article{ryujin-2025-8,
  author = {Jake Harmon and Martin Kronbichler and Matthias Maier and Eric Tovar},
  title = {A conservative invariant-domain preserving projection technique for hyperbolic systems under adaptive mesh refinement},
  url = {https://arxiv.org/abs/2507.18717},
  year = {2025},
  journal = {submitted}
}
```

Contact
-------

For questions either open an
[issue](https://github.com/conservation-laws/ryujin/issues), or contact
Matthias Maier (maier@tamu.edu).

Developers
----------

 - Martin Kronbichler ([@kronbichler](https://github.com/kronbichler)), Ruhr University Bochum, Germany
 - Matthias Maier ([@tamiko](https://github.com/tamiko)), Texas A&M University, TX, USA
 - Ignacio Tomas ([@nachosaurus](https://github.com/nachosaurus)), Texas Tech University, TX, USA
 - Eric Tovar ([@ejtovar](https://github.com/ejtovar)), Xcimer Energy Corporation, USA

Contributors
------------

 - Wolfgang Bangerth [@bangerth](https://github.com/bangerth), Colorado State University
 - Taylor Boylan ([@tmboylan](https://github.com/tmboylan)),
 - Jerett Cherry ([@jerret-cc](https://github.com/jerett-cc)), Colorado State University
 - Gregory Christian ([@gregorychristian](https://github.com/gregorychristian), Imperial College London, UK
 - Bennett Clayton ([@bgclayto](https://github.com/bgclayto))
 - Seth Gerberding ([@Gerbeset](https://github.com/Gerbeset)), Texas A&M University, TX, USA
 - Jake Harmon ([@harmonj](https://github.com/harmonj)), Los Alamos National Laboratory, USA
 - Jordan Hoffart ([@jordanhoffart](https://github.com/jordanhoffart)), Texas A&M University, TX, USA
 - David Pecoraro ([@ChrisPec27](https://github.com/ChrisPec27)), Texas A&M University, TX, USA
 - Madison Sheridan ([@Helblindi](https://github.com/Helblindi)), Texas A&M University, TX, USA
