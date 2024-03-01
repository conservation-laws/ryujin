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
[MIT license](https://spdx.org/licenses/MIT.html). Part of the source
code is dual licensed under the MIT and the
[BSD 3-clause license](https://spdx.org/licenses/BSD-3-Clause.html).
Third-party dependencies and header libraries are covered by different open
source licenses. For details consult [COPYING.md](COPYING.md).
Contributions to the ryujin source code are governed by the [Developer
Certificate of Origin version 1.1](https://developercertificate.org/); see
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
 - `equation = shallow water`, a module for the [shallow-water
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

References
----------

If you use this software for an academic publication please consider citing
the following references
([[1](https://arxiv.org/abs/2007.00094)],
[[3](https://arxiv.org/abs/2106.02159})]):

```
@article {ryujin-2021-1,
  author = {Matthias Maier and Martin Kronbichler},
  title = {Efficient parallel 3D computation of the compressible Euler
    equations with an invariant-domain preserving second-order
    finite-element scheme},
  doi = {10.1145/3470637},
  url = {https://arxiv.org/abs/2007.00094},
  journal = {ACM Transactions on Parallel Computing},
  year = {2021},
  volume = {8},
  number = {3},
  pages = {16:1-30},
}

@article{ryujin-2021-3,
  author = {Jean-Luc Guermond and Martin Kronbichler and Matthias Maier and
    Bojan Popov and Ignacio Tomas},
  title = {On the implementation of a robust and efficient finite
    element-based parallel solver for the compressible Navier-stokes
    equations},
  doi = {10.1016/j.cma.2021.114250},
  url = {https://arxiv.org/abs/2106.02159},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year = {2022},
  volume = {389},
  pages = {114250},
}
```

Contact
-------

For questions either open an
[issue](https://github.com/conservation-laws/ryujin/issues), or contact
Matthias Maier (maier@tamu.edu).

Developers
----------

 - Martin Kronbichler ([@kronbichler](https://github.com/kronbichler)), University of Augsburg, Germany
 - Matthias Maier ([@tamiko](https://github.com/tamiko)), Texas A&M University, TX, USA
 - David Pecoraro ([@ChrisPec27](https://github.com/ChrisPec27)), Texas A&M University, TX, USA
 - Ignacio Tomas ([@nachosaurus](https://github.com/nachosaurus)), Texas Tech University, TX, USA
 - Eric Tovar ([@ejtovar](https://github.com/ejtovar)), Los Alamos National Laboratory, USA
