<img align="right" height="150" src="doc/logo.png">

ryujin
======

Ryujin is a high-performance second-order collocation-type finite-element
scheme for solving the compressible Navier-Stokes and Euler equations of
gas dynamics on unstructured meshes. The solver is based on the convex
limiting technique introduced by [Guermond et
al.](https://doi.org/10.1137/17M1149961) and uses the finite element
library [deal.II](https://github.com/dealii/dealii)
([website](https://www.dealii.org)).

As such it is <i>invariant-domain preserving</i>, the solver maintains
important physical invariants and is guaranteed to be stable without the
use of ad-hoc tuning parameters.

Ryujin is freely available under the terms of the [MIT license](COPYING.md).

Resources
---------

 - [Installation instructions](https://conservation-laws.org/ryujin/doxygen/Installation.html)
 - [Usage instructions](https://conservation-laws.org/ryujin/doxygen/Usage.html)
 - [Doxygen documentation](https://conservation-laws.org/ryujin/doxygen)
 - [Website](https://conservation-laws.org/)

Videos
------

<h4>Compressible Euler flow (Mach 3) past a cylinder: (click on image)</h4>

[<img src="https://img.youtube.com/vi/pPP26zelb0M/0.jpg" width="400"/>](https://www.youtube.com/watch?v=pPP26zelb0M)
[<img src="https://img.youtube.com/vi/vBCRAF_c8m8/0.jpg" width="400"/>](https://www.youtube.com/watch?v=vBCRAF_c8m8)

References
----------

If you use this software for an academic publications please cite the
following references ([preprint](https://arxiv.org/abs/2007.00094),
[preprint](https://arxiv.org/abs/2009.06022)):

```
@article {ryujin-2021-a,
  author = {Matthias Maier and Martin Kronbichler},
  title = {Efficient parallel 3D computation of the compressible Euler equations
    with an invariant-domain preserving second-order finite-element scheme},
  url = {https://arxiv.org/abs/2007.00094},
  journal = {ACM Transactions on Parallel Computing},
  year = {2021},
  volume = {accepted},
  pages = {},
}

@article{ryujin-2021-b,
  author = {Jean-Luc Guermond and Matthias Maier and Bojan Popov and Ignacio Tomas},
  title = {Second-order invariant domain preserving approximation of the compressible Navier--Stokes equations},
  doi = {10.1016/j.cma.2020.113608},
  url = {https://arxiv.org/abs/2009.06022},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year = {2021},
  volume  = {375},
  number = {1},
  pages = {113608},
}
```

Contact
-------

For questions please contact Matthias Maier <maier@math.tamu.edu> and
Martin Kronbichler <kronbichler@lnm.mw.tum.de>.

Authors
-------

 - Martin Kronbichler ([@kronbichler](https://github.com/kronbichler)), Technical University of Munich, Germany
 - Matthias Maier ([@tamiko](https://github.com/tamiko)), Texas A&M University, TX, USA
 - Ignacio Tomas ([@itomasSNL](https://github.com/itomasSNL)), Texas A&M University, Sandia National Laboratories
 - Eric Tovar ([@ejtovar](https://github.com/ejtovar)), Texas A&M University, TX, USA
