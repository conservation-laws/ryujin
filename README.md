<img align="right" height="150" src="doc/logo.png">

ryujin
======

Ryujin is a high-performance second-order colocation-type finite-element
scheme for solving the compressible Euler equations of gas dynamics on
unstructured meshes. The solver is based on the convex limiting technique
introduced by [Guermond et al.](https://doi.org/10.1137/17M1149961) and
uses the finite element library [deal.II](https://github.com/dealii/dealii)
([website](https://www.dealii.org)).

As such it is <i>invariant-domain preserving</i>, the solver maintains
important physical invariants and is guaranteed to be stable without the
use of ad-hoc tuning parameters.

Ryujin is freely available under the terms of the [MIT license](COPYING.md).

References
----------

If you use this software for an academic publications please cite the
following reference:

```
@article{ryujin2020,
  title   = {Massively parallel 3D computation of the compressible Euler
      equations with an invariant-domain preserving second-order
      finite-element scheme}
  author  = {Matthias Maier and Martin Kronbichler},
  year    = {2020},
  journal = {submitted}
}
```

Resources
---------

A detailed doxygen documentation can be found
[here](https://conservation-laws.43-1.org/doxygen). The doxygen
documentation contains detailed
[Installation](https://conservation-laws.43-1.org/doxygen/Installation.html)
and [Usage](https://conservation-laws.43-1.org/doxygen/Usage.html)
instructions. General information about the overarching software project
can be found [here](https://conservation-laws.43-1.org/).

Contact
-------

For questions please contact Matthias Maier <maier@math.tamu.edu> and
Martin Kronbichler <kronbichler@lnm.mw.tum.de>.

Authors
-------

Martin Kronbichler ([@kronbichler](https://github.com/kronbichler)), Technical University of Munich, Germany
Matthias Maier ([@tamiko](https://github.com/tamiko)), Texas A&M University, TX, USA
Ignacio Tomas ([@itomasSNL](https://github.com/itomasSNL)), Texas A&M University, Sandia National Laboratories
Eric Tovar ([@ejtovar](https://github.com/ejtovar)), Texas A&M University, TX, USA
