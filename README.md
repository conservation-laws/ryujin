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

Resources
---------

Documentation, installation and usage instructions, references, and
contact information can be found on the project website:

 - [Website](https://conservation-laws.org/ryujin/)
 - [Installation](https://conservation-laws.org/ryujin/documentation/installation/)
 - [Usage](https://conservation-laws.org/ryujin/documentation/usage/)
 - [API reference (Doxygen)](https://conservation-laws.org/ryujin/doxygen/)
 - [About, license, and references](https://conservation-laws.org/ryujin/about/)
 - [Team](https://conservation-laws.org/ryujin/team/)
