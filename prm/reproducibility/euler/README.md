Euler equations: reproducibility
================================

This directory contains parameter files for numerical results of the
following publications that use the `euler` equation module (compressible
Euler equations with polytropic gas equation of state):

> [2018-GNPT] J.-L. Guermond, M. Nazarov, B. Popov, I. Tomas. Second-order
> invariant domain preserving approximation of the Euler equations using
> convex limiting. SIAM J. Sci. Comput., 40(5):A3211–A3239, 2018.
> [DOI:10.1137/17M1149961](https://doi.org/10.1137/17M1149961)

> [2021-MK] M. Maier, M. Kronbichler. Efficient parallel 3D computation of
> the compressible Euler equations with an invariant-domain preserving
> second-order finite-element scheme. ACM Trans. Parallel Comput.,
> 8(3):16:1–30, 2021. [arXiv:2007.00094](https://arxiv.org/abs/2007.00094)

> [2024-GMPST] J.-L. Guermond, M. Maier, B. Popov, L. Saavedra, I. Tomas.
> First-order greedy invariant-domain preserving approximation for
> hyperbolic problems: scalar conservation laws, and p-system. J. Sci.
> Comput., 100:46, 2024.
> [arXiv:2310.01713](https://arxiv.org/abs/2310.01713)

> [2025-KMT] M. Kronbichler, M. Maier, I. Tomas. Graph-based methods for
> hyperbolic systems of conservation laws using discontinuous space
> discretizations. Commun. Comput. Phys., 38:74–108, 2025.
> [arXiv:2402.04514](https://arxiv.org/abs/2402.04514)

> [2026-HKMT] J. J. Harmon, M. Kronbichler, M. Maier, E. J. Tovar. A
> conservative invariant-domain preserving projection technique for
> hyperbolic systems under adaptive mesh refinement. Preprint, 2025.
> [arXiv:2507.18717](https://arxiv.org/abs/2507.18717)

The file names start with the key of the publication followed by the
section number. Each parameter file has a header comment that describes the
problem, names the figure or table it corresponds to, and explains how to
obtain the other refinement levels and variants. Run a configuration with

```
mpirun -np <n> ryujin 2025-KMT-6.2-isentropic_vortex.prm
```

All configurations should be run with double precision (`NUMBER=double`).
The files are reconstructions; differences to the published numbers are
expected. The "Check" column summarizes a short test run with the current
code.

Contents
--------

| File                                                     | Paper                                | Problem                                        | Check                                                                                                                                                                                                                                                    |
| -------------------------------------------------------- | ------------------------------------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2018-GNPT-5.8-mach3_forward_facing_step.prm`            | [2018-GNPT] Suppl. Fig. 1            | Mach 3 forward facing step                     | level 2 at t = 4: shock pattern as in the standard reference solution (the paper PDF was not available)                                                                                                                                                  |
| `2018-GNPT-5.9-mach10_double_mach_reflection.prm`        | [2018-GNPT] Suppl. Fig. 2            | Mach 10 double Mach reflection                 | level 6 at t = 0.2: Mach stem, triple points and jet as in the standard reference solution (the paper PDF was not available)                                                                                                                             |
| `2021-MK-1-mach3_cylinder-2d.prm`                        | [2021-MK] Fig. 1, [2018-GNPT] Fig. 3 | Mach 3 flow past a cylinder, 2D                | level 7 at t = 3.5: bow shock, wall reflections and wake shocks at the positions of Fig. 1                                                                                                                                                               |
| `2021-MK-5-mach3_cylinder-3d.prm`                        | [2021-MK] Fig. 6                     | Mach 3 flow past a cylinder, 3D                | runs (level 2, t = 0.2); coarse mesh has 208 points as in [2021-MK]                                                                                                                                                                                      |
| `2021-MK-5-mach3_cylinder-3d-scaling.prm`                | [2021-MK] Tables 2–6, Fig. 7         | Same, fixed number of steps, no output         | runs (level 2, 434 steps)                                                                                                                                                                                                                                |
| `2024-GMPST-8-woodward_colella_blast-first_order.prm`    | [2024-GMPST] Fig. 6 ("GMS")          | Woodward–Colella blast, first order            | the rarefaction left of x = 0.65 agrees with the GMS curves (I = 400–1600), but the density peak is much lower: 4.9 at I = 400 against about 6.5 in Fig. 6 (6.3 only at I = 25600)                                                                       |
| `2025-KMT-6.2-isentropic_vortex.prm`                     | [2025-KMT] Tables 1, 5               | Isentropic vortex, dG Q1–Q3                    | with the `evc factor` of the header and `relaxation factor = 4`: Q2 and Q3 reproduce the rates of Table 1, L1 errors 6–25% below (levels 0–3); Q1 within 11% on levels 0–2, but rate about 1.6 instead of 1.8–1.9; first order matches Table 5 within 5% |
| `2025-KMT-6.2-rarefaction.prm`                           | [2025-KMT] Table 2                   | Rarefaction wave, 1D                           | L1 errors (levels 0–7): Q1 within 17% of Table 2 (2% on levels 4–7), mean rate 1.70 vs 1.71; Q2 6–30% above; Q3 1.25–1.8× above                                                                                                                          |
| `2025-KMT-6.2-leblanc.prm`                               | [2025-KMT] Table 3                   | LeBlanc shock tube, 1D                         | L1 errors (levels 0–7): Q1 4–21% above Table 3, Q2 −6% to +31%, Q3 25–64% above; mean rates 0.87, 0.73, 0.78 vs 0.84, 0.80, 0.81                                                                                                                         |
| `2025-KMT-6.3-isentropic_vortex-boundary_conditions.prm` | [2025-KMT] Table 4                   | Vortex leaving the domain, strategies (a), (b) | L1 errors 2–9× larger than Table 4, rates 1.1–1.9 vs 2.1–2.3 (strategies (a), (b), M = 1–2.5, levels 0–3); see below                                                                                                                                     |
| `2025-KMT-6.4-mach3_cylinder-dG.prm`                     | [2025-KMT] Fig. 1                    | Mach 3 flow past a cylinder, dG Q1–Q3          | Q1 level 7 at t = 4: shock pattern and wake as in Fig. 1(a)                                                                                                                                                                                              |
| `2026-HKMT-6.3.1-astrophysical_jet.prm`                  | [2026-HKMT] Fig. 4                   | Mach 4000 jet with adaptive mesh refinement    | runs (max. level 8 up to t = 3e-5); not compared                                                                                                                                                                                                         |

The boundary conditions have been changed since the computations of
[2025-KMT] were done; this affects the boundary condition test.

Not reproduced
--------------

 - [2018-GNPT], Sections 5.1–5.6: convergence tests and Riemann problems
   computed with Codes 1 and 2 (P1 elements); not part of this directory.
 - [2024-GMPST], Figure 6 "Greedy" curves: the greedy wave speed estimate is
   not available for the Euler equations.
 - [2025-KMT], Table 4 strategies (c) and (d): sub/supersonic boundary data
   from a far-field state or from the previous time step is not available;
   the `dynamic` boundary condition always uses the initial-state function.
 - [2025-KMT]: the "SSPRK-54" scheme is replaced by `erk 54`.

Configurations that need extra steps
------------------------------------

 - `2021-MK-5-mach3_cylinder-3d.prm` (1.8B gridpoints) and
   `2021-MK-1-mach3_cylinder-2d.prm` (38M gridpoints) are very large
   computations; lower the `mesh refinement` for a quick run.
 - `2026-HKMT-6.3.1-astrophysical_jet.prm` adapts the mesh every cycle,
   which dominates the run time; set the `trigger multiplier` of the mesh
   adaptor to 10 for a cheaper run.
