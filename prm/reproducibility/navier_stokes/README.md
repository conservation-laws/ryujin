Navier-Stokes equations: reproducibility
========================================

This directory contains parameter files for the numerical results of

> [2021-GMPT] J.-L. Guermond, M. Maier, B. Popov, I. Tomas. Second-order
> invariant domain preserving approximation of the compressible
> Navier-Stokes equations. Comput. Methods Appl. Mech. Engrg., 375:113608,
> 2021. [arXiv:2009.06022](https://arxiv.org/abs/2009.06022)

> [2022-GKMPT] J.-L. Guermond, M. Kronbichler, M. Maier, B. Popov,
> I. Tomas. On the implementation of a robust and efficient finite
> element-based parallel solver for the compressible Navier-Stokes
> equations. Comput. Methods Appl. Mech. Engrg., 389:114250, 2022.
> [arXiv:2106.02159](https://arxiv.org/abs/2106.02159)

The file names start with the year and the author initials of the paper,
followed by the number of the section. Each parameter file has a header
comment that names the table or figure it corresponds to and summarizes
the problem setup. Run a configuration with

```
mpirun -np <n> ryujin 2022-GKMPT-6.3-daru_tenaud_shocktube.prm
```

All configurations use the `navier stokes` equation module (except for
the inviscid non-reflecting boundary test, which uses `euler`) and should
be run with double precision (`NUMBER=double`).

The original parameter files of [2022-GKMPT] were archived on Zenodo; the
files here are reconstructions from the paper and from parameter files
found in the git history of ryujin.

[2022-GKMPT]
------------

| File                                       | Paper                 | Problem                                                  | Check                                                                                                                              |
| ------------------------------------------ | --------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `2022-GKMPT-6.1-becker_solution-2d.prm`    | Table 1               | Becker viscous shock, 2D                                 | levels 7–9 reproduce Table 1 to 3 digits (level 6 within 10%); the δ2 entries of rows 2 and 3 of the table read e-5 instead of e-4 |
| `2022-GKMPT-6.2-non_reflecting_vortex.prm` | Figure 1              | Vortex leaving through non-reflecting boundaries (Euler) | cases (i), (iii): delta_1 drops from 0.75 to 1e-2 after the vortex has left (t ≈ 1.5)                                              |
| `2022-GKMPT-6.3-daru_tenaud_shocktube.prm` | Figures 2, 6, Table 2 | Daru-Tenaud shocktube, uniform meshes L11-L14            | level 9 at t = 1: density field as in Figure 2; C_f not compared                                                                   |
| `2022-GKMPT-6.4-oat15a-3d.prm`             | Figure 3              | ONERA OAT15a airfoil, 3D, 274M gridpoints                | runs (level 2, 1 and 6 ranks); not compared                                                                                        |

The scaling tests of Section 6.5 (Figures 4, 5) use the OAT15a and the
shocktube configurations with varying mesh refinement and a fixed number
of time steps (`max cycles = 1000`), see the header of
`2022-GKMPT-6.4-oat15a-3d.prm`.

[2021-GMPT]
-----------

The results of [2021-GMPT] were computed with a serial P1 demonstration
code, not with ryujin. The following files set up the same problems in
ryujin.

| File                                      | Paper        | Problem                                                                     | Check                                                                                                                                                                      |
| ----------------------------------------- | ------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2021-GMPT-7.2-becker_solution-1d.prm`    | Table 1      | Becker viscous shock, 1D                                                    | the columns of Table 1 appear permuted: delta_1 matches our L2 within 8% and delta_2 our Linf within 10% (30% at I = 200; levels 0–6); Table 2 (2D) has the expected order |
| `2021-GMPT-7.3-becker_solution-2d.prm`    | Table 2      | Becker viscous shock, 2D, uniform quadrilaterals instead of Delaunay meshes | levels 4–8 (1617–394497 points): CFL 0.4 within 0.67–1.31× of Table 2 in all norms; CFL 0.9 2–2.7× above on coarse meshes, 0.7–1.2× on fine meshes; rates approach 2       |
| `2021-GMPT-7.4-daru_tenaud_shocktube.prm` | Figures 1, 2 | Daru-Tenaud shocktube, graded mesh, viscosity sweep                         | level 9 at t = 1 (0.5M points): density field as in Figure 1(i)                                                                                                            |

Not reproduced
--------------

 - [2022-GKMPT] Section 6.2, "Riemann S." boundary condition of §4.3.1:
   only the characteristic variant (`dynamic`) is implemented.
 - [2022-GKMPT] Figure 3, slip conditions on the z-planes: the airfoil
   geometry uses periodic faces (as did the code at the time).
 - [2021-GMPT] Sections 7.3, 7.4: the non-nested Delaunay and hand-graded
   triangular meshes of the paper are not available.
