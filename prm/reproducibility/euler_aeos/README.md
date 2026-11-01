Euler equations with arbitrary equation of state: reproducibility
=================================================================

This directory contains parameter files for the numerical results of

> [2023-CGMPT] B. Clayton, J.-L. Guermond, M. Maier, B. Popov,
> E. J. Tovar. Robust second-order approximation of the compressible Euler
> equations with an arbitrary equation of state. J. Comput. Phys.,
> 478:111926, 2023. [arXiv:2207.12832](https://arxiv.org/abs/2207.12832)

> [2026-HKMT] J. J. Harmon, M. Kronbichler, M. Maier, E. J. Tovar. A
> conservative invariant-domain preserving projection technique for
> hyperbolic systems under adaptive mesh refinement. Preprint, 2025.
> [arXiv:2507.18717](https://arxiv.org/abs/2507.18717)

The file names start with the year and the author initials of the paper,
followed by the number of the section. Each parameter file has a header
comment that names the table or figure it reproduces and summarizes the
problem setup. Run a configuration with

```
mpirun -np <n> ryujin 2023-CGMPT-5.6-shock_bubble-jwl.prm
```

All configurations use the `euler aeos` equation module and should be run
with double precision (`NUMBER=double`). The original parameter files were
not archived; the files here are reconstructions from the papers. The
"Check" column summarizes how close a short run with the current ryujin
comes to the published result.

[2023-CGMPT]
------------

The 1D results (Sections 5.3.1, 5.4.1) were computed with a separate
Fortran code; the corresponding files are ryujin counterparts. The 2D
results were computed with ryujin.

| File                                                                 | Paper        | Problem                              | Check                                                                                                                                                     |
| -------------------------------------------------------------------- | ------------ | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2023-CGMPT-5.3.1-smooth_wave-ideal.prm`                             | Table 1      | Smooth traveling wave, ideal gas     | δ∞ within 24% of Table 1 (101 points), identical from 1601 points on; same rates (101–6401 points)                                                        |
| `2023-CGMPT-5.3.1-smooth_wave-vdw.prm`                               | Table 1      | Same, van der Waals                  | δ∞ within 4% of Table 1, same rates (101–6401 points)                                                                                                     |
| `2023-CGMPT-5.3.1-smooth_wave-jwl.prm`                               | Table 1      | Same, Jones-Wilkins-Lee              | δ∞ within 9% of Table 1, same rates including the jump to 4.8 (101–6401 points)                                                                           |
| `2023-CGMPT-5.3.1-smooth_wave-mie_gruneisen.prm`                     | Table 1      | Same, Mie-Grüneisen (`function` EOS) | δ∞ matches Table 1 within 1% (101–6401 points)                                                                                                            |
| `2023-CGMPT-5.3.2-isentropic_vortex-vdw.prm`                         | Table 2      | Isentropic vortex, van der Waals     | with `evc factor = 0.1` δ_q −5%, +22%, +73%, +120% against Table 2 (levels 5–8); the paper used an activation function that is no longer used, see header |
| `2023-CGMPT-5.4.1-blast_wave-jwl-case_1.prm`                         | Fig. 1 left  | Woodward-Colella blast waves, JWL    | profile as in Fig. 1 with A = 632.1 (see below)                                                                                                           |
| `2023-CGMPT-5.4.1-blast_wave-jwl-case_2.prm`                         | Fig. 1 right | Same, JWL (HMX)                      | profile as in Fig. 1 (right) with T = 0.0367 instead of 8.2e-4 (see below)                                                                                |
| `2023-CGMPT-5.5-schardin_triangle-vdw.prm`, `-schardin_triangle.geo` | Fig. 3       | Schardin's problem, van der Waals    | level 4 at t = 2.2e-3 as in Fig. 3 (right)                                                                                                                |
| `2023-CGMPT-5.6-shock_bubble-jwl.prm`                                | Figs. 4–6    | Shock-bubble interaction, JWL        | shock positions and bubble shapes as in Figs. 4–6 (levels 7–9) with inflow on the left (see below)                                                        |

Notes:

 - Table 3 lists A = 6.321e3 for Case 1 of the blast wave problem, but
   only A = 632.1 reproduces Figure 1 (left), so the parameter file uses
   A = 632.1.
 - For Case 2 the stated initial data (rho = 1, rho_0 = 1891) turn the JWL
   EOS into a gamma-law gas with gamma = 1.3. Figure 1 (right) is
   reproduced at T = 0.0367 instead of the listed T = 8.2e-4
   (equivalently, with pressures scaled by 2000).
 - The shock-bubble interaction requires inflow conditions on the left
   boundary (the paper states slip conditions everywhere).

[2026-HKMT]
-----------

| File                                      | Paper  | Problem                                         | Check                                                                               |
| ----------------------------------------- | ------ | ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| `2026-HKMT-6.3.2-sedov_blast-jwl-amr.prm` | Fig. 5 | Sedov-like blast, JWL, adaptive mesh refinement | runs to T; the mesh adaptation algorithm has changed considerably since [2026-HKMT] |

Not reproduced
--------------

 - [2026-HKMT] Section 6.4.1, multi-species shock-bubble interaction: ryujin
   has no multi-species Euler module.

Configurations that need extra steps
------------------------------------

 - `2023-CGMPT-5.5-schardin_triangle-vdw.prm` reads a quadrilateral mesh
   that has to be generated with gmsh first:
   ```
   gmsh -2 -format msh22 -o schardin_triangle.msh \
     2023-CGMPT-5.5-schardin_triangle.geo
   ```
