Euler-Poisson equations: reproducibility
=======================================

This directory collects the parameter files for the Euler-Poisson family
of equation modules (`euler poisson` with an ideal gas equation of state
and `euler poisson barotropic`) for the numerical results of

> [2023-MST] M. Maier, J. N. Shadid, I. Tomas. Structure-preserving
> finite-element schemes for the Euler-Poisson equations. Commun. Comput.
> Phys., 33(3):647–691, 2023.
> [DOI:10.4208/cicp.OA-2022-0205](https://doi.org/10.4208/cicp.OA-2022-0205),
> [arXiv:2207.07860](https://arxiv.org/abs/2207.07860)

> [2026-HMST] J. Hoffart, M. Maier, J. N. Shadid, I. Tomas.
> Structure-preserving finite-element approximations of the magnetic
> Euler-Poisson equations. J. Comput. Phys., 566:115269, 2026.
> [DOI:10.1016/j.jcp.2026.115269](https://doi.org/10.1016/j.jcp.2026.115269),
> [arXiv:2510.11808](https://arxiv.org/abs/2510.11808)

The results of [2023-MST] were computed with a deal.II code that predates
ryujin's Euler-Poisson modules (added in 2025); the parameter files for
[2023-MST] are ryujin counterparts of those computations. [2026-HMST] was
computed with ryujin.

The file names start with the year and author initials of the paper,
followed by the number of the subsection. Each parameter file has a header
comment that names the table or figure it reproduces and summarizes the
problem setup. Run a configuration with

```
mpirun -np <n> ryujin 2026-HMST-5.3-diocotron-mode_5.prm
```

Section, figure and table numbers of [2026-HMST] refer to the arXiv
preprint; the numbering of the journal version may differ.

All configurations should be run with double precision (`NUMBER=double`).

Contents
--------

The "Check" column summarizes how close a short run with the current
code comes to the published result.

| File                                             | Paper                       | Problem                                                          | Check                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------ | --------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2023-MST-5.1-isentropic_vortex.prm`             | [2023-MST] Table 2          | Isentropic vortex with neutralizing background, distorted meshes | undistorted meshes (levels 2–6): rates 1.25, 1.63, 1.80, 1.80 against 1.19, 1.68, 1.79, 1.91 of Table 1(a), errors differ from the table by a nearly constant factor (normalization); distorted meshes: rates 1.3–1.8, no rate loss as in Table 2(a); full restart lowers the error by 10%, not 2x |
| `2023-MST-5.3-electrostatic_implosion.prm`       | [2023-MST] Figures 4, 5     | Electrostatic implosion in a disk                                | level 7 snapshots (t = 1/16 to 1 t_F) as in Figure 4; full restart and no restart differ little at t = 7/8 t_F, as in Figure 5                                                                                                                                                                     |
| `2026-HMST-5.2-isentropic_vortex-low_order.prm`  | [2026-HMST] Table 5.1a      | Isentropic vortex, forward/backward Euler                        | rates 0.76, 0.87 vs 0.73, 0.85 (levels 5–7); errors 30% below the table                                                                                                                                                                                                                            |
| `2026-HMST-5.2-isentropic_vortex-high_order.prm` | [2026-HMST] Table 5.1b      | Isentropic vortex, Strang split ERK33/Crank-Nicolson             | with `evc factor = 0.1` (a fit, see header) errors within 6% of the table (levels 5–7); 2.4–3.6x above with the default factor                                                                                                                                                                     |
| `2026-HMST-5.3-diocotron-mode_{3,4,5}.prm`       | [2026-HMST] Figures 5.1–5.5 | Diocotron instability, modes 3, 4, 5                             | growth rates 0.777, 0.938, 0.671 (level 6) and 0.773, 0.935, 0.685 (level 7) against 0.777, 0.935, 0.667 and 0.775, 0.919, 0.677 of Figure 5.4(d)                                                                                                                                                  |

Variants (refinement levels, restart strategies, distortion) are selected
by changing the parameters named in the header of each file.

Not reproduced
--------------

 - [2023-MST] Table 1: the nested mesh sequence (distorted coarse mesh,
   refined without further distortion) cannot be generated; ryujin distorts
   the mesh after global refinement.
 - [2023-MST] Section 5.2, plasma column (Figures 2, 3): needs homogeneous
   Neumann conditions for the potential with a mean value filter, which is
   currently not supported.
 - Relaxation restart of the Gauss law ([2023-MST] Tables 1, 2 (b), Figures
   3, 5, 7; [2026-HMST] Figure 5.5): the strategy `correction` is currently
   not implemented.
 - Gauss law residual and energy diagnostics ([2023-MST] Figure 6,
   [2026-HMST] Figure 5.5) and the Fourier analysis of the potential
   ([2026-HMST] Figure 5.4) are not computed by ryujin; they need
   postprocessing.

Notes
-----

 - `2023-MST-5.1-isentropic_vortex.prm` distorts the mesh randomly with a
   new seed on every run, so the errors vary slightly from run to run.
