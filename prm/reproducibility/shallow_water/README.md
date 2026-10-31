Shallow water equations: reproducibility
=========================================

This directory contains parameter files for the numerical results of

> [2025-GMT] J.-L. Guermond, M. Maier, E. J. Tovar. A high-order explicit
> Runge-Kutta approximation technique for the shallow water equations.
> Computers & Fluids, 288:106493, 2025.
> [arXiv:2403.17123](https://arxiv.org/abs/2403.17123)

> [2026-HKMT] J. J. Harmon, M. Kronbichler, M. Maier, E. J. Tovar. A
> conservative invariant-domain preserving projection technique for
> hyperbolic systems under adaptive mesh refinement. Preprint, 2025.
> [arXiv:2507.18717](https://arxiv.org/abs/2507.18717)

The file names start with the year and author initials of the paper,
followed by the number of the section. Each parameter file has a header
comment that names the table or figure it reproduces and summarizes the
problem setup. The original parameter files of these papers were not
archived; the files here are reconstructions, so differences to the
published numbers are to be expected. Run a configuration with

```
mpirun -np <n> ryujin 2025-GMT-5.3.1-smooth_vortex.prm
```

All configurations use the `shallow water` equation module and should be
run with double precision (`NUMBER=double`).

The time stepping schemes RK(s,p;c_eff) of [2025-GMT] correspond to the
following values of `time stepping scheme`: RK(2,2;1/2) = `ssprk 22`,
RK(2,2;1) = `erk 22`, RK(3,3;1/3) = `ssprk 33`, RK(3,3;1) = `erk 33`,
RK(4,3;1) = `erk 43`, RK(5,4;1) = `erk 54`.

Contents
--------

| File                                            | Paper                 | Problem                                                      | Check                                                                                             |
| ----------------------------------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `2025-GMT-5.2.1-well_balancing_at_rest.prm`     | [2025-GMT] Table 2    | Lake at rest over three cones, distorted mesh                | Linf error of h at round-off (1e-16 to 6e-13), levels 6–7                                         |
| `2025-GMT-5.2.2-inclined_plane_friction.prm`    | [2025-GMT] Table 3    | Steady flow over an inclined plane with Manning friction, 1D | errors at round-off (3e-14 to 9e-13), levels 9–10                                                 |
| `2025-GMT-5.3.1-smooth_vortex.prm`              | [2025-GMT] Table 4    | Smooth traveling vortex, convergence                         | L1 errors within 2% of Table 4, levels 5–8, both schemes                                          |
| `2025-GMT-5.3.2-paraboloid.prm`                 | [2025-GMT] Table 5    | Thacker's planar surface in a paraboloid                     | L1 errors 4–13% below Table 5, similar rates, levels 5–7                                          |
| `2025-GMT-5.4.1-g2_s2_steady_flow.prm`          | [2025-GMT] Figs. 4, 5 | Flume experiment G2-S.2, steady inflow                       | steady state, q = 0.0104 conserved along the flume, level 0                                       |
| `2025-GMT-5.4.2-g3_d1_dam_break.prm`            | [2025-GMT] Figs. 7, 8 | Flume experiment G3-D.1, dam break, wave gauges              | gauge depths within about 0.3 cm of Fig. 8 for t > 8 s at level 1; wetting about 0.5 s late       |
| `2025-GMT-5.5-efficiency.prm`                   | [2025-GMT] Table 6    | Smooth vortex, comparison of RK schemes                      | cycle counts 11% above Table 6 for all schemes                                                    |
| `2026-HKMT-6.2.1-three_cones_dam_break-amr.prm` | [2026-HKMT] Fig. 2    | Dam break over three cones with adaptive mesh refinement     | to T = 8 s with max. level 6: total mass constant to round-off, dt ≈ 1e-2 throughout (cf. Fig. 3) |

The convergence tables are obtained by changing `mesh refinement` and
`time stepping scheme` as described in the header of each parameter file.

Remarks:

 - The paraboloid errors of Table 5 are reproduced to three digits by
   the code version of the time (see
   `prm/verification/shallow_water-paraboloid_2d-erk33.baseline`); the
   current code gives slightly smaller errors. The current code also
   takes about 10% more time steps (by the same factor for all time
   stepping schemes), hence the higher cycle counts of the efficiency
   test.

 - The flume length is rounded from 6.0078 m to 6.01 m so that the
   measurement sections and gauges are mesh vertices.

 - The measurement data of Martínez-Aranda et al. for Figures 5 and 8
   of [2025-GMT] are not part of this repository.

Not reproduced
--------------

 - [2025-GMT] Section 5.2.3 (rainfall over an inclined plane, Figure 2):
   ryujin has no rain source term.
 - [2025-GMT] Section 5.6 (Lake Dunlap dam break, Figure 9): needs a ryujin
   build with GDAL and the USGS digital elevation model, which is not
   part of this repository. The setup can be built from the `geotiff`
   initial state and the `geotiff profile` geometry.
 - [2026-HKMT] Section 6.2.2 (dam break without limiting of the projection,
   Figure 3): the projection uses the limiter of `F - HyperbolicModule`, so
   limiting cannot be switched off for the projection alone.
