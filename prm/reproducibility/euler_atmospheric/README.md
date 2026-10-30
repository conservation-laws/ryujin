Atmospheric Euler equations: reproducibility
=============================================

This directory contains the parameter files for the numerical results of

> C. Farris, M. Maier, E. J. Tovar. Well-balanced second-order
> approximation of the compressible atmospheric Euler equations.
> [arXiv:2606.19764](https://arxiv.org/abs/2606.19764), 2026.

The file names start with the number of the subsection in the preprint.
Each parameter file has a header comment that names the table or figure
it reproduces and summarizes the problem setup. Run a configuration with

```
ryujin 4.1.1-smooth_wave.prm
```

or in parallel with `mpirun -np <n> ryujin <file>.prm`. All configurations
use the `euler atmospheric` equation module and should be run with double
precision (`NUMBER=double`).

Contents
--------

| Files | Preprint | Problem |
|---|---|---|
| `4.1.1-smooth_wave.prm` | Table 4.1 | Smooth traveling wave, no gravity |
| `4.1.2-exact_source-1d-high_order.prm` | Table 4.3 | Exact solution with gravity source, 1D, high order |
| `4.1.2-exact_source-2d-low_order-uniform.prm` | Table 4.2a | Same, 2D, low order, uniform mesh |
| `4.1.2-exact_source-2d-low_order-distorted.prm` | Table 4.2b | Same, 2D, low order, 10% random mesh distortion |
| `4.1.3-well_balanced-warped_mesh-{a,b,c,d}-*.prm` | Figure 4.1 | Isentropic atmosphere at rest on a curvilinear mesh |
| `4.1.4-isothermal_well_balanced.prm` | Table 4.4 | Isothermal steady state, 1D |
| `4.1.4-pressure_perturbation.prm` | Figure 4.2 | Pressure perturbation over the isothermal state |
| `4.1.5-conservation.prm` | Table 4.5 | Conservation of mass and potential temperature |
| `4.1.5-sod_shock.prm` | Figure 4.3 | Sod shock tube with gravity |
| `4.2.1-rising_bubble.prm` | Figure 4.4 | Rising thermal bubble |
| `4.2.2-density_current.prm` | Figure 4.5 | Density current |
| `4.2.3-schaer_mountain.prm` | Figure 4.6 | Schär mountain, with Rayleigh damping sponge layers |
| `4.2.4-blast_over_mountain.prm`, `-rectangle_mountain.geo` | Figure 4.7 | Blast over a mountain |

The convergence tables of Sections 4.1.1 and 4.1.2 are obtained by
increasing `mesh refinement` (or the number of subdivisions) as described
in the header of each parameter file.

Configurations that need extra steps
------------------------------------

 - `4.1.3-well_balanced-warped_mesh-b-high_order-no_mass_correction.prm`
   switches off the mass matrix correction of the high-order flux. This
   requires a modified source tree:
   ```
   git apply prm/reproducibility/euler_atmospheric/4.1.3-well_balanced-b-no_mass_correction.patch
   ```
   `4.1.3-well_balanced-b-no_mass_correction-dG.patch` makes the same
   change in the second code path of
   `include/ryujin/typed/hyperbolic_module.template.h`.

 - `4.1.2-exact_source-2d-low_order-distorted.prm` distorts the mesh
   randomly with a new seed on every run, so the errors vary slightly from
   run to run.

 - `4.2.4-blast_over_mountain.prm` reads a triangular mesh that has to be
   generated with gmsh first:
   ```
   gmsh -2 -format msh22 -o rectangle_mountain.msh 4.2.4-rectangle_mountain.geo
   ```

Regression tests
----------------

Shortened versions of most of these configurations (coarse meshes, short
final times) are part of the testsuite in `tests/euler_atmospheric/`.
