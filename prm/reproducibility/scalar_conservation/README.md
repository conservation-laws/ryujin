Scalar conservation equations: reproducibility
==============================================

This directory contains parameter files for the numerical results of

> [2024-GMPST] J.-L. Guermond, M. Maier, B. Popov, L. Saavedra, I. Tomas.
> First-order greedy invariant-domain preserving approximation for
> hyperbolic problems: scalar conservation laws, and p-system. J. Sci.
> Comput., 100(2):46, 2024.
> [DOI:10.1007/s10915-024-02592-4](https://doi.org/10.1007/s10915-024-02592-4),
> [arXiv:2310.01713](https://arxiv.org/abs/2310.01713)

The file names start with `2024-GMPST-` followed by the number of the
section in the paper. Each parameter file has a header comment that
names the figure or table it reproduces and summarizes the problem setup.
Run a configuration with

```
mpirun -np <n> ryujin 2024-GMPST-6.3-kpp-random_entropy.prm
```

All configurations use the `scalar conservation` equation module and should
be run with double precision (`NUMBER=double`).

The results of the paper were computed with a Fortran code and with ryujin.
The parameter files are reconstructions, so differences to the published
numbers are to be expected. All configurations use the first-order method
(`limiter iterations = 0`) and the greedy wave speed (`use greedy wavespeed
= true`). The Kruzkov entropies of the paper map to the `wave speed
estimator` options as follows:

| Paper                                                                        | ryujin                                                                                             |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| lambda_12 only                                                               | no additional entropy                                                                              |
| k_i = (U_i^min + U_i^max) / 2 per grid point                                 | `use averaged entropy = true`: k = (u_i + u_j) / 2 per pair (i,j)                                  |
| k_i = theta_i U_i^min + (1 - theta_i) U_i^max, random theta_i per grid point | `random entropies = 1`: k = theta u_i + (1 - theta) u_j, random theta per pair (i,j) and time step |

The ryujin variants are the ones written in the caption of Table 1 of the
paper. The random number generator is seeded differently on every run.

Contents
--------

| File                                                        | Paper           | Problem                                                   | Check                                                                                                              |
| ----------------------------------------------------------- | --------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `2024-GMPST-6.1-piecewise_linear_flux-roe_average.prm`      | Figure 2a       | Riemann problem, piecewise linear flux, lambda_12 only    | Left discontinuity stays at x = 0, as in Fig. 2a                                                                   |
| `2024-GMPST-6.1-piecewise_linear_flux-averaged_entropy.prm` | Figure 2b       | Same, averaged entropy                                    | Entropy solution as in Fig. 2b; L1 error 2.4e-2 to 1.2e-2 for 201 to 801 points                                    |
| `2024-GMPST-6.2-two_sonic_points-random_entropy.prm`        | Table 1 (left)  | Riemann problem, f(u) = sin(u), random entropy, t0 = 1e-8 | L1 7% (fine) to 1.7 times (coarse) above Table 1, L2 about 2.4 times larger; within 4% and 7% with the patch below |
| `2024-GMPST-6.2-two_sonic_points-averaged_entropy.prm`      | Table 1 (right) | Same, averaged entropy, t0 = 1e-8                         | Stagnation as in Table 1 only with the patch below (L1 5–16% below); converges without it                          |
| `2024-GMPST-6.3-kpp-roe_average.prm`                        | Figure 4a       | 2D KPP problem, lambda_12 only                            | Wrong solution with a sharp front on the initial circle, as in Fig. 4a                                             |
| `2024-GMPST-6.3-kpp-averaged_entropy.prm`                   | Figure 4b       | Same, averaged entropy                                    | Spiral structure as in Fig. 4b; hardly differs from 4c                                                             |
| `2024-GMPST-6.3-kpp-random_entropy.prm`                     | Figure 4c       | Same, random entropy                                      | Spiral structure as in Fig. 4c (119k Q1 points instead of a 118850 point Delaunay mesh)                            |

Not reproduced
--------------

 - Figure 3 (Remark 6.1): the wave speed based on the square entropy eta(u)
   = u^2 / 2 is not implemented.
 - Section 7, Table 2 and Figure 5: ryujin has no Lagrangian p-system
   module.
 - Section 8, Figure 6 (Woodward-Colella blast wave, compressible Euler):
   the greedy wave speed is only implemented for scalar conservation
   equations.

Configurations that need extra steps
------------------------------------

 - `2024-GMPST-6.2-two_sonic_points-averaged_entropy.prm`: the stall of
   the convergence in Table 1 (right) is only seen with the signed entropy
   wave speeds (4.3) of the paper. The current ryujin takes their absolute
   value; apply the patch to see the stall:
   ```
   git apply prm/reproducibility/scalar_conservation/2024-GMPST-6.2-signed_wave_speed.patch
   ```
