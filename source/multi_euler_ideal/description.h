//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include "../stub_parabolic_module.h"
#include "../stub_parabolic_system.h"
#include "hyperbolic_system.h"
#include "indicator.h"
#include "limiter.h"
#include "riemann_solver.h"

namespace ryujin
{
  namespace MultiSpeciesEuler
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * The compressible multi-species Euler equations of gas dynamics for
     * an ideal gas mixture under the assumption of thermal-mechanical
     * equilibrium. Each species \f$k\f$ is modeled by an ideal gas with
     * specific heat capacities \f$c_{p,k}\f$ and \f$c_{v,k}\f$.
     *
     * The governing equations are:
     * \f{align}
     *   \partial_t (\alpha_k \rho_k) + \nabla \cdot (\alpha_k \rho_k
     *   \mathbf{v}) &= 0, \quad k = 1, \ldots, n_s, \\
     *   \partial_t \mathbf{m} + \nabla \cdot (\mathbf{v} \otimes \mathbf{m}
     *   + p \mathbb{I}_d) &= 0, \\
     *   \partial_t E + \nabla \cdot ((E + p) \mathbf{v}) &= 0,
     * \f}
     * where \f$\alpha_k \rho_k\f$ are the partial densities, \f$\mathbf{m}
     * = \rho \mathbf{v}\f$ is the mixture momentum, \f$E\f$ is the mixture
     * total energy, and \f$p\f$ is the mixture pressure. The mixture
     * pressure is given by \f$p = (\bar{\gamma} - 1) \varepsilon\f$ where
     * \f$\bar{\gamma} = \bar{c}_p / \bar{c}_v\f$ is the mixture ratio of
     * specific heats.
     *
     * The numerical scheme ensures preservation of an invariant domain
     * \f$\mathcal{A} = \{ \mathbf{u} : \alpha_k \rho_k \geq 0,\;
     * \varepsilon(\mathbf{u}) > 0,\; s(\mathbf{u}) \geq s_{\min} \}\f$,
     * where \f$s(\mathbf{u})\f$ is the mixture specific entropy.
     *
     * The parabolic subsystem is chosen to be the identity (inviscid flow).
     *
     * @note See "Second-order invariant-domain preserving approximation to
     * the multi-species Euler equations" by Clayton, Dzanic, and Tovar
     * (arXiv:2505.09581) for full details of the mathematical formulation
     * and numerical scheme.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    struct Description {
      using HyperbolicSystem = MultiSpeciesEuler::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView =
          MultiSpeciesEuler::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = ryujin::StubParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicModule =
          ryujin::StubParabolicModule<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = MultiSpeciesEuler::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = MultiSpeciesEuler::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = MultiSpeciesEuler::RiemannSolver<dim, Number>;
    };
  } // namespace MultiSpeciesEuler
} // namespace ryujin
