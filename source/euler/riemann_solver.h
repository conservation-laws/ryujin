//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace Euler
  {
    template <typename ScalarNumber = double>
    class RiemannSolverParameters : public dealii::ParameterAcceptor
    {
    public:
      RiemannSolverParameters(const std::string &subsection = "/RiemannSolver")
          : ParameterAcceptor(subsection)
      {
      }
    };


    /**
     * A fast approximative solver for the 1D Riemann problem. The solver
     * ensures that the estimate \f$\lambda_{\text{max}}\f$ that is returned
     * for the maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite GuermondPopov2016b.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number = double>
    class RiemannSolver
    {
    public:
      /**
       * @copydoc HyperbolicSystem::View
       */
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::View::problem_dimension
       */
      static constexpr unsigned int problem_dimension =
          HyperbolicSystemView::problem_dimension;

      /**
       * Number of components in a primitive state, we store \f$[\rho, v,
       * p, a]\f$, thus, 4.
       */
      static constexpr unsigned int riemann_data_size = 4;

      /**
       * The array type to store the expanded primitive state for the
       * Riemann solver \f$[\rho, v, p, a]\f$
       */
      using primitive_type = std::array<Number, riemann_data_size>;

      /**
       * @copydoc HyperbolicSystem::View::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

      /**
       * @copydoc HyperbolicSystem::View::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystem::View::ScalarNumber
       */
      using ScalarNumber = typename HyperbolicSystemView::ScalarNumber;

      /**
       * @copydoc RiemannSolverParameters
       */
      using Parameters = RiemannSolverParameters<ScalarNumber>;

      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      RiemannSolver(
          const HyperbolicSystem &hyperbolic_system,
          const Parameters &parameters,
          const MultiComponentVector<ScalarNumber, n_precomputed_values>
              &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * For two given 1D primitive states riemann_data_i and riemann_data_j,
       * compute an estimation of an upper bound for the maximum wavespeed
       * lambda.
       */
      Number compute(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimation of an upper bound for lambda.
       *
       * Returns a tuple consisting of lambda max and the number of Newton
       * iterations used in the solver to find it.
       */
      Number compute(const state_type &U_i,
                     const state_type &U_j,
                     const unsigned int i,
                     const unsigned int *js,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

      //@}

    protected:
      /** @name Internal functions used in the Riemann solver */
      //@{

#ifndef DOXYGEN
      /**
       * FIXME
       */
      Number f(const primitive_type &riemann_data, const Number p_star) const;


      /**
       * FIXME
       */
      Number phi(const primitive_type &riemann_data_i,
                 const primitive_type &riemann_data_j,
                 const Number p_in) const;
#endif


      /**
       * See @cite GuermondPopov2016b, page 912, (3.3).
       *
       * The approximate Riemann solver is based on a function phi(p) that is
       * montone increasing in p, concave down and whose (weak) third
       * derivative is non-negative and locally bounded [1, p. 912]. Because
       * we actually do not perform any iteration for computing our wavespeed
       * estimate we can get away by only implementing a specialized variant
       * of the phi function that computes phi(p_max). It inlines the
       * implementation of the "f" function and eliminates all unnecessary
       * branches in "f".
       *
       * Cost: 0x pow, 2x division, 2x sqrt
       */
      Number phi_of_p_max(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j) const;


      /**
       * see @cite GuermondPopov2016 page 912, (3.7)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda1_minus(const primitive_type &riemann_data,
                           const Number p_star) const;


      /**
       * see @cite GuermondPopov2016 page 912, (3.8)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda3_plus(const primitive_type &primitive_state,
                          const Number p_star) const;


      /**
       * see @cite GuermondPopov2016 page 912, (3.9)
       *
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
       * for lambda.
       *
       * Cost: 0x pow, 2x division, 2x sqrt (inclusive)
       */
      Number compute_lambda(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j,
                            const Number p_star) const;


      /**
       * Two-rarefaction approximation to p_star computed for two primitive
       * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
       *
       * See @cite GuermondPopov2016b, page 914, (4.3)
       *
       * Cost: 2x pow, 2x division, 0x sqrt
       */
      Number p_star_two_rarefaction(const primitive_type &riemann_data_i,
                                    const primitive_type &riemann_data_j) const;

      /**
       * Failsafe approximation to p_star computed for two primitive states
       * <code>riemann_data_i</code> and <code>riemann_data_j</code>.
       *
       * See @cite ClaytonGuermondPopov-2022, (5.11):
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */
      Number p_star_failsafe(const primitive_type &riemann_data_i,
                             const primitive_type &riemann_data_j) const;


      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, and a
       * (normalized) "direction" n_ij, first compute the corresponding
       * projected state in the corresponding 1D Riemann problem, and then
       * compute and return the Riemann data [rho, u, p, a] (used in the
       * approximative Riemann solver).
       */
      primitive_type
      riemann_data_from_state(const state_type &U,
                              const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;
      //@}
    };


    /* Inline definitions */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    RiemannSolver<dim, Number>::riemann_data_from_state(
        const state_type &U, const dealii::Tensor<1, dim, Number> &n_ij) const
        -> primitive_type
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      const auto rho = view.density(U);
      const auto rho_inverse = Number(1.0) / rho;

      const auto m = view.momentum(U);
      const auto proj_m = n_ij * m;
      const auto perp = m - proj_m * n_ij;

      const auto E =
          view.total_energy(U) - Number(0.5) * perp.norm_square() * rho_inverse;

      using state_type_1d =
          typename HyperbolicSystem::View<1, Number>::state_type;
      const auto view_1d = hyperbolic_system.view<1, Number>();

      const auto state = state_type_1d{{rho, proj_m, E}};
      const auto p = view_1d.pressure(state);
      const auto a = view_1d.speed_of_sound(state);
      return {{rho, proj_m * rho_inverse, p, a}};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::compute(
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int /*i*/,
        const unsigned int * /*js*/,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto riemann_data_i = riemann_data_from_state(U_i, n_ij);
      const auto riemann_data_j = riemann_data_from_state(U_j, n_ij);

      return compute(riemann_data_i, riemann_data_j);
    }
  } // namespace Euler
} // namespace ryujin
