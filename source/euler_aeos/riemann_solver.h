//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <hyperbolic_system.h>

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * A fast approximative solver for the 1D Riemann problem. The solver
     * ensures that the estimate \f$\lambda_{\text{max}}\f$ that is returned
     * for the maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite ClaytonGuermondPopov-2022.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number = double>
    class RiemannSolver
    {
    public:
      /**
       * @copydoc HyperbolicSystem::problem_dimension
       */
      static constexpr unsigned int problem_dimension =
          HyperbolicSystem::problem_dimension<dim>;

      static constexpr unsigned int riemann_data_size = 5;
      using primitive_type = std::array<Number, riemann_data_size>;

      /**
       * @copydoc HyperbolicSystem::state_type
       */
      using state_type = HyperbolicSystem::state_type<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::precomputed_type
       */
      using precomputed_type = HyperbolicSystem::precomputed_type<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystem::n_precomputed_values<dim>;

      /**
       * @copydoc HyperbolicSystem::ScalarNumber
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      RiemannSolver(
          const HyperbolicSystem &hyperbolic_system,
          const MultiComponentVector<ScalarNumber, n_precomputed_values>
              &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
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

      /**
       * FIXME
       */
      Number c(const Number gamma_Z) const;

      /**
       * FIXME
       */
      Number
      alpha(const Number &rho, const Number &gamma, const Number &a) const;

      /**
       * Compute the best available, but expensive, upper bound on the
       * expansion-shock case as described in §5.4, Eqn. (5.7) and (5.8) in
       * @cite ClaytonGuermondPopov-2022
       */
      Number p_star_RS_full(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j) const;

      /**
       * Compute the best available, but expensive, upper bound on the
       * shock-shock case as described in §5.5, Eqn. (5.10) and (5.12) in
       * @cite ClaytonGuermondPopov-2022
       */
      Number p_star_SS_full(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j) const;

      /**
       * FIXME
       */
      Number p_star_quadratic(const primitive_type &riemann_data_i,
                              const primitive_type &riemann_data_j) const;

      /**
       * FIXME
       */
      Number p_star_interpolated(const primitive_type &riemann_data_i,
                                 const primitive_type &riemann_data_j) const;


      /**
       * See [Tabulated paper], page ???
       *
       * Cost: ???
       */
      Number phi_of_p_max(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j) const;


      /**
       * see [1], page 912, (3.7)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda1_minus(const primitive_type &riemann_data,
                           const Number p_star) const;


      /**
       * see [1], page 912, (3.8)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda3_plus(const primitive_type &primitive_state,
                          const Number p_star) const;


      /**
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
       * for lambda.
       *
       * This is the same lambda_max as computed by compute_gap. The function
       * simply avoids a number of unnecessary computations (in case we do
       * not need to know the gap).
       *
       * Cost: 0x pow, 2x division, 2x sqrt
       */
      Number compute_lambda(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j,
                            const Number p_star) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, and a
       * (normalized) "direction" n_ij, first compute the corresponding
       * projected state in the corresponding 1D Riemann problem, and then
       * compute and return the Riemann data [rho, u, p, a] (used in the
       * approximative Riemann solver).
       */
      primitive_type riemann_data_from_state(
          const HyperbolicSystem::state_type<dim, Number> &U,
          const Number &p,
          const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      const HyperbolicSystem &hyperbolic_system;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;
      //@}
    };


    /* Inline definitions */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    RiemannSolver<dim, Number>::riemann_data_from_state(
        const HyperbolicSystem::state_type<dim, Number> &U,
        const Number &p,
        const dealii::Tensor<1, dim, Number> &n_ij) const -> primitive_type
    {
      const auto rho = hyperbolic_system.density(U);
      const auto rho_inverse = ScalarNumber(1.0) / rho;

      const auto m = hyperbolic_system.momentum(U);
      const auto proj_m = n_ij * m;

      const auto gamma = hyperbolic_system.surrogate_gamma(U, p);

      const ScalarNumber b_interp = hyperbolic_system.b_interp();
      const Number x = Number(1.) - b_interp * rho;
      const Number a = std::sqrt(gamma * p / (rho * x));

#ifdef CHECK_BOUNDS
      AssertThrowSIMD(Number(p),
                      [](auto val) { return val > ScalarNumber(0.); },
                      dealii::ExcMessage("Internal error: p <= 0."));

      AssertThrowSIMD(x,
                      [](auto val) { return val > ScalarNumber(0.); },
                      dealii::ExcMessage("Internal error: 1. - b * rho <= 0."));

      AssertThrowSIMD(gamma,
                      [](auto val) { return val > ScalarNumber(1.); },
                      dealii::ExcMessage("Internal error: gamma <= 1."));
#endif

      return {{rho, proj_m * rho_inverse, p, gamma, a}};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::compute(
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int i,
        const unsigned int *js,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
#ifdef DEBUG_OUTPUT
    std::cout << "RiemannSolver<dim, Number>::compute()" << std::endl;
#endif
      const auto &[p_i, unused_i, s_i, eta_i] =
          precomputed_values.template get_tensor<Number, precomputed_type>(i);

      const auto &[p_j, unused_j, s_j, eta_j] =
          precomputed_values.template get_tensor<Number, precomputed_type>(js);

      const auto riemann_data_i = riemann_data_from_state(U_i, p_i, n_ij);
      const auto riemann_data_j = riemann_data_from_state(U_j, p_j, n_ij);

      return compute(riemann_data_i, riemann_data_j);
    }

  } // namespace EulerAEOS
} /* namespace ryujin */
