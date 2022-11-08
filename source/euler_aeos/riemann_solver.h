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
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension =
        HyperbolicSystem::problem_dimension<dim>;

    static constexpr unsigned int riemann_data_size = 4;
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
    RiemannSolver(const HyperbolicSystem &hyperbolic_system,
                  const MultiComponentVector<ScalarNumber, n_precomputed_values>
                      &precomputed_values)
        : hyperbolic_system(hyperbolic_system)
        , precomputed_values(precomputed_values)
        , gamma(hyperbolic_system.legacy_gamma())
        , gamma_inverse(1. / gamma)
        , gamma_minus_one_inverse(1. / (gamma - 1.))
        , gamma_minus_one_over_gamma_plus_one((gamma - 1.) / (gamma + 1.))
        , gamma_plus_one_inverse(1. / (gamma + 1.))
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
     * See [1], page 912, (3.3).
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
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
     *
     * See [1], page 914, (4.3)
     *
     * Cost: 2x pow, 2x division, 0x sqrt
     */
    Number p_star_two_rarefaction(const primitive_type &riemann_data_i,
                                  const primitive_type &riemann_data_j) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, and a
     * (normalized) "direction" n_ij, first compute the corresponding
     * projected state in the corresponding 1D Riemann problem, and then
     * compute and return the Riemann data [rho, u, p, a] (used in the
     * approximative Riemann solver).
     */
    primitive_type
    riemann_data_from_state(const HyperbolicSystem::state_type<dim, Number> &U,
                            const Number &gamma,
                            const Number &p,
                            const dealii::Tensor<1, dim, Number> &n_ij) const;

  private:
    const HyperbolicSystem &hyperbolic_system;

    const MultiComponentVector<ScalarNumber, n_precomputed_values>
        &precomputed_values;

    const ScalarNumber gamma;
    const ScalarNumber gamma_inverse;
    const ScalarNumber gamma_minus_one_inverse;
    const ScalarNumber gamma_minus_one_over_gamma_plus_one;
    const ScalarNumber gamma_plus_one_inverse;

    //@}
  };


  /* Inline definitions */


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline auto
  RiemannSolver<dim, Number>::riemann_data_from_state(
      const HyperbolicSystem::state_type<dim, Number> &U,
      const Number &gamma,
      const Number &p,
      const dealii::Tensor<1, dim, Number> &n_ij) const -> primitive_type
  {
    const auto rho = hyperbolic_system.density(U);
    const auto rho_inverse = Number(1.0) / rho;

    const auto m = hyperbolic_system.momentum(U);
    const auto proj_m = n_ij * m;
    const auto perp = m - proj_m * n_ij;

    const auto E = hyperbolic_system.total_energy(U) -
                   Number(0.5) * perp.norm_square() * rho_inverse;

    const auto state =
        HyperbolicSystem::state_type<1, Number>({rho, proj_m, E});
    const auto a = hyperbolic_system.speed_of_sound(state, gamma, p);

    return {{rho, proj_m * rho_inverse, p, a}};
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::compute(
      const state_type &U_i,
      const state_type &U_j,
      const unsigned int i,
      const unsigned int *js,
      const dealii::Tensor<1, dim, Number> &n_ij) const
  {
    const auto &[p_i, gamma_min_i, s_i, eta_i] =
        precomputed_values.template get_tensor<Number, precomputed_type>(i);

    const auto &[p_j, gamma_min_j, s_j, eta_j] =
        precomputed_values.template get_tensor<Number, precomputed_type>(js);

    const auto riemann_data_i =
        riemann_data_from_state(U_i, gamma_min_i, p_i, n_ij);
    const auto riemann_data_j =
        riemann_data_from_state(U_j, gamma_min_j, p_j, n_ij);

    return compute(riemann_data_i, riemann_data_j);
  }

} /* namespace ryujin */
