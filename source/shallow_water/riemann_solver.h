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

    static constexpr unsigned int riemann_data_size = 3;
    using primitive_type = std::array<Number, riemann_data_size>;

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = HyperbolicSystem::state_type<dim, Number>;

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
    RiemannSolver(const HyperbolicSystem &hyperbolic_system)
        : hyperbolic_system(hyperbolic_system)
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
                   const dealii::Tensor<1, dim, Number> &n_ij) const;


  protected:
    //@}
    /**
     * @name Internal functions used in the Riemann solver
     */
    //@{

    Number f(const primitive_type &primitive_state, const Number &h_star) const;

    Number phi(const primitive_type &riemann_data_i,
               const primitive_type &riemann_data_j,
               const Number &h) const;

    Number lambda1_minus(const primitive_type &riemann_data,
                         const Number h_star) const;

    Number lambda3_plus(const primitive_type &riemann_data,
                        const Number h_star) const;

    Number compute_lambda(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j,
                          const Number h_star) const;

    Number h_star_two_rarefaction(const primitive_type &riemann_data_i,
                                  const primitive_type &riemann_data_j) const;

    primitive_type
    riemann_data_from_state(const state_type &U,
                            const dealii::Tensor<1, dim, Number> &n_ij) const;

  private:
    const HyperbolicSystem &hyperbolic_system;

    //@}
  };

} /* namespace ryujin */
