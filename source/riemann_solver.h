//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "simd.h"

#include "problem_description.h"

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
   * @ingroup EulerModule
   */
  template <int dim, typename Number = double>
  class RiemannSolver
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type = ProblemDescription::rank1_type<dim, Number>;

    /**
     * @copydoc ProblemDescription::ScalarNumber
     */
    using ScalarNumber = typename get_value_type<Number>::type;

    /**
     * @name RiemannSolver compile time options
     */
    //@{

    /**
     * Maximal number of Newton iterations used in the approximate Riemann to
     * improve the upper bound \f$\lambda_{\text{max}}\f$ on the wavespeed.
     * @ingroup CompileTimeOptions
     */
    static constexpr unsigned int newton_max_iter_ = RIEMANN_NEWTON_MAX_ITER;

    //@}
    /**
     * @name Compute wavespeed estimates
     */
    //@{

    /**
     * Constructor taking a ProblemDescription instance as argument
     */
    RiemannSolver(const ProblemDescription &problem_description)
        : problem_description(problem_description)
        , gamma(problem_description.gamma())
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
    std::tuple<Number /*lambda_max*/,
               Number /*p_star*/,
               unsigned int /*iteration*/>
    compute(const std::array<Number, 4> &riemann_data_i,
            const std::array<Number, 4> &riemann_data_j);

    /**
     * For two given states U_i a U_j and a (normalized) "direction" n_ij
     * compute an estimation of an upper bound for lambda.
     *
     * Returns a tuple consisting of lambda max and the number of Newton
     * iterations used in the solver to find it.
     */
    std::tuple<Number /*lambda_max*/,
               Number /*p_star*/,
               unsigned int /*iteration*/>
    compute(const rank1_type &U_i,
            const rank1_type &U_j,
            const dealii::Tensor<1, dim, Number> &n_ij);

    //@}

  protected:
    /** @name Internal functions used in the Riemann solver */
    //@{
    /**
     * See [1], page 912, (3.4).
     *
     * Cost: 1x pow, 1x division, 2x sqrt
     */
    Number f(const std::array<Number, 4> &primitive_state,
             const Number &p_star);

    /**
     * See [1], page 912, (3.4).
     *
     * Cost: 1x pow, 3x division, 1x sqrt
     */
    Number df(const std::array<Number, 4> &primitive_state,
              const Number &p_star);


    /**
     * See [1], page 912, (3.3).
     *
     * Cost: 2x pow, 2x division, 4x sqrt
     */
    Number phi(const std::array<Number, 4> &riemann_data_i,
               const std::array<Number, 4> &riemann_data_j,
               const Number &p);


    /**
     * This is a specialized variant of phi() that computes phi(p_max). It
     * inlines the implementation of f() and eliminates all unnecessary
     * branches in f().
     *
     * Cost: 0x pow, 2x division, 2x sqrt
     */
    Number phi_of_p_max(const std::array<Number, 4> &riemann_data_i,
                        const std::array<Number, 4> &riemann_data_j);


    /**
     * See [1], page 912, (3.3).
     *
     * Cost: 2x pow, 6x division, 2x sqrt
     */
    Number dphi(const std::array<Number, 4> &riemann_data_i,
                const std::array<Number, 4> &riemann_data_j,
                const Number &p);


    /**
     * see [1], page 912, (3.7)
     *
     * Cost: 0x pow, 1x division, 1x sqrt
     */
    Number lambda1_minus(const std::array<Number, 4> &riemann_data,
                         const Number p_star);


    /**
     * see [1], page 912, (3.8)
     *
     * Cost: 0x pow, 1x division, 1x sqrt
     */
    Number lambda3_plus(const std::array<Number, 4> &primitive_state,
                        const Number p_star);


    /**
     * For two given primitive states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>, and two guesses p_1 <= p* <= p_2,
     * compute the gap in lambda between both guesses.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     *
     * Cost: 0x pow, 4x division, 4x sqrt
     */
    std::array<Number, 2>
    compute_gap(const std::array<Number, 4> &riemann_data_i,
                const std::array<Number, 4> &riemann_data_j,
                const Number p_1,
                const Number p_2);


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
    Number compute_lambda(const std::array<Number, 4> &riemann_data_i,
                          const std::array<Number, 4> &riemann_data_j,
                          const Number p_star);


    /**
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
     *
     * See [1], page 914, (4.3)
     *
     * Cost: 2x pow, 2x division, 0x sqrt
     */
    Number p_star_two_rarefaction(const std::array<Number, 4> &riemann_data_i,
                                  const std::array<Number, 4> &riemann_data_j);


    /**
     * Given the pressure minimum and maximum and two corresponding
     * densities we compute approximations for the density of corresponding
     * shock and expansion waves.
     *
     * [2] Formula (4.4)
     *
     * Cost: 2x pow, 2x division, 0x sqrt
     */
    inline std::array<Number, 4>
    shock_and_expansion_density(const Number p_min,
                                const Number p_max,
                                const Number rho_p_min,
                                const Number rho_p_max,
                                const Number p_1,
                                const Number p_2);


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, and a
     * (normalized) "direction" n_ij, first compute the corresponding
     * projected state in the corresponding 1D Riemann problem, and then
     * compute and return the Riemann data [rho, u, p, a] (used in the
     * approximative Riemann solver).
     */
    std::array<Number, 4> riemann_data_from_state(
        const ProblemDescription &problem_description,
        const ProblemDescription::rank1_type<dim, Number> &U,
        const dealii::Tensor<1, dim, Number> &n_ij);

  private:
    const ProblemDescription &problem_description;
    const ScalarNumber gamma;
    const ScalarNumber gamma_inverse;
    const ScalarNumber gamma_minus_one_inverse;
    const ScalarNumber gamma_minus_one_over_gamma_plus_one;
    const ScalarNumber gamma_plus_one_inverse;

    //@}
  };

} /* namespace ryujin */

#ifdef OBSESSIVE_INLINING
#include "riemann_solver.template.h"
#endif
