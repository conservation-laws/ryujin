//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

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
    static constexpr unsigned int problem_dimension = ProblemDescription<dim>::problem_dimension;
    // clang-format on

    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type =
        typename ProblemDescription<dim>::template rank1_type<Number>;

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
     * For two given 1D primitive states riemann_data_i and riemann_data_j,
     * compute an estimation of an upper bound for the maximum wavespeed
     * lambda.
     */
    static std::tuple<Number /*lambda_max*/,
                      Number /*p_star*/,
                      unsigned int /*iteration*/>
    compute(const ProblemDescription<dim, ScalarNumber> &problem_description,
            const std::array<Number, 4> &riemann_data_i,
            const std::array<Number, 4> &riemann_data_j);

    /**
     * For two given states U_i a U_j and a (normalized) "direction" n_ij
     * compute an estimation of an upper bound for lambda.
     *
     * Returns a tuple consisting of lambda max and the number of Newton
     * iterations used in the solver to find it.
     */
    static std::tuple<Number /*lambda_max*/,
                      Number /*p_star*/,
                      unsigned int /*iteration*/>
    compute(const ProblemDescription<dim, ScalarNumber> &problem_description,
            const rank1_type &U_i,
            const rank1_type &U_j,
            const dealii::Tensor<1, dim, Number> &n_ij);

    //@}
  };

} /* namespace ryujin */

#endif /* RIEMANN_SOLVER_H */

#ifdef OBSESSIVE_INLINING
#include "riemann_solver.template.h"
#endif
