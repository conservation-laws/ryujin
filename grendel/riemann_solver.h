#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include <compile_time_options.h>
#include "helper.h"
#include "simd.h"

#include "problem_description.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace grendel
{

  /**
   * A fast approximative solver for the 1D Riemann problem. The solver
   * ensures that the estimate lambda_max that is returned for the maximal
   * wavespeed is a strict upper bound ensuring that all important
   * invariance principles are obtained.
   *
   * The solver is based on two publications [1,2].
   *
   * References:
   *   [1] J.-L. Guermond, B. Popov. Fast estimation from above for the
   *       maximum wave speed in the Riemann problem for the Euler equations.
   *
   *   [2] J.-L. Guermond, et al. In progress.
   */
  template <int dim, typename Number = double>
  class RiemannSolver
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    using ScalarNumber = typename get_value_type<Number>::type;

    /*
     * Options:
     */

    static constexpr unsigned int newton_max_iter_ = RIEMANN_NEWTON_MAX_ITER;

    /**
     * Try to improve the maximal wavespeed estimate. When enabled the
     * RiemannSolver also computes
     *
     *  - appropriate bounds on density and specific entropy
     *
     *  - average and flux of the state and a Harten entropy
     *
     *  - does a full limiter pass against the inviscid Galerkin update
     */
    static constexpr bool greedy_dij_ = RIEMANN_GREEDY_DIJ;

    /**
     * The above computation is obviously very expensive (similarly in cost
     * to almost a complete high-order limiter pass). Therefore, we use a
     * threshold for the greedy d_ij computation. If the variance in
     * density rho between two states is less than
     *   (1 - greedy_threshold_)/100 %
     * we simply don't do anything.
     */
    static constexpr ScalarNumber greedy_threshold_ =
        ScalarNumber(RIEMANN_GREEDY_THRESHOLD);

    static constexpr bool greedy_relax_bounds_ = RIEMANN_GREEDY_RELAX_BOUNDS;

    /**
     * For two given 1D primitive states riemann_data_i and riemann_data_j,
     * compute an estimation of an upper bound for the maximum wavespeed
     * lambda.
     *
     * If necessary, we also compute and return bounds = {rho_min, rho_max,
     * s_min, salpha_avg, salpha_flux} that are needed as bounds in the
     * limiter for the "greedy d_ij" computation.
     */
    static std::tuple<Number /*lambda_max*/,
                      Number /*p_star*/,
                      unsigned int /*iteration*/,
                      std::array<Number, 5> /*bounds*/>
    compute(const std::array<Number, 4> &riemann_data_i,
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
    compute(const rank1_type &U_i,
            const rank1_type &U_j,
            const dealii::Tensor<1, dim, Number> &n_ij,
            const Number hd_i = Number(0.));
  };

} /* namespace grendel */

#endif /* RIEMANN_SOLVER_H */
