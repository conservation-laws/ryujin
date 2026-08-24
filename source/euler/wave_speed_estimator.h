//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <gpu.h>
#include <newton.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// #define DEBUG_WAVE_SPEED_ESTIMATOR

namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class WaveSpeedEstimatorView;

    /**
     * A fast approximative solver for the 1D Riemann problem. The solver
     * ensures that the estimate \f$\lambda_{\text{max}}\f$ that is returned
     * for the maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite GuermondPopov2016b.
     *
     * @ingroup EulerEquations
     */
    template <typename ScalarNumber = double>
    class WaveSpeedEstimator : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * A structure holding all runtime parameters of the wave speed
       * estimator.
       */
      struct Parameters {
        double newton_tolerance;
        unsigned int newton_max_iterations;
      };

      /**
       * Alias for the view on the wave speed estimator for a given
       * dimension @p dim, choice of number type @p Number, and memory
       * space @p MemorySpace.
       */
      template <int dim,
                typename Number = double,
                typename MemorySpace = dealii::MemorySpace::Host>
      using View = WaveSpeedEstimatorView<dim, Number, MemorySpace>;

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      WaveSpeedEstimator(const HyperbolicSystem &hyperbolic_system,
                         const std::string &subsection = "/WaveSpeedEstimator")
          : ParameterAcceptor(subsection)
          , parameters_("euler_wave_speed_estimator_parameters",
                        TransferPolicy::implicit_transfers)
          , hyperbolic_system_(&hyperbolic_system)
      {
        /*
         * Note: We bind the parameters directly to the storage held by the
         * Mirrored object. The corresponding memory is allocated once in
         * the constructor and never reallocated, so the addresses remain
         * valid for the lifetime of this object.
         */
        auto &parameters = *parameters_.view();

        if constexpr (std::is_same<ScalarNumber, double>::value)
          parameters.newton_tolerance = 1.e-10;
        else
          parameters.newton_tolerance = 1.e-4;
        add_parameter("newton tolerance",
                      parameters.newton_tolerance,
                      "Tolerance for the quadratic newton stopping criterion");

        parameters.newton_max_iterations = 0;
        add_parameter("newton max iterations",
                      parameters.newton_max_iterations,
                      "Maximal number of quadratic newton iterations performed "
                      "during limiting");
      }

      /**
       * Return a view on the WaveSpeedEstimator for a given dimension @p dim
       * and choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars). The
       * optional @p MemorySpace template parameter selects whether the
       * view is intended for the host or device memory space.
       */
      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return View<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
      }

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      Mirrored<Parameters> parameters_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      //@}

      template <int, typename, typename>
      friend class WaveSpeedEstimatorView;
    };


    /**
     * A view of the WaveSpeedEstimator that makes the interface available
     * for a given dimension @p dim and choice of number type @p Number
     * (which can be a scalar float, or double, as well as a VectorizedArray
     * holding packed scalars).
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename MemorySpace>
    class WaveSpeedEstimatorView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

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

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      //@}
      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystemView and a WaveSpeedEstimator
       * object as arguments
       */
      WaveSpeedEstimatorView(
          const View &view,
          const WaveSpeedEstimator<ScalarNumber> &wave_speed_estimator)
          : view_(view)
          , parameters_(
                wave_speed_estimator.parameters_.template view<MemorySpace>())
      {
      }

      /**
       * Return the tolerance for the quadratic Newton stopping criterion.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber newton_tolerance() const
      {
        return ScalarNumber(parameters_->newton_tolerance);
      }

      /**
       * Return the maximal number of quadratic Newton iterations.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
      newton_max_iterations() const
      {
        return parameters_->newton_max_iterations;
      }

      /**
       * For two given 1D primitive states riemann_data_i and riemann_data_j,
       * compute an estimation of an upper bound for the maximum wavespeed
       * lambda.
       */
      DEAL_II_HOST_DEVICE Number
      compute(const primitive_type &riemann_data_i,
              const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimation of an upper bound for lambda.
       *
       * Returns a tuple consisting of lambda max and the number of Newton
       * iterations used in the solver to find it.
       */
      DEAL_II_HOST_DEVICE Number
      compute(const PrecomputedVectorView &pv,
              const state_type &U_i,
              const state_type &U_j,
              const unsigned int i,
              const unsigned int *js,
              const dealii::Tensor<1, dim, Number> &n_ij) const;

      //@}

    protected:
      /**
       * @name Internal methods
       */
      //@{

      /**
       * See @cite GuermondPopov2016b, page 912, (3.4).
       *
       * Cost: 1x pow, 1x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number f(const primitive_type &riemann_data,
                                   const Number p_star) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.4).
       *
       * Cost: 1x pow, 3x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number df(const primitive_type &riemann_data,
                                    const Number &p_star) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.3).
       *
       * Cost: 2x pow, 6x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number phi(const primitive_type &riemann_data_i,
                                     const primitive_type &riemann_data_j,
                                     const Number p_in) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.3).
       *
       * Cost: 2x pow, 6x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number dphi(const primitive_type &riemann_data_i,
                                      const primitive_type &riemann_data_j,
                                      const Number &p) const;


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
      DEAL_II_HOST_DEVICE Number
      phi_of_p_max(const primitive_type &riemann_data_i,
                   const primitive_type &riemann_data_j) const;


      /**
       * see @cite GuermondPopov2016b, page 912, (3.7)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number lambda1_minus(
          const primitive_type &riemann_data, const Number p_star) const;


      /**
       * see @cite GuermondPopov2016b, page 912, (3.8)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number lambda3_plus(
          const primitive_type &primitive_state, const Number p_star) const;


      /**
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and two guesses p_1 <= p* <= p_2,
       * compute the gap in lambda between both guesses.
       *
       * See @cite GuermondPopov2016b, page 914, (4.4a), (4.4b), (4.5), and
       * (4.6)
       *
       * Cost: 0x pow, 4x division, 4x sqrt
       */
      DEAL_II_HOST_DEVICE std::array<Number, 2>
      compute_gap(const primitive_type &riemann_data_i,
                  const primitive_type &riemann_data_j,
                  const Number p_1,
                  const Number p_2) const;


      /**
       * see @cite GuermondPopov2016b, page 912, (3.9)
       *
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
       * for lambda.
       *
       * Cost: 0x pow, 2x division, 2x sqrt (inclusive)
       */
      DEAL_II_HOST_DEVICE Number
      compute_lambda(const primitive_type &riemann_data_i,
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
      DEAL_II_HOST_DEVICE Number
      p_star_two_rarefaction(const primitive_type &riemann_data_i,
                             const primitive_type &riemann_data_j) const;

      /**
       * Failsafe approximation to p_star computed for two primitive states
       * <code>riemann_data_i</code> and <code>riemann_data_j</code>.
       *
       * See @cite ClaytonGuermondPopov-2022, (5.11):
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      p_star_failsafe(const primitive_type &riemann_data_i,
                      const primitive_type &riemann_data_j) const;


      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, and a
       * (normalized) "direction" n_ij, first compute the corresponding
       * projected state in the corresponding 1D Riemann problem, and then
       * compute and return the Riemann data [rho, u, p, a] (used in the
       * approximative Riemann solver).
       */
      DEAL_II_HOST_DEVICE primitive_type
      riemann_data_from_state(const state_type &U,
                              const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;
      const WaveSpeedEstimator<ScalarNumber>::Parameters *const parameters_;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      /*
       * For exactly solving the Riemann problem we need to start with a
       * good upper and lower bound, p_1 <= p_star <= p_2, for finding
       * phi(p_star) == 0. This implies that we have to ensure that
       * phi(p_2) >= 0 and phi(p_1) <= 0.
       *
       * Instead of solving the Riemann problem exactly, however we will
       * simply use the upper bound p_2 (with p_2 >= p_star) to compute
       * lambda_max and return the estimate.
       *
       * We will use three candidates, p_min, p_max and the two rarefaction
       * approximation p_star_tilde. We have (up to round-off errors) that
       * phi(p_star_tilde) >= 0. So this is a safe upper bound, it might
       * just be too large.
       *
       * Depending on the sign of phi(p_max) we select the following ranges:
       *
       *   phi(p_max) <  0:
       *     p_1  <-  p_max   and   p_2  <-  p_star_tilde
       *
       *   phi(p_max) >= 0:
       *     p_1  <-  p_min   and   p_2  <-  min(p_max, p_star_tilde)
       *
       * Nota bene:
       *
       *  - The special case phi(p_max) == 0 as discussed in [1] is already
       *    contained in the second condition.
       *
       *  - In principle, we would have to treat the case phi(p_min) > 0 as
       *    well. This corresponds to two expansion waves and a good
       *    estimate for the wavespeed is obtained by simply computing
       *    lambda_max with p_2 = 0.
       *
       *    However, it turns out that numerically in this case we will
       *    have
       *
       *      0 < p_star <= p_star_tilde <= p_min <= p_max.
       *
       *    So it is sufficient to end up with p_2 = p_star_tilde (!!) to
       *    compute the exact same wave speed as for p_2 = 0.
       *
       *    Note: If for some reason p_star should be computed exactly,
       *    then p_1 has to be set to zero. This can be done efficiently by
       *    simply checking for p_2 < p_1 and setting p_1 <- 0 if
       *    necessary.
       */

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "rho_left: " << rho_i << std::endl;
      std::cout << "u_left: " << u_i << std::endl;
      std::cout << "p_left: " << p_i << std::endl;
      std::cout << "a_left: " << a_i << std::endl;
      std::cout << "rho_right: " << rho_j << std::endl;
      std::cout << "u_right: " << u_j << std::endl;
      std::cout << "p_right: " << p_j << std::endl;
      std::cout << "a_right: " << a_j << std::endl;
#endif

      const Number p_max = std::max(p_i, p_j);

      const Number rarefaction =
          p_star_two_rarefaction(riemann_data_i, riemann_data_j);
      const Number failsafe = p_star_failsafe(riemann_data_i, riemann_data_j);
      const Number p_star_tilde = std::min(rarefaction, failsafe);

      const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);

      Number p_2 =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max,
              Number(0.),
              p_star_tilde,
              std::min(p_max, p_star_tilde));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "   p^*_tilde  = " << p_2 << "\n";
      std::cout << "   phi(p_*_t) = "
                << phi(riemann_data_i, riemann_data_j, p_2) << std::endl;
#endif

      /*
       * If we do no Newton iteration, cut it short:
       */

      if (newton_max_iterations() == 0) {
        const auto lambda_max =
            compute_lambda(riemann_data_i, riemann_data_j, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "-> lambda_max = " << lambda_max << std::endl;
#endif
        return lambda_max;
      }

      /*
       * Compute p_1 and ensure that p_1 < p_2. If we hit a case with two
       * expansions we might indeed have that p_star_tilde < p_1. Set p_1 =
       * p_2 in this case.
       */

      const Number p_min = std::min(riemann_data_i[2], riemann_data_j[2]);

      Number p_1 =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max, Number(0.), p_max, p_min);

      p_1 = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(p_1, p_2, p_1, p_2);

      /*
       * Step 2: Perform quadratic Newton iteration.
       *
       * See [1], p. 915f (4.8) and (4.9)
       */

      auto [gap, lambda_max] =
          compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << std::fixed << std::setprecision(16);
      std::cout << "p_1: (start) " << p_1 << std::endl;
      std::cout << "p_2: (start) " << p_2 << std::endl;
      std::cout << "gap: (start) " << gap << std::endl;
      std::cout << "l_m: (start) " << lambda_max << std::endl;
#endif

      for (unsigned int i = 0; i < newton_max_iterations(); ++i) {

        /* We accept our current guess if we reach the tolerance... */
        const Number tolerance(newton_tolerance());
        if (std::max(Number(0.), gap - tolerance) == Number(0.)) {
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
          std::cout << "converged after " << i << " iterations." << std::endl;
#endif
          break;
        }

        // FIXME: Fuse these computations:
        const Number phi_p_1 = phi(riemann_data_i, riemann_data_j, p_1);
        const Number phi_p_2 = phi(riemann_data_i, riemann_data_j, p_2);
        const Number dphi_p_1 = dphi(riemann_data_i, riemann_data_j, p_1);
        const Number dphi_p_2 = dphi(riemann_data_i, riemann_data_j, p_2);

        quadratic_newton_step(p_1, p_2, phi_p_1, phi_p_2, dphi_p_1, dphi_p_2);

        /* Update  lambda_max and gap: */
        auto [gap_new, lambda_max_new] =
            compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);
        gap = gap_new;
        lambda_max = lambda_max_new;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "phi_p_1:     " << phi_p_1 << std::endl;
        std::cout << "phi_p_2:     " << phi_p_2 << std::endl;
        std::cout << "dphi_p_1:    " << dphi_p_1 << std::endl;
        std::cout << "dphi_p_2:    " << dphi_p_2 << std::endl;
        std::cout << "p_1: (  " << i << "  ) " << p_1 << std::endl;
        std::cout << "p_2: (  " << i << "  ) " << p_2 << std::endl;
        std::cout << "gap:         " << gap << std::endl;
        std::cout << "l_m:         " << lambda_max << std::endl;
#endif
      }

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "-> lambda_max = " << lambda_max << std::endl;
#endif

      return lambda_max;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute(
        const PrecomputedVectorView & /*pv*/,
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::f(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto &gamma = view_.gamma();

      const auto &[rho, u, p, a] = riemann_data;

      const Number Az = ScalarNumber(2.) / (rho * (gamma + Number(1.)));
      const Number Bz =
          (gamma - ScalarNumber(1.)) / (gamma + ScalarNumber(1.)) * p;
      const Number radicand = Az / (p_star + Bz);
      const Number true_value = (p_star - p) * std::sqrt(radicand);

      const auto exponent =
          ScalarNumber(0.5) * (gamma - ScalarNumber(1.)) / gamma;
      const Number factor = ryujin::pow(p_star / p, exponent) - Number(1.);
      const auto false_value =
          ScalarNumber(2.) * a * factor / (gamma - ScalarNumber(1.));

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::df(
        const primitive_type &riemann_data, const Number &p_star) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;
      const auto &gamma = view_.gamma();
      const auto &gamma_inverse = view_.gamma_inverse();
      const auto &gamma_minus_one_inverse = view_.gamma_minus_one_inverse();
      const auto &gamma_plus_one_inverse = view_.gamma_plus_one_inverse();

      const auto &[rho, u, p, a] = riemann_data;

      const Number radicand_inverse = ScalarNumber(0.5) * rho *
                                      ((gamma + ScalarNumber(1.)) * p_star +
                                       (gamma - ScalarNumber(1.)) * p);
      const Number denominator =
          (p_star + (gamma - ScalarNumber(1.)) * gamma_plus_one_inverse * p);
      const Number true_value =
          (denominator - ScalarNumber(0.5) * (p_star - p)) /
          (denominator * std::sqrt(radicand_inverse));

      const auto exponent =
          (ScalarNumber(-1.) - gamma) * ScalarNumber(0.5) * gamma_inverse;
      const Number factor = (gamma - ScalarNumber(1.)) * ScalarNumber(0.5) *
                            gamma_inverse * ryujin::pow(p_star / p, exponent) /
                            p;
      const auto false_value =
          factor * ScalarNumber(2.) * a * gamma_minus_one_inverse;

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::phi(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_in) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f(riemann_data_i, p_in) + f(riemann_data_j, p_in) + u_j - u_i;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::dphi(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number &p) const
    {
      return df(riemann_data_i, p) + df(riemann_data_j, p);
    }


    /*
     * The approximate Riemann solver is based on a function phi(p) that is
     * montone increasing in p, concave down and whose (weak) third
     * derivative is non-negative and locally bounded [1, p. 912]. Because we
     * actually do not perform any iteration for computing our wavespeed
     * estimate we can get away by only implementing a specialized variant of
     * the phi function that computes phi(p_max). It inlines the
     * implementation of the "f" function and eliminates all unnecessary
     * branches in "f".
     *
     * Cost: 0x pow, 2x division, 2x sqrt
     */
    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::phi_of_p_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &gamma = view_.gamma();

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

      const Number p_max = std::max(p_i, p_j);

      const Number radicand_inverse_i = ScalarNumber(0.5) * rho_i *
                                        ((gamma + ScalarNumber(1.)) * p_max +
                                         (gamma - ScalarNumber(1.)) * p_i);

      const Number value_i = (p_max - p_i) / std::sqrt(radicand_inverse_i);

      const Number radicand_inverse_j = ScalarNumber(0.5) * rho_j *
                                        ((gamma + ScalarNumber(1.)) * p_max +
                                         (gamma - ScalarNumber(1.)) * p_j);

      const Number value_j = (p_max - p_j) / std::sqrt(radicand_inverse_j);

      return value_i + value_j + u_j - u_i;
    }


    /*
     * Next we construct approximations for the two extreme wave speeds of
     * the Riemann fan [1, p. 912, (3.7) + (3.8)] and compute an upper bound
     * lambda_max of the maximal wave speed:
     */


    /*
     * see [1], page 912, (3.7)
     *
     * Cost: 0x pow, 1x division, 1x sqrt
     */
    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::lambda1_minus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto &gamma = view_.gamma();
      const auto &gamma_inverse = view_.gamma_inverse();
      const auto factor =
          (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;

      const auto &[rho, u, p, a] = riemann_data;
      const auto inv_p = ScalarNumber(1.0) / p;

      const Number tmp = positive_part((p_star - p) * inv_p);

      return u - a * std::sqrt(ScalarNumber(1.0) + factor * tmp);
    }


    /*
     * see [1], page 912, (3.8)
     *
     * Cost: 0x pow, 1x division, 1x sqrt
     */
    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::lambda3_plus(
        const primitive_type &primitive_state, const Number p_star) const
    {
      const auto &gamma = view_.gamma();
      const auto &gamma_inverse = view_.gamma_inverse();
      const Number factor =
          (gamma + ScalarNumber(1.0)) * ScalarNumber(0.5) * gamma_inverse;

      const auto &[rho, u, p, a] = primitive_state;
      const auto inv_p = ScalarNumber(1.0) / p;

      const Number tmp = positive_part((p_star - p) * inv_p);
      return u + a * std::sqrt(Number(1.0) + factor * tmp);
    }


    /**
     * For two given primitive states <code>riemann_data_i</code> and
     * <code>riemann_data_j</code>, and two guesses p_1 <= p* <= p_2,
     * compute the gap in lambda between both guesses.
     *
     * See [1], page 914, (4.4a), (4.4b), (4.5), and (4.6)
     *
     * Cost: 0x pow, 4x division, 4x sqrt
     */
    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE std::array<Number, 2>
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute_gap(
        const std::array<Number, 4> &riemann_data_i,
        const std::array<Number, 4> &riemann_data_j,
        const Number p_1,
        const Number p_2) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_2 /*SIC!*/);
      const Number nu_12 = lambda1_minus(riemann_data_i, p_1 /*SIC!*/);

      const Number nu_31 = lambda3_plus(riemann_data_j, p_1);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_2);

      const Number lambda_max =
          std::max(positive_part(nu_32), negative_part(nu_11));

      const Number gap =
          std::max(std::abs(nu_32 - nu_31), std::abs(nu_12 - nu_11));

      return {{gap, lambda_max}};
    }


    /*
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
    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute_lambda(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_star) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

      return std::max(positive_part(nu_32), negative_part(nu_11));
    }


    /*
     * Two-rarefaction approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
     *
     * See [1], page 914, (4.3)
     *
     * Cost: 2x pow, 2x division, 0x sqrt
     */
    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::p_star_two_rarefaction(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &gamma = view_.gamma();
      const auto &gamma_inverse = view_.gamma_inverse();
      const auto &gamma_minus_one_inverse = view_.gamma_minus_one_inverse();

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;
      const auto inv_p_j = ScalarNumber(1.) / p_j;

      /*
       * Nota bene (cf. [1, (4.3)]):
       *   a_Z^0 * sqrt(1 - b * rho_Z) = a_Z * (1 - b * rho_Z)
       * We have computed a_Z already, so we are simply going to use this
       * identity below:
       */

      const auto factor = (gamma - ScalarNumber(1.)) * ScalarNumber(0.5);

      /*
       * Nota bene (cf. [1, (3.6)]: The condition "numerator > 0" is the
       * well-known non-vacuum condition. In case we encounter numerator <= 0
       * then p_star = 0 is the correct pressure to compute the wave speed.
       * Therefore, all we have to do is to take the positive part of the
       * expression:
       */

      const Number numerator = positive_part(a_i + a_j - factor * (u_j - u_i));
      const Number denominator =
          a_i * ryujin::pow(p_i * inv_p_j, -factor * gamma_inverse) + a_j;

      const auto exponent = ScalarNumber(2.0) * gamma * gamma_minus_one_inverse;

      const auto p_1_tilde =
          p_j * ryujin::pow(numerator / denominator, exponent);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_two_rarefaction = " << p_1_tilde << std::endl;
#endif
      return p_1_tilde;
    }


    /*
     * Failsafe approximation to p_star computed for two primitive
     * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
     *
     * See [1], page 914, (4.3)
     *
     * Cost: 2x pow, 2x division, 0x sqrt
     */
    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::p_star_failsafe(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &gamma = view_.gamma();

      const auto &[rho_i, u_i, p_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, a_j] = riemann_data_j;

      /*
       * Compute (5.11) formula for \tilde p_2^\ast:
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */

      const Number p_max = std::max(p_i, p_j);

      Number radicand_i = ScalarNumber(2.) * p_max;
      radicand_i /=
          rho_i * ((gamma + Number(1.)) * p_max + (gamma - Number(1.)) * p_i);

      const Number x_i = std::sqrt(radicand_i);

      Number radicand_j = ScalarNumber(2.) * p_max;
      radicand_j /=
          rho_j * ((gamma + Number(1.)) * p_max + (gamma - Number(1.)) * p_j);

      const Number x_j = std::sqrt(radicand_j);

      const Number a = x_i + x_j;
      const Number b = u_j - u_i;
      const Number c = -p_i * x_i - p_j * x_j;

      const Number base = (-b + std::sqrt(b * b - ScalarNumber(4.) * a * c)) /
                          (ScalarNumber(2.) * a);
      const Number p_2_tilde = base * base;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_failsafe = " << p_2_tilde << std::endl;
#endif
      return p_2_tilde;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::riemann_data_from_state(
        const state_type &U, const dealii::Tensor<1, dim, Number> &n_ij) const
        -> primitive_type
    {
      const auto rho = view_.density(U);
      const auto rho_inverse = Number(1.0) / rho;

      const auto m = view_.momentum(U);
      const auto proj_m = n_ij * m;
      const auto perp = m - proj_m * n_ij;

      const auto E = view_.total_energy(U) -
                     Number(0.5) * perp.norm_square() * rho_inverse;

      /*
       * Compute the pressure and speed of sound of the projected
       * one-dimensional state [rho, proj_m, E]:
       */
      const auto gamma = view_.gamma();
      const auto internal_energy =
          E - ScalarNumber(0.5) * (proj_m * proj_m) * rho_inverse;
      const auto p = (gamma - ScalarNumber(1.)) * internal_energy;
      const auto a = std::sqrt(gamma * p * rho_inverse);

      return {{rho, proj_m * rho_inverse, p, a}};
    }
  } // namespace Euler
} // namespace ryujin
