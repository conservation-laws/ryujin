//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <random>

// #define DEBUG_WAVE_SPEED_ESTIMATOR

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim, typename Number = double>
    class WaveSpeedEstimatorView;

    /**
     * A fast estimate for a sufficient maximal wavespeed of the 1D Riemann
     * problem. The wavespeed estimate is based on a guaranteed upper bound
     * on the maximal wavespeed for convex fluxes, see Example 79.17 on
     * page 333 of @cite GuermondErn2021. As well as an augmented "Roe
     * average" based on an entropy inequality of a suitable Krŭzkov
     * entropy, see @cite ryujin-2023-5 Section 4.
     *
     * @ingroup ScalarConservationEquations
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
       * Alias for the view on the wave speed estimator for a given dimension @p
       * dim and choice of number type @p Number.
       */
      template <int dim, typename Number = double>
      using View = WaveSpeedEstimatorView<dim, Number>;

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
          , hyperbolic_system_(&hyperbolic_system)
      {
        use_greedy_wavespeed_ = false;
        add_parameter("use greedy wavespeed",
                      use_greedy_wavespeed_,
                      "Use a greedy wavespeed estimate instead of a guaranteed "
                      "upper bound "
                      "on the maximal wavespeed (for convex fluxes).");

        use_averaged_entropy_ = false;
        add_parameter("use averaged entropy",
                      use_averaged_entropy_,
                      "In addition to the wavespeed estimate based on the Roe "
                      "average and "
                      "flux gradients of the left and right state also enforce "
                      "an entropy "
                      "inequality on the averaged Krŭzkov entropy.");

        random_entropies_ = 0;
        add_parameter(
            "random entropies",
            random_entropies_,
            "In addition to the wavespeed estimate based on the Roe average "
            "and "
            "flux gradients of the left and right state also enforce an "
            "entropy "
            "inequality on the prescribed number of random Krŭzkov entropies.");
      }

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      ACCESSOR_READ_ONLY(use_greedy_wavespeed);
      ACCESSOR_READ_ONLY(use_averaged_entropy);
      ACCESSOR_READ_ONLY(random_entropies);

      /**
       * Return a view on the WaveSpeedEstimator for a given dimension @p dim
       * and choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars).
       */
      template <int dim, typename Number>
      auto view() const
      {
        return View<dim, Number>{
            hyperbolic_system_->template view<dim, Number>(), *this};
      }

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      bool use_greedy_wavespeed_;
      bool use_averaged_entropy_;
      unsigned int random_entropies_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      //@}
    };


    /**
     * A view of the WaveSpeedEstimator that makes the interface available
     * for a given dimension @p dim and choice of number type @p Number
     * (which can be a scalar float, or double, as well as a VectorizedArray
     * holding packed scalars).
     *
     * @ingroup ScalarConservationEquations
     */
    template <int dim, typename Number>
    class WaveSpeedEstimatorView
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

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
          , wave_speed_estimator_(wave_speed_estimator)
      {
      }

      /**
       * For two states @p u_i, @p u_j, precomputed values @p prec_i,
       * @p prec_j, and a (normalized) "direction" n_ij
       * compute an upper bound estimate for the wavespeed.
       */
      Number compute(const Number &u_i,
                     const Number &u_j,
                     const precomputed_type &prec_i,
                     const precomputed_type &prec_j,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimate for an upper bound of lambda.
       */
      Number compute(const PrecomputedVectorView &pv,
                     const state_type &U_i,
                     const state_type &U_j,
                     const unsigned int i,
                     const unsigned int *js,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;
      const WaveSpeedEstimator<ScalarNumber> &wave_speed_estimator_;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    inline Number WaveSpeedEstimatorView<dim, Number>::compute(
        const Number &u_i,
        const Number &u_j,
        const precomputed_type &prec_i,
        const precomputed_type &prec_j,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      /* Project all fluxes to 1D: */
      const Number f_i = view_.construct_flux_tensor(prec_i) * n_ij;
      const Number f_j = view_.construct_flux_tensor(prec_j) * n_ij;
      const Number df_i = view_.construct_flux_gradient_tensor(prec_i) * n_ij;
      const Number df_j = view_.construct_flux_gradient_tensor(prec_j) * n_ij;

      const auto h2 = Number(2. * view_.derivative_approximation_delta());

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "\nu_i  = " << u_i << std::endl;
      std::cout << "u_j  = " << u_j << std::endl;
      std::cout << "f_i  = " << f_i << std::endl;
      std::cout << "f_j  = " << f_j << std::endl;
      std::cout << "df_i = " << df_i << std::endl;
      std::cout << "df_j = " << df_j << std::endl;
#endif

      /*
       * The Roe average with a regularization based on $h$ which is the
       * step size used for the central difference approximation of f'(u).
       *
       * The regularization max(|u_i - u_j|, 2 * h) ensures that the
       * quotient approximates the derivative f'( (u_i + u_j)/2 ) to the
       * same precision that we use to compute f'(u_i) and f'(u_j) in the
       * FunctionParser (via a central difference approximation).
       *
       * This implies that in contrast to the actual limit of the
       * difference quotient we will approach 0 as |u_j - u_i| goes to
       * zero. We fix this by taking the maximum with our approximation of
       * f'(u_i) and f'(u_j) further down below.
       */

      auto lambda_max = std::abs(f_i - f_j) / std::max(std::abs(u_i - u_j), h2);
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "   Roe average       = " << lambda_max << std::endl;
#endif

      constexpr auto gte = dealii::SIMDComparison::greater_than_or_equal;

      if (wave_speed_estimator_.use_greedy_wavespeed()) {
        /*
         * In case of a greedy estimate we make sure that we always use the
         * Roe average and only fall back to the derivative approximation
         * when u_i and u_j are close to each other within 2h:
         */
        lambda_max = dealii::compare_and_apply_mask<gte>(
            std::abs(u_i - u_j),
            h2,
            lambda_max,
            /* Approximate derivative in centerpoint: */
            std::abs(ScalarNumber(0.5) * (df_i + df_j)));
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "   interpolated      = "
                  << std::abs(ScalarNumber(0.5) * (df_i + df_j)) << std::endl;
#endif

      } else {
        /*
         * Always take the maximum with |f'(u_i)| and |f'(u_j)|.
         *
         * For convex fluxes this implies that lambda_max is indeed the
         * maximal wavespeed of the system. See Example 79.17 in reference
         * @cite ErnGuermond2021.
         */
        lambda_max = std::max(lambda_max, std::abs(df_i));
        lambda_max = std::max(lambda_max, std::abs(df_j));
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "   left  derivative  = " << std::abs(df_i) << std::endl;
        std::cout << "   right derivative  = " << std::abs(df_j) << std::endl;
#endif
      }

      /*
       * Thread-local helper lambda to generate a random number in [0,1]:
       */

      thread_local static const auto draw = []() {
        static std::random_device random_device;
        static auto generator = std::default_random_engine(random_device());
        static std::uniform_real_distribution<ScalarNumber> dist(0., 1.);

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          /*
           * Scalar quantity:
           */
          return dist(generator);

        } else {
          /*
           * Populate a vectorized array:
           */
          Number result;
          for (unsigned int s = 0; s < Number::size(); ++s)
            result[s] = dist(generator);
          return result;
        }
      };

      /*
       * Helper functions for enforcing entropy inequalities:
       */

      const auto enforce_entropy = [&](const Number &k) {
        const Number f_k = view_.flux_function(k) * n_ij;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "k    = " << k << std::endl;
        std::cout << "f_k  = " << f_k << std::endl;
#endif

        const Number eta_i = view_.kruzkov_entropy(k, u_i);
        const Number q_i =
            view_.kruzkov_entropy_derivative(k, u_i) * (f_i - f_k);

        const Number eta_j = view_.kruzkov_entropy(k, u_j);
        const Number q_j =
            view_.kruzkov_entropy_derivative(k, u_j) * (f_j - f_k);

        const Number a = u_i + u_j - ScalarNumber(2.) * k;
        const Number b = f_j - f_i;
        const Number c = eta_i + eta_j;
        const Number d = q_j - q_i;

        /*
         * FIXME: Ordinarily, lambda_left and lambda_right would be
         * computed without taking the absolute value of the numerator.
         * (The denominator is - in the absence of rounding errors - always
         * nonnegative. The numerator has a sign.)
         * But empirically it turns out that taking the absolute value and
         * letting both estimates participate in the maximal wavespeed
         * estimate helps a lot.
         */
        const Number lambda_left = std::abs(d + b) / (std::abs(c + a) + h2);
        const Number lambda_right = std::abs(d - b) / (std::abs(c - a) + h2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "   left  wavespeed   = " << lambda_left << std::endl;
        std::cout << "   right wavespeed   = " << lambda_right << std::endl;
#endif
        lambda_max = std::max(lambda_max, lambda_left);
        lambda_max = std::max(lambda_max, lambda_right);
      };


      if (wave_speed_estimator_.use_averaged_entropy()) {
        const Number k = ScalarNumber(0.5) * (u_i + u_j);
        enforce_entropy(k);
      }

      const unsigned int n_entropies = wave_speed_estimator_.random_entropies();
      for (unsigned int i = 0; i < n_entropies; ++i) {
        const Number factor = draw();
        const Number k = factor * u_i + (Number(1.) - factor) * u_j;
        enforce_entropy(k);
      }

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "-> lambda_max        = " << lambda_max << std::endl;
#endif
      return lambda_max;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    WaveSpeedEstimatorView<dim, Number>::compute(
        const PrecomputedVectorView &pv,
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int i,
        const unsigned int *js,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      using pst = typename View::precomputed_type;

      const auto u_i = view_.state(U_i);
      const auto u_j = view_.state(U_j);

      const auto prec_i = pv.template read_tensor<Number, pst>(i);
      const auto prec_j = pv.template read_tensor<Number, pst>(js);

      return compute(u_i, u_j, prec_i, prec_j, n_ij);
    }
  } // namespace ScalarConservation
} // namespace ryujin
