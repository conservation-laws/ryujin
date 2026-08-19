//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim, typename Number = double>
    class WaveSpeedEstimatorView;

    template <typename ScalarNumber = double>
    class WaveSpeedEstimator : public dealii::ParameterAcceptor
    {
    public:
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

      ACCESSOR_READ_ONLY(use_greedy_wavespeed);
      ACCESSOR_READ_ONLY(use_averaged_entropy);
      ACCESSOR_READ_ONLY(random_entropies);

      /**
       * Alias for the view on the wave speed estimator for a given dimension @p
       * dim and choice of number type @p Number.
       */
      template <int dim, typename Number = double>
      using View = WaveSpeedEstimatorView<dim, Number>;

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
      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
      bool use_greedy_wavespeed_;
      bool use_averaged_entropy_;
      unsigned int random_entropies_;
    };


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
       * Constructor taking a HyperbolicSystemView and a parameters
       * object as arguments
       */
      WaveSpeedEstimatorView(const View &view,
                             const WaveSpeedEstimator<ScalarNumber> &parameters)
          : view(view)
          , parameters(parameters)
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
      const View view;
      const WaveSpeedEstimator<ScalarNumber> &parameters;
      //@}
    };
  } // namespace ScalarConservation
} // namespace ryujin
