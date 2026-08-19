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

namespace ryujin
{
  namespace Skeleton
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
      }

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
    };


    /**
     * A fast approximative solver for the associated 1D Riemann problem.
     * The solver has to ensure that the estimate
     * \f$\lambda_{\text{max}}\f$ that is returned for the maximal
     * wavespeed is a strict upper bound.
     *
     * @ingroup SkeletonEquations
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

      using state_type = typename View::state_type;

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
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimation of an upper bound for lambda.
       */
      Number compute(const PrecomputedVectorView & /*pv*/,
                     const state_type & /*U_i*/,
                     const state_type & /*U_j*/,
                     const unsigned int /*i*/,
                     const unsigned int * /*js*/,
                     const dealii::Tensor<1, dim, Number> & /*n_ij*/) const
      {
        return Number(std::numeric_limits<ScalarNumber>::epsilon());
      }

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
  } // namespace Skeleton
} // namespace ryujin
