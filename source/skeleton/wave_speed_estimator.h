//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace Skeleton
  {
    template <typename ScalarNumber = double>
    class WaveSpeedEstimator : public dealii::ParameterAcceptor
    {
    public:
      WaveSpeedEstimator(const std::string &subsection = "/WaveSpeedEstimator")
          : ParameterAcceptor(subsection)
      {
      }
    };


    /**
     * A fast approximative solver for the associated 1D Riemann problem.
     * The solver has to ensure that the estimate
     * \f$\lambda_{\text{max}}\f$ that is returned for the maximal
     * wavespeed is a strict upper bound.
     *
     * @ingroup SkeletonEquations
     */
    template <int dim, typename Number = double>
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

      using Parameters = WaveSpeedEstimator<ScalarNumber>;

      //@}
      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      WaveSpeedEstimatorView(const HyperbolicSystem &hyperbolic_system,
                             const Parameters &parameters)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
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
      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;
      //@}
    };
  } // namespace Skeleton
} // namespace ryujin
