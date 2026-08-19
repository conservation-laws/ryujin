//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace EulerBarotropic
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
     * Specialized approximative solver for the 1D Riemann problem of the
     * barotropic Euler equations. The solver ensures that the estimate
     * \f$\lambda_{\text{max}}\f$ that is returned by compute() is a
     * guaranteed upper bound of the maximal wavespeed.
     *
     * @ingroup EulerEquations
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

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      /**
       * Number of components in a primitive state, we store \f$[v, a]\f$.
       */
      static constexpr unsigned int riemann_data_size = 2;

      /**
       * The array type to store the primitive state for the Riemann solver
       * \f$[v, a]\f$
       */
      using primitive_type = typename std::array<Number, riemann_data_size>;

      using precomputed_type = typename View::precomputed_type;

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
       * For two given 1D primitive states riemann_data_i and
       * riemann_data_j, compute an estimate for an upper bound of the
       * maximum wavespeed lambda.
       */
      Number compute(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimate for an upper bound of the maximum wavespeed
       * lambda.
       */
      Number compute(const PrecomputedVectorView &pv,
                     const state_type &U_i,
                     const state_type &U_j,
                     const unsigned int i,
                     const unsigned int *js,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;
      //@}

    protected:
      /** @name Internal functions used in the Riemann solver */
      //@{

    private:
      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;
      //@}
    };
  } // namespace EulerBarotropic
} /* namespace ryujin */
