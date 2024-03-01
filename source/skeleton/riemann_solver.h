//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace Skeleton
  {
    template <typename ScalarNumber = double>
    class RiemannSolverParameters : public dealii::ParameterAcceptor
    {
    public:
      RiemannSolverParameters(const std::string &subsection = "/RiemannSolver")
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
    class RiemannSolver
    {
    public:
      /**
       * @copydoc HyperbolicSystemView
       */
      using View = HyperbolicSystemView<dim, Number>;

      /**
       * @copydoc HyperbolicSystemView::state_type
       */
      using state_type = typename View::state_type;

      /**
       * @copydoc HyperbolicSystemView::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          View::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystemView::ScalarNumber
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * @copydoc RiemannSolverParameters
       */
      using Parameters = RiemannSolverParameters<ScalarNumber>;

      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      RiemannSolver(
          const HyperbolicSystem &hyperbolic_system,
          const Parameters &parameters,
          const MultiComponentVector<ScalarNumber, n_precomputed_values>
              &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimation of an upper bound for lambda.
       */
      Number compute(const state_type & /*U_i*/,
                     const state_type & /*U_j*/,
                     const unsigned int /*i*/,
                     const unsigned int * /*js*/,
                     const dealii::Tensor<1, dim, Number> & /*n_ij*/) const
      {
        return Number(0.);
      }

    private:
      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;
      //@}
    };
  } // namespace Skeleton
} // namespace ryujin
