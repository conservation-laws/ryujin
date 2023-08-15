//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace ryujin
{
  namespace ScalarConservation
  {
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
    template <int dim, typename Number = double>
    class RiemannSolver
    {
    public:
      /**
       * @copydoc HyperbolicSystem::View
       */
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::View::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

      /**
       * @copydoc HyperbolicSystem::View::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystem::View::ScalarNumber
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      RiemannSolver(
          const HyperbolicSystem &hyperbolic_system,
          const MultiComponentVector<ScalarNumber, n_precomputed_values>
              &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimate for an upper bound of lambda.
       */
      Number compute(const state_type &U_i,
                     const state_type &U_j,
                     const unsigned int i,
                     const unsigned int *js,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

      //@}
      //
    private:
      const HyperbolicSystemView hyperbolic_system;
      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;
    };
  } // namespace ScalarConservation
} // namespace ryujin
