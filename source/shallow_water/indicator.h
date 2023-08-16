//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>

#include <deal.II/base/vectorization.h>


namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * An suitable indicator strategy that is used to form the preliminary
     * high-order update.
     *
     * @ingroup ShallowWaterEquations
     */
    template <int dim, typename Number = double>
    class Indicator
    {
    public:
      /**
       * @copydoc HyperbolicSystem::View
       */
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::problem_dimension
       */
      static constexpr unsigned int problem_dimension =
          HyperbolicSystemView::problem_dimension;

      /**
       * @copydoc HyperbolicSystem::precomputed_state_type
       */
      using precomputed_state_type =
          typename HyperbolicSystemView::precomputed_state_type;

      /**
       * @copydoc HyperbolicSystem::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystem::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

      /**
       * @copydoc HyperbolicSystem::flux_type
       */
      using flux_type = typename HyperbolicSystemView::flux_type;

      /**
       * @copydoc HyperbolicSystem::ScalarNumber
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * @name Stencil-based computation of indicators
       *
       * Intended usage:
       * ```
       * Indicator<dim, Number> indicator;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   indicator.reset(i, U_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     indicator.accumulate(js, U_j, c_ij);
       *   }
       *   indicator.alpha(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      Indicator(const HyperbolicSystem &hyperbolic_system,
                const MultiComponentVector<ScalarNumber, n_precomputed_values>
                    &precomputed_values,
                const ScalarNumber evc_factor)
          : hyperbolic_system(hyperbolic_system)
          , precomputed_values(precomputed_values)
          , evc_factor(evc_factor)
      {
      }

      /**
       * Reset temporary storage and initialize for a new row corresponding
       * to state vector U_i.
       */
      void reset(const unsigned int /*i*/, const state_type &U_i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const unsigned int *js,
                      const state_type &U_j,
                      const dealii::Tensor<1, dim, Number> &c_ij);

      /**
       * Return the computed alpha_i value.
       */
      Number alpha(const Number h_i);

      //@}

    private:
      /**
       * @name
       */
      //@{
      const HyperbolicSystemView hyperbolic_system;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;

      const ScalarNumber evc_factor;

      Number h_i = 0.;
      Number eta_i = 0.;
      flux_type f_i;
      state_type d_eta_i;
      Number pressure_i = 0.;

      Number left = 0.;
      state_type right;
      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    Indicator<dim, Number>::reset(const unsigned int i, const state_type &U_i)
    {
      h_i = hyperbolic_system.water_depth(U_i);
      /* entropy viscosity commutator: */

      const auto &[eta_m, h_star] =
          precomputed_values
              .template get_tensor<Number, precomputed_state_type>(i);

      eta_i = eta_m;

      d_eta_i = hyperbolic_system.mathematical_entropy_derivative(U_i);
      f_i = hyperbolic_system.f(U_i);
      pressure_i = hyperbolic_system.pressure(U_i);

      left = 0.;
      right = 0.;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Indicator<dim, Number>::accumulate(
        const unsigned int *js,
        const state_type &U_j,
        const dealii::Tensor<1, dim, Number> &c_ij)
    {
      /* entropy viscosity commutator: */

      const auto &[eta_j, h_star_j] =
          precomputed_values
              .template get_tensor<Number, precomputed_state_type>(js);

      const auto velocity_j = hyperbolic_system.momentum(U_j) *
                              hyperbolic_system.inverse_water_depth_sharp(U_j);
      const auto f_j = hyperbolic_system.f(U_j);
      const auto pressure_j = hyperbolic_system.pressure(U_j);

      left += (eta_j + pressure_j) * (velocity_j * c_ij);

      for (unsigned int k = 0; k < problem_dimension; ++k)
        right[k] += (f_j[k] - f_i[k]) * c_ij;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    Indicator<dim, Number>::alpha(const Number hd_i)
    {
      Number my_sum = 0.;
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        my_sum += d_eta_i[k] * right[k];
      }

      Number numerator = std::abs(left - my_sum);
      Number denominator = std::abs(left) + std::abs(my_sum);

      const auto quotient =
          std::abs(numerator) /
          (denominator +
           std::max(hd_i * std::abs(eta_i),
                    Number(100. * std::numeric_limits<ScalarNumber>::min())));

      return std::min(Number(1.), evc_factor * quotient);
    }


  } // namespace ShallowWater
} // namespace ryujin
