//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>
#include <simd.h>

#include <deal.II/base/vectorization.h>


namespace ryujin
{
  namespace ScalarConservation
  {
    /**
     * An suitable indicator strategy that is used to form the preliminary
     * high-order update.
     *
     * @ingroup ScalarConservationEquations
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
       * @copydoc HyperbolicSystem::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystem::View::precomputed_state_type
       */
      using precomputed_state_type =
          typename HyperbolicSystemView::precomputed_state_type;

      /**
       * @copydoc HyperbolicSystem::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

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
      void reset(const unsigned int i, const state_type &U_i);

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
      Number alpha(const Number h_i) const;

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

      Number u_i;
      Number u_abs_max;
      dealii::Tensor<1, dim, Number> f_i;
      Number left;
      Number right;
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
      const auto prec_i =
          precomputed_values
              .template get_tensor<Number, precomputed_state_type>(i);

      /* entropy viscosity commutator: */

      u_i = hyperbolic_system.state(U_i);
      u_abs_max = std::abs(u_i);
      f_i = hyperbolic_system.construct_flux_tensor(prec_i);
      left = 0.;
      right = 0.;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Indicator<dim, Number>::accumulate(
        const unsigned int *js,
        const state_type &U_j,
        const dealii::Tensor<1, dim, Number> &c_ij)
    {
      const auto prec_j =
          precomputed_values
              .template get_tensor<Number, precomputed_state_type>(js);

      /* entropy viscosity commutator: */

      const auto u_j = hyperbolic_system.state(U_j);
      u_abs_max = std::max(u_abs_max, std::abs(u_j));
      const auto d_eta_j =
          hyperbolic_system.kruzkov_entropy_derivative(u_i, u_j);
      const auto f_j = hyperbolic_system.construct_flux_tensor(prec_j);

      left += d_eta_j * (f_j * c_ij);
      right += d_eta_j * (f_i * c_ij);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    Indicator<dim, Number>::alpha(const Number hd_i) const
    {
      Number numerator = left - right;
      Number denominator = std::abs(left) + std::abs(right);

      const auto quotient =
          std::abs(numerator) /
          (denominator +
           std::max(hd_i * std::abs(u_abs_max),
                    Number(100. * std::numeric_limits<ScalarNumber>::min())));

      return std::min(Number(1.), evc_factor * quotient);
    }

  } // namespace ScalarConservation
} // namespace ryujin
