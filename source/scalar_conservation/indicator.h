//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>


namespace ryujin
{
  namespace ScalarConservation
  {
    template <typename ScalarNumber = double>
    class IndicatorParameters : public dealii::ParameterAcceptor
    {
    public:
      IndicatorParameters(const std::string &subsection = "/Indicator")
          : ParameterAcceptor(subsection)
      {
        evc_factor_ = ScalarNumber(1.);
        add_parameter("evc factor",
                      evc_factor_,
                      "Factor for scaling the entropy viscocity commuator");
      }

      ACCESSOR_READ_ONLY(evc_factor);

    private:
      ScalarNumber evc_factor_;
    };


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
       * @copydoc HyperbolicSystemView
       */
      using HyperbolicSystemView = HyperbolicSystemView<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystemView::precomputed_state_type
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
       * @copydoc IndicatorParameters
       */
      using Parameters = IndicatorParameters<ScalarNumber>;

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
                const Parameters &parameters,
                const MultiComponentVector<ScalarNumber, n_precomputed_values>
                    &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
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

      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;

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
      /* entropy viscosity commutator: */

      const auto view = hyperbolic_system.view<dim, Number>();

      const auto prec_i =
          precomputed_values
              .template get_tensor<Number, precomputed_state_type>(i);

      u_i = view.state(U_i);
      u_abs_max = std::abs(u_i);
      f_i = view.construct_flux_tensor(prec_i);
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

      const auto view = hyperbolic_system.view<dim, Number>();

      const auto prec_j =
          precomputed_values
              .template get_tensor<Number, precomputed_state_type>(js);

      const auto u_j = view.state(U_j);
      u_abs_max = std::max(u_abs_max, std::abs(u_j));
      const auto d_eta_j = view.kruzkov_entropy_derivative(u_i, u_j);
      const auto f_j = view.construct_flux_tensor(prec_j);

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

      return std::min(Number(1.), parameters.evc_factor() * quotient);
    }

  } // namespace ScalarConservation
} // namespace ryujin
