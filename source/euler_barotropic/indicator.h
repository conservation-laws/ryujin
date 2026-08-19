//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>


namespace ryujin
{
  namespace EulerBarotropic
  {
    template <int dim, typename Number = double>
    class IndicatorView;

    template <typename ScalarNumber = double>
    class Indicator : public dealii::ParameterAcceptor
    {
    public:
      Indicator(const HyperbolicSystem &hyperbolic_system,
                const std::string &subsection = "/Indicator")
          : ParameterAcceptor(subsection)
          , hyperbolic_system_(&hyperbolic_system)
      {
        evc_factor_ = ScalarNumber(1.);
        add_parameter("evc factor",
                      evc_factor_,
                      "Factor for scaling the entropy viscocity commuator");
      }

      ACCESSOR_READ_ONLY(evc_factor);

      /**
       * Return a view on the Indicator for a given dimension @p dim and
       * choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars).
       */
      template <int dim, typename Number>
      auto view() const
      {
        return IndicatorView<dim, Number>{
            hyperbolic_system_->template view<dim, Number>(), *this};
      }

    private:
      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
      ScalarNumber evc_factor_;
    };


    /**
     * This class implements an indicator strategy used to form the
     * preliminary high-order update.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number>
    class IndicatorView
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

      using flux_type = typename View::flux_type;

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      using Parameters = Indicator<ScalarNumber>;

      //@}
      /**
       * @name Stencil-based computation of indicators
       *
       * Intended usage:
       * ```
       * IndicatorView<dim, Number> indicator_view;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   indicator_view.reset(pv, i, U_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     indicator_view.accumulate(pv, js, U_j, c_ij);
       *   }
       *   indicator_view.alpha(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystemView and a
       * Parameters object as arguments
       */
      IndicatorView(const View &view, const Parameters &parameters)
          : view(view)
          , parameters(parameters)
      {
      }

      /**
       * Reset temporary storage and initialize for a new row corresponding
       * to state vector U_i.
       */
      void reset(const PrecomputedVectorView &pv,
                 const unsigned int i,
                 const state_type &U_i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const PrecomputedVectorView &pv,
                      const unsigned int *js,
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

      const View view;
      const Parameters &parameters;

      Number eta_i = 0.;
      state_type d_eta_i;

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
    IndicatorView<dim, Number>::reset(const PrecomputedVectorView &pv,
                                      const unsigned int i,
                                      const state_type &U_i)
    {
      /* Entropy viscosity commutator: */

      const auto &[e_i, p_i, a_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      eta_i = view.total_energy(U_i, e_i);
      d_eta_i = view.total_energy_derivative(U_i, e_i, p_i);

      // left = sum_j F(U_j) * c_ij, where F is the mathematical entropy flux
      left = 0.;

      // right = sum_j f(U_j) * c_ij, where f is the flux of the system
      right = 0.;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void IndicatorView<dim, Number>::accumulate(
        const PrecomputedVectorView &pv,
        const unsigned int *js,
        const state_type &U_j,
        const dealii::Tensor<1, dim, Number> &c_ij)
    {
      /* Entropy viscosity commutator: */

      const auto &[e_j, p_j, a_j] =
          pv.template read_tensor<Number, precomputed_type>(js);

      const auto rho_j = view.density(U_j);
      const auto rho_j_inverse = Number(1.) / rho_j;
      const auto eta_j = view.total_energy(U_j, e_j);

      const auto m_j = view.momentum(U_j);

      const auto f_j = view.f(U_j, p_j);

      const auto entropy_flux = (eta_j + p_j) * rho_j_inverse * (m_j * c_ij);

      left += entropy_flux;
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        right[k] += f_j[k] * c_ij;
      }
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    IndicatorView<dim, Number>::alpha(const Number hd_i) const
    {
      /* Entropy viscosity commutator: */

      Number numerator = left;
      Number denominator = std::abs(left);
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        numerator -= d_eta_i[k] * right[k];
        denominator += std::abs(d_eta_i[k] * right[k]);
      }

      const auto quotient = safe_division(std::abs(numerator),
                                          denominator + hd_i * std::abs(eta_i));

      return std::min(Number(1.), parameters.evc_factor() * quotient);
    }
  } // namespace EulerBarotropic
} // namespace ryujin
