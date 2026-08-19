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
  namespace ScalarConservation
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
       * Alias for the view on the indicator for a given dimension @p dim
       * and choice of number type @p Number.
       */
      template <int dim, typename Number = double>
      using View = IndicatorView<dim, Number>;

      /**
       * Return a view on the Indicator for a given dimension @p dim and
       * choice of number type @p Number (which can be a scalar float, or
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
      ScalarNumber evc_factor_;
    };


    /**
     * An suitable indicator strategy that is used to form the preliminary
     * high-order update.
     *
     * @ingroup ScalarConservationEquations
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
       * Constructor taking a HyperbolicSystemView and an Indicator
       * object as arguments
       */
      IndicatorView(const View &view, const Indicator<ScalarNumber> &indicator)
          : view_(view)
          , indicator_(indicator)
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
       * @name Internal data
       */
      //@{

      const View view_;
      const Indicator<ScalarNumber> &indicator_;

      Number u_i_;
      Number u_abs_max_;
      dealii::Tensor<1, dim, Number> f_i_;
      Number left_;
      Number right_;
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
      /* entropy viscosity commutator: */

      const auto prec_i = pv.template read_tensor<Number, precomputed_type>(i);

      u_i_ = view_.state(U_i);
      u_abs_max_ = std::abs(u_i_);
      f_i_ = view_.construct_flux_tensor(prec_i);
      left_ = 0.;
      right_ = 0.;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void IndicatorView<dim, Number>::accumulate(
        const PrecomputedVectorView &pv,
        const unsigned int *js,
        const state_type &U_j,
        const dealii::Tensor<1, dim, Number> &c_ij)
    {
      /* entropy viscosity commutator: */

      const auto prec_j = pv.template read_tensor<Number, precomputed_type>(js);

      const auto u_j = view_.state(U_j);
      u_abs_max_ = std::max(u_abs_max_, std::abs(u_j));
      const auto d_eta_j = view_.kruzkov_entropy_derivative(u_i_, u_j);
      const auto f_j = view_.construct_flux_tensor(prec_j);

      left_ += d_eta_j * (f_j * c_ij);
      right_ += d_eta_j * (f_i_ * c_ij);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    IndicatorView<dim, Number>::alpha(const Number hd_i) const
    {
      Number numerator = left_ - right_;
      Number denominator = std::abs(left_) + std::abs(right_);

      const auto regularization =
          Number(100. * std::numeric_limits<ScalarNumber>::min());

      const auto quotient =
          std::abs(numerator) /
          (denominator + std::max(hd_i * std::abs(u_abs_max_), regularization));

      return std::min(Number(1.), indicator_.evc_factor() * quotient);
    }

  } // namespace ScalarConservation
} // namespace ryujin
