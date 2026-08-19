//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2023 - 2026 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <multicomponent_vector.h>
#include <observer_pointer.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>


namespace ryujin
{
  namespace ShallowWater
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
     * @ingroup ShallowWaterEquations
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
                 const unsigned int /*i*/,
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
      Number alpha(const Number h_i);


    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;
      const Indicator<ScalarNumber> &indicator_;

      Number h_i_ = 0.;
      Number eta_i_ = 0.;
      flux_type f_i_;
      state_type d_eta_i_;
      Number pressure_i_ = 0.;

      Number left_ = 0.;
      state_type right_;
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

      const auto &[eta_m, h_star] =
          pv.template read_tensor<Number, precomputed_type>(i);

      h_i_ = view_.water_depth(U_i);
      eta_i_ = eta_m;
      d_eta_i_ = view_.mathematical_entropy_derivative(U_i);
      f_i_ = view_.f(U_i);
      pressure_i_ = view_.pressure(U_i);

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

      const auto &[eta_j, h_star_j] =
          pv.template read_tensor<Number, precomputed_type>(js);

      const auto velocity_j =
          view_.momentum(U_j) * view_.inverse_water_depth_sharp(U_j);
      const auto f_j = view_.f(U_j);
      const auto pressure_j = view_.pressure(U_j);

      left_ += (eta_j + pressure_j) * (velocity_j * c_ij);

      for (unsigned int k = 0; k < problem_dimension; ++k)
        right_[k] += (f_j[k] - f_i_[k]) * c_ij;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    IndicatorView<dim, Number>::alpha(const Number hd_i)
    {
      Number my_sum = 0.;
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        my_sum += d_eta_i_[k] * right_[k];
      }

      Number numerator = std::abs(left_ - my_sum);
      Number denominator = std::abs(left_) + std::abs(my_sum);

      const auto regularization =
          Number(100. * std::numeric_limits<ScalarNumber>::min());

      const auto quotient =
          std::abs(numerator) /
          (denominator + std::max(hd_i * std::abs(eta_i_), regularization));

      return std::min(Number(1.), indicator_.evc_factor() * quotient);
    }


  } // namespace ShallowWater
} // namespace ryujin
