//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <gpu.h>
#include <multicomponent_vector.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class IndicatorView;

    /**
     * A suitable indicator strategy that is used to form the preliminary
     * high-order update.
     *
     * @ingroup ScalarConservationEquations
     */
    template <typename ScalarNumber = double>
    class Indicator : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * A structure holding all runtime parameters of the indicator.
       */
      struct Parameters {
        double evc_factor;
      };

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      Indicator(const HyperbolicSystem &hyperbolic_system,
                const std::string &subsection = "/Indicator")
          : ParameterAcceptor(subsection)
          , parameters_("scalar_conservation_indicator_parameters",
                        TransferPolicy::implicit_transfers_host_resident)
          , hyperbolic_system_(&hyperbolic_system)
      {
        /* reference remains valid due to implicit_transfers_host_resident */
        auto &parameters = *parameters_.view();

        parameters.evc_factor = 1.;
        add_parameter("evc factor",
                      parameters.evc_factor,
                      "Factor for scaling the entropy viscocity commuator");

        /* invalidates view on default memory space */
        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { parameters_.view(); });
      }

      /**
       * Return a view on the Indicator for a given dimension @p dim and
       * choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars). The
       * optional @p MemorySpace template parameter selects whether the
       * view is intended for the host or device memory space.
       */
      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return IndicatorView<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
      }

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      Mirrored<Parameters> parameters_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      template <int, typename, typename>
      friend class IndicatorView;

      //@}
    };


    /**
     * A view of the Indicator that makes the interface available for a
     * given dimension @p dim and choice of number type @p Number (which can
     * be a scalar float, or double, as well as a VectorizedArray holding
     * packed scalars).
     *
     * @ingroup ScalarConservationEquations
     */
    template <int dim, typename Number, typename MemorySpace>
    class IndicatorView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      using ScalarNumber = typename View::ScalarNumber;

      using state_type = typename View::state_type;

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
          , parameters_(indicator.parameters_.template view<MemorySpace>())
      {
      }

      /**
       * Return the factor used for scaling the entropy viscosity
       * commutator.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber evc_factor() const
      {
        return ScalarNumber(parameters_->evc_factor);
      }

      /**
       * Reset temporary storage and initialize for a new row corresponding
       * to state vector U_i.
       */
      DEAL_II_HOST_DEVICE void reset(const PrecomputedVectorView &pv,
                                     const unsigned int i,
                                     const state_type &U_i);

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      DEAL_II_HOST_DEVICE void
      accumulate(const PrecomputedVectorView &pv,
                 const unsigned int *js,
                 const state_type &U_j,
                 const dealii::Tensor<1, dim, Number> &c_ij);

      /**
       * Return the computed alpha_i value.
       */
      DEAL_II_HOST_DEVICE Number alpha(const Number h_i) const;


    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;
      const Indicator<ScalarNumber>::Parameters *const parameters_;

      Number u_i_ = 0.;
      Number u_abs_max_ = 0.;
      dealii::Tensor<1, dim, Number> f_i_;
      Number left_ = 0.;
      Number right_ = 0.;
      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    IndicatorView<dim, Number, MemorySpace>::reset(
        const PrecomputedVectorView &pv,
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    IndicatorView<dim, Number, MemorySpace>::accumulate(
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    IndicatorView<dim, Number, MemorySpace>::alpha(const Number hd_i) const
    {
      Number numerator = left_ - right_;
      Number denominator = std::abs(left_) + std::abs(right_);

      const auto regularization =
          Number(100. * std::numeric_limits<ScalarNumber>::min());

      const auto quotient =
          std::abs(numerator) /
          (denominator + std::max(hd_i * std::abs(u_abs_max_), regularization));

      return std::min(Number(1.), evc_factor() * quotient);
    }

  } // namespace ScalarConservation
} // namespace ryujin
