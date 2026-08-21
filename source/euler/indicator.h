//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <mirrored.h>
#include <multicomponent_vector.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>

namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class IndicatorView;

    /**
     * An indicator strategy used to form the preliminary high-order
     * update.
     *
     * The indicator is an entropy-viscosity commutator as described
     * in @cite GuermondEtAl2011 and @cite GuermondEtAl2018. For a given
     * entropy \f$\eta\f$ (either the mathematical entropy, or a Harten
     * entropy, see the documentation of HyperbolicSystem) we let
     * \f$\eta'\f$ denote its derivative with respect to the state variables.
     * We then compute a normalized entropy viscosity ratio \f$\alpha_i^n\f$
     * for the state \f$\boldsymbol U_i^n\f$ as follows:
     * \f{align}
     *   \alpha_i^n\;=\;\frac{N_i^n}{D_i^n},
     *   \quad
     *   N_i^n\;:=\;\left|a_i^n- \eta'(\boldsymbol U^n_i)\cdot\boldsymbol
     *   b_i^n +\frac{\eta(\boldsymbol U^n_i)}{\rho_i^n}\big(\boldsymbol
     *   b_i^n\big)_1\right|,
     *   \quad
     *   D_i^n\;:=\;\left|a_i^n\right| +
     *   \sum_{k=1}^{d+1}\left|\big(\eta'(\boldsymbol U^n_i)\big)_k-
     *   \delta_{1k}\frac{\eta(\boldsymbol U^n_i)}{\rho_i^n}\right|
     *   \,\left|\big(\boldsymbol b_i^n\big)_k\right|,
     * \f}
     * where where \f$\big(\,.\,\big)_k\f$ denotes the \f$k\f$-th component
     * of a vector, \f$\delta_{ij}\f$ is Kronecker's delta, and where we have
     * set
     * \f{align}
     *   a_i^n \;:=\;
     *   \sum_{j\in\mathcal{I}_i}\left(\frac{\eta(\boldsymbol U_j^n)}{\rho_j^n}
     *   -\frac{\eta(\boldsymbol U_i^n)}{\rho_i^n}\right)\,
     *   \boldsymbol m_j^n\cdot\boldsymbol c_{ij},
     *   \qquad
     *   \boldsymbol b_i^n \;:=\;
     *   \sum_{j\in\mathcal{I}_i}\left(\mathbf{f}(\boldsymbol U_j^n)-
     *   \mathbf{f}(\boldsymbol U_i^n)\right)\cdot\boldsymbol c_{ij},
     * \f}
     *
     * @ingroup EulerEquations
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

      /**
       * Alias for the view on the indicator for a given dimension @p dim,
       * choice of number type @p Number, and memory space @p MemorySpace.
       */
      template <int dim,
                typename Number = double,
                typename MemorySpace = dealii::MemorySpace::Host>
      using View = IndicatorView<dim, Number, MemorySpace>;

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
          , parameters_("euler_indicator_parameters")
          , hyperbolic_system_(&hyperbolic_system)
      {
        /*
         * Note: We bind the parameters directly to the storage held by the
         * Mirrored object. The corresponding memory is allocated once in
         * the constructor and never reallocated, so the addresses remain
         * valid for the lifetime of this object.
         */
        auto &parameters = parameters_.value();

        parameters.evc_factor = 1.;
        add_parameter("evc factor",
                      parameters.evc_factor,
                      "Factor for scaling the entropy viscocity commuator");

        /* Copy the parameters over to the default memory space: */
        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { parameters_.update(); });
        parameters_.update();
      }

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      ScalarNumber evc_factor() const
      {
        return ScalarNumber(parameters_.value().evc_factor);
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
        return View<dim, Number, MemorySpace>{
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

      //@}

      template <int, typename, typename>
      friend class IndicatorView;
    };


    /**
     * A view of the Indicator that makes the interface available for a
     * given dimension @p dim and choice of number type @p Number (which can
     * be a scalar float, or double, as well as a VectorizedArray holding
     * packed scalars).
     *
     * @ingroup EulerEquations
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
          , parameters_(indicator.parameters_.template get_view<MemorySpace>())
      {
      }

      /**
       * Return the factor used for scaling the entropy viscosity
       * commutator.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber evc_factor() const
      {
        return ScalarNumber(parameters_().evc_factor);
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

      using ParameterView =
          typename Mirrored<typename Indicator<ScalarNumber>::Parameters>::
              template View<MemorySpace>;

      const View view_;
      ParameterView parameters_;

      Number rho_i_inverse_ = 0.;
      Number eta_i_ = 0.;
      flux_type f_i_;
      state_type d_eta_i_;

      Number left_ = 0.;
      state_type right_;

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
      /* Entropy viscosity commutator: */

      const auto &[s_i, eta_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      const auto rho_i = view_.density(U_i);
      rho_i_inverse_ = Number(1.) / rho_i;
      eta_i_ = eta_i;

      d_eta_i_ = view_.harten_entropy_derivative(U_i);
      d_eta_i_[0] -= eta_i_ * rho_i_inverse_;
      f_i_ = view_.f(U_i);

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
      /* Entropy viscosity commutator: */

      const auto &[s_j, eta_j] =
          pv.template read_tensor<Number, precomputed_type>(js);

      const auto rho_j = view_.density(U_j);
      const auto rho_j_inverse = Number(1.) / rho_j;

      const auto m_j = view_.momentum(U_j);
      const auto f_j = view_.f(U_j);

      const auto entropy_flux =
          (eta_j * rho_j_inverse - eta_i_ * rho_i_inverse_) * (m_j * c_ij);

      left_ += entropy_flux;
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        const auto component = (f_j[k] - f_i_[k]) * c_ij;
        right_[k] += component;
      }
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    IndicatorView<dim, Number, MemorySpace>::alpha(const Number hd_i) const
    {
      /* Entropy viscosity commutator: */

      Number numerator = left_;
      Number denominator = std::abs(left_);
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        numerator -= d_eta_i_[k] * right_[k];
        denominator += std::abs(d_eta_i_[k] * right_[k]);
      }

      const auto quotient =
          std::abs(numerator) / (denominator + hd_i * std::abs(eta_i_));

      return std::min(Number(1.), evc_factor() * quotient);
    }
  } // namespace Euler
} // namespace ryujin
