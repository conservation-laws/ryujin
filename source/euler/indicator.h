//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
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
  namespace Euler
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
     * This class implements an indicator strategy used to form the
     * preliminary high-order update.
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


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void
    IndicatorView<dim, Number>::reset(const PrecomputedVectorView &pv,
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


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void IndicatorView<dim, Number>::accumulate(
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


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    IndicatorView<dim, Number>::alpha(const Number hd_i) const
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

      return std::min(Number(1.), indicator_.evc_factor() * quotient);
    }
  } // namespace Euler
} // namespace ryujin
