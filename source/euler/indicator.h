//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/vectorization.h>


namespace ryujin
{
  namespace Euler
  {
    enum class IndicatorStrategy {
      /** Indicator returns constant 0 */
      galerkin,
      /** The classical entropy viscosity commutator */
      evc,
      /** Entropy viscosity with more aggressive denominator split */
      evc_fullsplit,
    };
  }
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Euler::IndicatorStrategy,
             LIST({ryujin::Euler::IndicatorStrategy::galerkin, "galerkin"},
                  {ryujin::Euler::IndicatorStrategy::evc, "entropy viscosity"},
                  {ryujin::Euler::IndicatorStrategy::evc_fullsplit,
                   "entropy viscosity full split"}));
#endif

namespace ryujin
{
  namespace Euler
  {
    template <typename ScalarNumber = double>
    class IndicatorParameters : public dealii::ParameterAcceptor
    {
    public:
      IndicatorParameters(const std::string &subsection = "/Indicator")
          : ParameterAcceptor(subsection)
      {
        indicator_strategy_ = IndicatorStrategy::evc;
        add_parameter(
            "indicator strategy",
            indicator_strategy_,
            "The chosen indicator strategy. Possible values are: galerkin, "
            "entropy viscosity, entropy viscosity full split");

        evc_factor_ = ScalarNumber(1.);
        add_parameter("evc factor",
                      evc_factor_,
                      "Factor for scaling the entropy viscocity commuator");
      }

      ACCESSOR_READ_ONLY(indicator_strategy);

      ACCESSOR_READ_ONLY(evc_factor);

    private:
      IndicatorStrategy indicator_strategy_;
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
    template <int dim, typename Number = double>
    class Indicator
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

      using PrecomputedVector = typename View::PrecomputedVector;

      using Parameters = IndicatorParameters<ScalarNumber>;

      //@}
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
                const PrecomputedVector &precomputed_values)
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
      const PrecomputedVector &precomputed_values;

      Number rho_i_inverse = 0.;
      Number eta_i = 0.;
      flux_type f_i;
      state_type d_eta_i;

      Number left = 0.;
      state_type right;

      Number left_absolute = 0.;
      Number right_value = 0.;
      Number right_absolute = 0.;
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
      if (parameters.indicator_strategy() == IndicatorStrategy::galerkin)
        return;

      /* entropy viscosity commutator: */

      const auto view = hyperbolic_system.view<dim, Number>();

      const auto &[new_s_i, new_eta_i] =
          precomputed_values.template get_tensor<Number, precomputed_type>(i);

      const auto rho_i = view.density(U_i);
      rho_i_inverse = Number(1.) / rho_i;
      eta_i = new_eta_i;

      d_eta_i = view.harten_entropy_derivative(U_i);
      d_eta_i[0] -= eta_i * rho_i_inverse;
      f_i = view.f(U_i);

      left = 0.;
      right = 0.;

      left_absolute = 0.;
      right_value = 0.;
      right_absolute = 0.;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline void Indicator<dim, Number>::accumulate(
        const unsigned int *js,
        const state_type &U_j,
        const dealii::Tensor<1, dim, Number> &c_ij)
    {
      if (parameters.indicator_strategy() == IndicatorStrategy::galerkin)
        return;

      /* entropy viscosity commutator: */

      const auto view = hyperbolic_system.view<dim, Number>();

      const auto &[s_j, eta_j] =
          precomputed_values.template get_tensor<Number, precomputed_type>(js);

      const auto rho_j = view.density(U_j);
      const auto rho_j_inverse = Number(1.) / rho_j;

      const auto m_j = view.momentum(U_j);
      const auto f_j = view.f(U_j);

      const auto entropy_flux =
          (eta_j * rho_j_inverse - eta_i * rho_i_inverse) * (m_j * c_ij);

      if (parameters.indicator_strategy() == IndicatorStrategy::evc_fullsplit) {
        /* Entropy viscosity commutator with aggressive denominator split: */

        left += entropy_flux;
        left_absolute += std::abs(entropy_flux);
        for (unsigned int k = 0; k < problem_dimension; ++k) {
          const auto component = d_eta_i[k] * (f_j[k] - f_i[k]) * c_ij;
          right_value += component;
          right_absolute += std::abs(component);
        }

      } else {
        /* Entropy viscosity commutator with conservative denominator split: */

        left += entropy_flux;
        for (unsigned int k = 0; k < problem_dimension; ++k) {
          const auto component = (f_j[k] - f_i[k]) * c_ij;
          right[k] += component;
        }
      }
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    Indicator<dim, Number>::alpha(const Number hd_i) const
    {
      if (parameters.indicator_strategy() == IndicatorStrategy::galerkin)
        return Number(0.);

      if (parameters.indicator_strategy() == IndicatorStrategy::evc_fullsplit) {
        /* Entropy viscosity commutator with aggressive denominator split: */

        Number numerator = left - right_value;
        Number denominator = left_absolute + right_absolute;

        const auto quotient =
            std::abs(numerator) / (denominator + hd_i * std::abs(eta_i));

        return std::min(Number(1.), parameters.evc_factor() * quotient);

      } else {
        /* Entropy viscosity commutator with conservative denominator split: */

        Number numerator = left;
        Number denominator = std::abs(left);
        for (unsigned int k = 0; k < problem_dimension; ++k) {
          numerator -= d_eta_i[k] * right[k];
          denominator += std::abs(d_eta_i[k] * right[k]);
        }

        const auto quotient =
            std::abs(numerator) / (denominator + hd_i * std::abs(eta_i));

        return std::min(Number(1.), parameters.evc_factor() * quotient);
      }
    }
  } // namespace Euler
} // namespace ryujin
