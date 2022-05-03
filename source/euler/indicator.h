//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <simd.h>

#include <deal.II/base/vectorization.h>


namespace ryujin
{
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
     * @copydoc HyperbolicSystem::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = HyperbolicSystem::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = HyperbolicSystem::state_type<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::flux_type
     */
    using flux_type = HyperbolicSystem::flux_type<dim, Number>;

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
     *     indicator.add(js, U_j, c_ij);
     *   }
     *   indicator.alpha(hd_i);
     * }
     * ```
     */
    //@{

    /**
     * The number of precomputed values.
     */
    static constexpr unsigned int n_precomputed_values = 1;

    /**
     * Array type used for precomputed values.
     */
    using PrecomputedValues = std::array<Number, n_precomputed_values>;

    /**
     * Precomputed values for a given state.
     */
    static PrecomputedValues
    precompute_values(const HyperbolicSystem &hyperbolic_system,
                      const state_type &U);

    /**
     * Constructor taking a HyperbolicSystem instance as argument
     */
    Indicator(const HyperbolicSystem &hyperbolic_system,
              const MultiComponentVector<ScalarNumber, n_precomputed_values>
                  &precomputed_values)
        : hyperbolic_system(hyperbolic_system)
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
    void add(const unsigned int *js,
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

    const HyperbolicSystem &hyperbolic_system;

    const MultiComponentVector<ScalarNumber, n_precomputed_values>
        &precomputed_values;

    Number rho_i_inverse = 0.;
    Number eta_i = 0.;
    flux_type f_i;
    state_type d_eta_i;

    Number left = 0.;
    state_type right;

    //@}
  };


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline
      typename Indicator<dim, Number>::PrecomputedValues
      Indicator<dim, Number>::precompute_values(
          const HyperbolicSystem &hyperbolic_system, const state_type &U_i)
  {
    PrecomputedValues result;
    result[0] = hyperbolic_system.harten_entropy(U_i);
    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Indicator<dim, Number>::reset(const unsigned int i, const state_type &U_i)
  {
    /* entropy viscosity commutator: */

    const auto &[harten_entropy] =
        precomputed_values.template get_tensor<Number, PrecomputedValues>(i);

    const auto rho_i = hyperbolic_system.density(U_i);
    rho_i_inverse = Number(1.) / rho_i;
    eta_i = harten_entropy;

    d_eta_i = hyperbolic_system.harten_entropy_derivative(U_i);
    d_eta_i[0] -= eta_i * rho_i_inverse;
    f_i = hyperbolic_system.f(U_i);

    left = 0.;
    right = 0.;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Indicator<dim, Number>::add(const unsigned int *js,
                              const state_type &U_j,
                              const dealii::Tensor<1, dim, Number> &c_ij)
  {
    /* entropy viscosity commutator: */

    const auto &[eta_j] =
        precomputed_values.template get_tensor<Number, PrecomputedValues>(js);

    const auto rho_j = hyperbolic_system.density(U_j);
    const auto rho_j_inverse = Number(1.) / rho_j;

    const auto m_j = hyperbolic_system.momentum(U_j);
    const auto f_j = hyperbolic_system.f(U_j);

    left += (eta_j * rho_j_inverse - eta_i * rho_i_inverse) * (m_j * c_ij);
    for (unsigned int k = 0; k < problem_dimension; ++k)
      right[k] += (f_j[k] - f_i[k]) * c_ij;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  Indicator<dim, Number>::alpha(const Number hd_i)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    Number numerator = left;
    Number denominator = std::abs(left);
    for (unsigned int k = 0; k < problem_dimension; ++k) {
      numerator -= d_eta_i[k] * right[k];
      denominator += std::abs(d_eta_i[k] * right[k]);
    }

    /* FIXME: this can be refactoring into a runtime parameter... */
    const ScalarNumber evc_alpha_0_ = ScalarNumber(1.);
    const auto quotient =
        std::abs(numerator) / (denominator + hd_i * std::abs(eta_i));
    return std::min(Number(1.), evc_alpha_0_ * quotient);
  }

} // namespace ryujin
