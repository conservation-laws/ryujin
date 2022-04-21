//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "offline_data.h"
#include "problem_description.h"
#include "simd.h"

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
   * entropy, see the documentation of ProblemDescription) we let
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
   * @ingroup EulerModule
   */
  template <int dim, typename Number = double>
  class Indicator
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc ProblemDescription::state_type
     */
    using state_type = ProblemDescription::state_type<dim, Number>;

    /**
     * @copydoc ProblemDescription::flux_type
     */
    using flux_type = ProblemDescription::flux_type<dim, Number>;

    /**
     * @copydoc ProblemDescription::ScalarNumber
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
     *   indicator.reset(U_i, evc_entropy_i, evc_entropy_derivative_i);
     *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
     *     // ...
     *     indicator.add(U_j, c_ij, evc_entropy_j);
     *   }
     *   indicator.alpha(hd_i);
     * }
     * ```
     */
    //@{

    /**
     * Constructor taking a ProblemDescription instance as argument
     */
    Indicator(const ProblemDescription &problem_description)
        : problem_description(problem_description)
    {
    }

    /**
     * Reset temporary storage and initialize for a new row corresponding
     * to state vector U_i.
     */
    void reset(
        const state_type &U_i,
        const Number entropy,
        const dealii::Tensor<1, problem_dimension, Number> &entropy_derivative);

    /**
     * When looping over the sparsity row, add the contribution associated
     * with the neighboring state U_j.
     */
    void add(const state_type &U_j,
             const dealii::Tensor<1, dim, Number> &c_ij,
             const Number entropy_j);
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

    const ProblemDescription &problem_description;

    Number rho_i_inverse = 0.;
    Number eta_i = 0.;
    flux_type f_i;
    state_type d_eta_i;

    Number left = 0.;
    state_type right;

    //@}
  };


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Indicator<dim, Number>::reset(
      const state_type &U_i,
      const Number entropy,
      const dealii::Tensor<1, problem_dimension, Number> &entropy_derivative)
  {
    /* entropy viscosity commutator: */

    const auto &rho_i = problem_description.density(U_i);
    rho_i_inverse = Number(1.) / rho_i;
    eta_i = entropy;
    d_eta_i = entropy_derivative;
    d_eta_i[0] -= eta_i * rho_i_inverse;
    f_i = problem_description.f(U_i);

    left = 0.;
    right = 0.;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Indicator<dim, Number>::add(const state_type &U_j,
                              const dealii::Tensor<1, dim, Number> &c_ij,
                              const Number entropy_j)
  {
    /* entropy viscosity commutator: */

    const auto &rho_j = problem_description.density(U_j);
    const auto rho_j_inverse = Number(1.) / rho_j;
    const auto m_j = problem_description.momentum(U_j);
    const auto f_j = problem_description.f(U_j);
    const auto eta_j = entropy_j;

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

} /* namespace ryujin */
