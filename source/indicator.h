//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef INDICATOR_H
#define INDICATOR_H

#include <compile_time_options.h>

#include "offline_data.h"
#include "problem_description.h"
#include "simd.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.templates.h>


namespace ryujin
{

  /**
   * This class implements two indicator strategies used to form the
   * preliminary high-order update.
   *
   * The first one is a smoothness indicator as described in
   * @cite GuermondEtAl2018 and that is similar in nature to smoothness
   * indicators used in the finite volume communities (see for example
   * @cite Jameson2017).
   *
   * @todo explain algorithmic details of the smoothness indicator
   *
   * The second indicator is an entropy-viscosity commutator as described
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
   * In addition, the class also computes second variations of the density
   * \f$\rho\f$ that is used for relaxing the limiter bounds, see
   * documentation of class Limiter.
   *
   * @ingroup EulerStep
   */
  template <int dim, typename Number = double>
  class Indicator
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription<dim, Number>::problem_dimension;
    // clang-format on

    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    /**
     * @copydoc ProblemDescription::rank2_type
     */
    using rank2_type = typename ProblemDescription<dim, Number>::rank2_type;

    /**
     * @copydoc ProblemDescription::ScalarNumber
     */
    using ScalarNumber = typename get_value_type<Number>::type;

    /**
     * An enum describing different indicator strategies
     */
    enum class Indicators {
      /**
       * Indicator returns a constant zero, i.e., the high-order update is
       * equal to an inviscid Galerkin update.
       */
      zero,
      /**
       * Indicator returns a constant one, i.e., the high-order update is
       * equal to the low-order update.
       */
      one,
      /** Use a smoothness indicator. */
      smoothness_indicator,
      /** Use an entropy-viscosity commutator. */
      entropy_viscosity_commutator
    };

    /**
     * An enum describing different choices of entropies used for the
     * entropy-viscosity commutator
     */
    enum class Entropy {
      /** The (scaled) mathematical entropy */
      mathematical,
      /** A generalized Harten-type entropy */
      harten
    };

    /**
     * An scalar physical quantity used for computing the smoothness
     * indicator.
     */
    enum class SmoothnessIndicators {
      /** The density */
      rho,
      /** The internal energy */
      internal_energy,
      /** The pressure */
      pressure,
    };

    /**
     * @name Indicator compile time options
     */
    //@{

    // clang-format off
    /**
     * Selected indicator used for the preliminary high-order update.
     * @ingroup CompileTimeOptions
     */
    static constexpr Indicators indicator_ = INDICATOR;

    /**
     * Compute second variations of the density.
     * @ingroup CompileTimeOptions
     */
    static constexpr bool compute_second_variations_ = COMPUTE_SECOND_VARIATIONS;

    /**
     * Selected entropy used for the entropy-viscosity commutator.
     * @ingroup CompileTimeOptions
     */
    static constexpr Entropy evc_entropy_ = ENTROPY;

    /**
     * Selected quantity used for the smoothness indicator.
     * @ingroup CompileTimeOptions
     */
    static constexpr SmoothnessIndicators smoothness_indicator_ = SMOOTHNESS_INDICATOR;

    /**
     * Tuning parameter for the smoothness indicator.
     * @ingroup CompileTimeOptions
     */
    static constexpr ScalarNumber smoothness_indicator_alpha_0_ = SMOOTHNESS_INDICATOR_ALPHA_0;

    /**
     * Tuning parameter for the smoothness indicator.cator.
     * @ingroup CompileTimeOptions
     */
    static constexpr unsigned int smoothness_indicator_power_ = SMOOTHNESS_INDICATOR_POWER;
    // clang-format on

    //@}
    /**
     * @name Stencil-based computation of indicators
     *
     * Intended usage:
     * ```
     * Indicator<dim, Number> indicator;
     * for (unsigned int i = n_internal; i < n_owned; ++i) {
     *   // ...
     *   indicator.reset(U_i, evc_entropies_i);
     *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
     *     // ...
     *     indicator.add(U_j, c_ij, beta_ij, evc_entropies_j);
     *   }
     *   indicator.alpha(hd_i);
     *   indicator.second_variations();
     * }
     * ```
     */
    //@{

    /**
     * Reset temporary storage and initialize for a new row corresponding
     * to state vector U_i.
     */
    void
    reset(const rank1_type &U_i, const Number entropy);

    /**
     * When looping over the sparsity row, add the contribution associated
     * with the neighboring state U_j.
     */
    void add(const rank1_type &U_j,
             const dealii::Tensor<1, dim, Number> &c_ij,
             const Number beta_ij,
             const Number entropy_j);
    /**
     * Return the computed alpha_i value.
     */
    Number alpha(const Number h_i);

    /**
     * Return the computed second variation of rho.
     */
    Number second_variations();

    //@}

  private:
    /**
     * @name
     */
    //@{

    Number rho_i = 0.;         // also used for second variations
    Number rho_i_inverse = 0.; // also used for second variations
    Number eta_i = 0.;
    rank2_type f_i;
    rank1_type d_eta_i;

    Number left = 0.;
    rank1_type right;

    /* Temporary storage used for the smoothness indicator: */

    Number indicator_i = 0.;

    Number numerator = 0.;
    Number denominator = 0.;
    Number denominator_abs = 0.;

    /* Temporary storage used to compute second variations: */

    Number rho_second_variation_numerator = 0.;
    Number rho_second_variation_denominator = 0.;

    //@}
  };


  namespace
  {
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    smoothness_indicator(const typename Indicator<dim, Number>::rank1_type &U)
    {
      switch (Indicator<dim, Number>::smoothness_indicator_) {
      case Indicator<dim, Number>::SmoothnessIndicators::rho:
        return U[0];

      case Indicator<dim, Number>::SmoothnessIndicators::internal_energy:
        return ProblemDescription<dim, Number>::internal_energy(U);

      case Indicator<dim, Number>::SmoothnessIndicators::pressure:
        return ProblemDescription<dim, Number>::pressure(U);
      }
    }
  } // namespace


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Indicator<dim, Number>::reset(
      const rank1_type &U_i, const Number entropy)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      rho_i = U_i[0];
      rho_i_inverse = Number(1.) / rho_i;
      eta_i = entropy;
      f_i = ProblemDescription<dim, Number>::f(U_i);

      d_eta_i =
          evc_entropy_ == Entropy::mathematical
              ? ProblemDescription<dim,
                                   Number>::mathematical_entropy_derivative(U_i)
              : ProblemDescription<dim, Number>::harten_entropy_derivative(U_i);
      d_eta_i[0] -= eta_i * rho_i_inverse;

      left = 0.;
      right = 0.;
    }

    if constexpr (indicator_ == Indicators::smoothness_indicator) {
      numerator = 0.;
      denominator = 0.;
      denominator_abs = 0.;

      indicator_i = smoothness_indicator<dim, Number>(U_i);
    }

    if constexpr (compute_second_variations_) {
      rho_i = U_i[0];
      rho_second_variation_numerator = 0.;
      rho_second_variation_denominator = 0.;
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Indicator<dim, Number>::add(const rank1_type &U_j,
                              const dealii::Tensor<1, dim, Number> &c_ij,
                              const Number beta_ij,
                              const Number entropy_j)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      const auto &rho_j = U_j[0];
      const auto rho_j_inverse = Number(1.) / rho_j;
      const auto m_j = ProblemDescription<dim, Number>::momentum(U_j);
      const auto f_j = ProblemDescription<dim, Number>::f(U_j);
      const auto eta_j = entropy_j;

      left += (eta_j * rho_j_inverse - eta_i * rho_i_inverse) * (m_j * c_ij);
      for (unsigned int k = 0; k < problem_dimension; ++k)
        right[k] += (f_j[k] - f_i[k]) * c_ij;
    }

    if constexpr (indicator_ == Indicators::smoothness_indicator) {
      const auto indicator_j = smoothness_indicator<dim, Number>(U_j);

      numerator += beta_ij * (indicator_i - indicator_j);
      denominator += std::abs(beta_ij) * std::abs(indicator_i - indicator_j);
      denominator_abs +=
          std::abs(beta_ij) * (std::abs(indicator_i) + std::abs(indicator_j));
    }

    if constexpr (compute_second_variations_) {
      const auto &rho_j = U_j[0];

      rho_second_variation_numerator += beta_ij * (rho_j - rho_i);
      rho_second_variation_denominator += beta_ij;
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  Indicator<dim, Number>::alpha(const Number hd_i)
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    if constexpr (indicator_ == Indicators::zero) {
      return Number(0.);
    }

    if constexpr (indicator_ == Indicators::one) {
      return Number(1.);
    }

    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      Number numerator = left;
      Number denominator = std::abs(left);
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        numerator -= d_eta_i[k] * right[k];
        denominator += std::abs(d_eta_i[k] * right[k]);
      }

      const auto quotient =
          std::abs(numerator) / (denominator + hd_i * std::abs(eta_i));
      return quotient;
    }

    if constexpr (indicator_ == Indicators::smoothness_indicator) {
      const Number beta_i = std::abs(numerator) / denominator_abs;

      const Number ratio = std::abs(numerator) / denominator;
      const Number alpha_i =
          dealii::Utilities::fixed_power<smoothness_indicator_power_>(
              std::max(ratio - smoothness_indicator_alpha_0_, Number(0.))) /
          dealii::Utilities::fixed_power<smoothness_indicator_power_>(
              1 - smoothness_indicator_alpha_0_);

      return dealii::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than>(denominator,
                                                ScalarNumber(1.0e-7) *
                                                    denominator_abs,
                                                std::min(alpha_i, beta_i),
                                                beta_i);
    }

    __builtin_unreachable();
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  Indicator<dim, Number>::second_variations()
  {
    constexpr Number eps = std::numeric_limits<Number>::epsilon();
    return rho_second_variation_numerator /
           (rho_second_variation_denominator + Number(eps));
  }


} /* namespace ryujin */

#endif /* INDICATOR_H */
