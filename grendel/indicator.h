#ifndef INDICATOR_H
#define INDICATOR_H

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim, typename Number = double>
  class Indicator
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    using rank2_type = typename ProblemDescription<dim, Number>::rank2_type;

    /**
     * Scalar number type
     */
    using ScalarNumber = typename get_value_type<Number>::type;

    /*
     * Options:
     */

    static constexpr enum class Indicators {
      zero,
      one,
      smoothness_indicator,
      entropy_viscosity_commutator
    } indicator_ = Indicators::entropy_viscosity_commutator;

    static constexpr enum class SmoothnessIndicators {
      rho,
      internal_energy,
      pressure,
    } smoothness_indicator_ = SmoothnessIndicators::pressure;

    static constexpr ScalarNumber smoothness_indicator_alpha_0_ = 0.;

    static constexpr unsigned int smoothness_indicator_power_ = 3;

    static constexpr bool compute_second_variations_ = true;

    /**
     * We take a reference to an OfflineData object in order to store
     * references to the beta_ij and c_ij matrices.
     */
    Indicator(const OfflineData<dim, Number> &offline_data);

    /**
     * Reset temporary storage and initialize for a new row corresponding
     * to state vector U_i:
     */
    DEAL_II_ALWAYS_INLINE inline void reset(const rank1_type U_i);

    /**
     * When looping over the sparsity row, add the contribution associated
     * with the neighboring state U_j:
     */
    template <typename ITERATOR>
    DEAL_II_ALWAYS_INLINE inline void add(const rank1_type U_j,
                                          const ITERATOR jt);
    /**
     * Return the computed alpha_i value.
     */
    DEAL_II_ALWAYS_INLINE inline Number alpha(const Number h_i);

    /**
     * Return the computed second variation of rho.
     */
    DEAL_II_ALWAYS_INLINE inline Number rho_second_variation();

  private:
    const std::array<dealii::SparseMatrix<Number>, dim> &cij_matrix_;
    const dealii::SparseMatrix<Number> &betaij_matrix_;

    /* Temporary storage used for the entropy_viscosity_commutator: */

    Number rho_i = 0.; // also used for second variations
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
  };


  template <int dim, typename Number>
  Indicator<dim, Number>::Indicator(const OfflineData<dim, Number> &offline_data)
      : cij_matrix_(offline_data.cij_matrix())
      , betaij_matrix_(offline_data.betaij_matrix())
  {
  }


  namespace
  {
    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    smoothness_indicator(const typename Indicator<dim, Number>::rank1_type U)
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
  DEAL_II_ALWAYS_INLINE inline void
  Indicator<dim, Number>::reset(const rank1_type U_i)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      rho_i = U_i[0];
      eta_i = ProblemDescription<dim, Number>::entropy(U_i);
      f_i = ProblemDescription<dim, Number>::f(U_i);

      d_eta_i = ProblemDescription<dim, Number>::entropy_derivative(U_i);
      d_eta_i[0] -= eta_i / rho_i;

      left = 0.;
      right *= 0.;
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
  template <typename ITERATOR>
  DEAL_II_ALWAYS_INLINE inline void
  Indicator<dim, Number>::add(const rank1_type U_j, const ITERATOR jt)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      const auto c_ij = gather_get_entry(cij_matrix_, jt);

      const auto &rho_j = U_j[0];
      const auto m_j = ProblemDescription<dim, Number>::momentum(U_j);
      const auto eta_j = ProblemDescription<dim, Number>::entropy(U_j);
      const auto f_j = ProblemDescription<dim, Number>::f(U_j);

      left += (eta_j / rho_j - eta_i / rho_i) * m_j * c_ij;
      for (unsigned int k = 0; k < problem_dimension; ++k)
        right[k] += (f_j[k] - f_i[k]) * c_ij;
    }

    const auto beta_ij = get_entry(betaij_matrix_, jt);

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
    if constexpr (indicator_ == Indicators::zero) {
      return 0.;
    }

    if constexpr (indicator_ == Indicators::one) {
      return 1.;
    }

    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      Number numerator = left;
      Number denominator = std::abs(left);
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        numerator -= d_eta_i[k] * right[k];
        denominator += std::abs(d_eta_i[k] * right[k]);
      }
      return std::abs(numerator) / (denominator + hd_i * std::abs(eta_i));
    }

    if constexpr (indicator_ == Indicators::smoothness_indicator) {
      const auto beta_i = std::abs(numerator) / denominator_abs;

      if (denominator > 1.e-7 * denominator_abs) {
        const auto ratio = std::abs(numerator) / denominator;
        const auto alpha_i =
            std::pow(std::max(ratio - smoothness_indicator_alpha_0_, 0.),
                     smoothness_indicator_power_) /
            std::pow(1 - smoothness_indicator_alpha_0_,
                     smoothness_indicator_power_);
        return std::min(alpha_i, beta_i);
      }

      return beta_i;
    }

    __builtin_unreachable();
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  Indicator<dim, Number>::rho_second_variation()
  {
    constexpr Number eps = std::numeric_limits<Number>::epsilon();
    return rho_second_variation_numerator /
           (rho_second_variation_denominator + eps);
  }


} /* namespace grendel */

#endif /* INDICATOR_H */
