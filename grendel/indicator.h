#ifndef INDICATOR_H
#define INDICATOR_H

#include "problem_description.h"
#include "offline_data.h"

#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim>
  class Indicator
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    using rank2_type = typename ProblemDescription<dim>::rank2_type;

    /*
     * Options:
     */

    static constexpr enum class Indicators {
      smoothness_indicator,
      entropy_viscosity_commutator
    } indicator_ = Indicators::entropy_viscosity_commutator;

    /*
     * Options for smoothness indicator:
     */

    static constexpr enum class SmoothnessIndicators {
      none,
      rho,
      internal_energy,
      pressure,
    } smoothness_indicator_ = SmoothnessIndicators::pressure;

    static constexpr double smoothness_indicator_alpha_0 = 0;

    static constexpr unsigned int smoothness_indicator_power = 3;

    /**
     * We take a reference to an OfflineData object in order to store
     * references to the beta_ij and c_ij matrices.
     */
    Indicator(const OfflineData<dim> &offline_data);

    /**
     * Reset temporary storage and initialize for a new row corresponding
     * to state vector U_i:
     */
    inline DEAL_II_ALWAYS_INLINE void reset(const rank1_type& U_i);

    /**
     * When looping over the sparsity row, add the contribution associated
     * with the neighboring state U_j:
     */
    template <typename ITERATOR>
    inline DEAL_II_ALWAYS_INLINE void add(const rank1_type &U_j,
                                          const ITERATOR jt);

    /**
     * Return the computed alpha_i value.
     */
    inline DEAL_II_ALWAYS_INLINE double alpha();

  private:

    const std::array<dealii::SparseMatrix<double>, dim> &cij_matrix_;
    const dealii::SparseMatrix<double> &betaij_matrix_;

    // FIXME: Unify memory regions:

    /*
     * temporary storage used for the entropy_viscosity_commutator:
     */

    double eta_i = 0.;
    double rho_i = 0.;
    rank2_type f_i;
    rank1_type d_eta_i;

    double left = 0.;
    rank1_type right;

    /*
     * temporary storage used for the smoothness indicator:
     */

    double indicator_i = 0.;

    double numerator = 0.;
    double denominator = 0.;
    double denominator_abs = 0.;

  };


  template <int dim>
  Indicator<dim>::Indicator(const OfflineData<dim> &offline_data)
      : cij_matrix_(offline_data.cij_matrix())
      , betaij_matrix_(offline_data.betaij_matrix())
  {
  }


  namespace
  {
    template<int dim>
    inline DEAL_II_ALWAYS_INLINE double
    smoothness_indicator(const typename Indicator<dim>::rank1_type &U)
    {
      switch (Indicator<dim>::smoothness_indicator_) {
      case Indicator<dim>::SmoothnessIndicators::none:
        return 1.;

      case Indicator<dim>::SmoothnessIndicators::rho:
        return U[0];

      case Indicator<dim>::SmoothnessIndicators::internal_energy:
        return ProblemDescription<dim>::internal_energy(U);

      case Indicator<dim>::SmoothnessIndicators::pressure:
        return ProblemDescription<dim>::pressure(U);
      }
    }
  } // namespace


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void Indicator<dim>::reset(const rank1_type &U_i)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      rho_i = U_i[0];
      eta_i = ProblemDescription<dim>::entropy(U_i);
      f_i = ProblemDescription<dim>::f(U_i);

      d_eta_i = ProblemDescription<dim>::entropy_derivative(U_i);
      d_eta_i[0] -= eta_i / rho_i;

      left = 0.;
      right *= 0.;
    }

    if constexpr (indicator_ == Indicators::smoothness_indicator) {
      numerator = 0.;
      denominator = 0.;
      denominator_abs = 0.;

      indicator_i = smoothness_indicator(U_i);
    }

    __builtin_unreachable();
  }


  template <int dim>
  template <typename ITERATOR>
  inline DEAL_II_ALWAYS_INLINE void Indicator<dim>::add(const rank1_type &U_j,
                                                        const ITERATOR jt)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      const auto c_ij = gather_get_entry(cij_matrix_, jt);

      const auto &rho_j = U_j[0];
      const auto m_j = ProblemDescription<dim>::momentum(U_j);
      const auto eta_j = ProblemDescription<dim>::entropy(U_j);
      const auto f_j = ProblemDescription<dim>::f(U_j);

      left += (eta_j / rho_j - eta_i / rho_i) * m_j * c_ij;
      for (unsigned int k = 0; k < problem_dimension; ++k)
        right[k] += (f_j[k] - f_i[k]) * c_ij;
    }

    if constexpr (indicator_ == Indicators::smoothness_indicator) {
      const auto beta_ij = get_entry(betaij_matrix_, jt);

      const auto indicator_j = smoothness_indicator(U_j);

      numerator += beta_ij * (indicator_i - indicator_j);
      denominator +=
          std::abs(beta_ij) * std::abs(indicator_i - indicator_j);
      denominator_abs += std::abs(beta_ij) * (std::abs(indicator_i) +
                                                std::abs(indicator_j));
    }

    __builtin_unreachable();
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double Indicator<dim>::alpha()
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      double numerator = left;
      double denominator = std::abs(left);
      for (unsigned int k = 0; k < problem_dimension; ++k) {
        numerator -= d_eta_i[k] * right[k];
        denominator += std::abs(d_eta_i[k] * right[k]);
      }
      return std::abs(numerator) / (denominator + 1.e-12); // FIXME
    }

    if constexpr (indicator_ == Indicators::smoothness_indicator) {
      const auto beta_i = std::abs(numerator) / denominator_abs;

      if(denominator > 1.e-7 * denominator_abs)
      {
        const auto ratio = std::abs(numerator) / denominator;
        const auto alpha_i =
            std::pow(std::max(ratio - smoothness_indicator_alpha_0, 0.),
                     smoothness_indicator_power) /
            std::pow(1 - smoothness_indicator_alpha_0,
                     smoothness_indicator_power);
        return std::min(alpha_i, beta_i);
      }

      return beta_i;
    }

    __builtin_unreachable();
  }


} /* namespace grendel */

#endif /* INDICATOR_H */
