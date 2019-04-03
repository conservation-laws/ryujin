#ifndef INDICATOR_H
#define INDICATOR_H

#include "problem_description.h"

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
      entropy_viscosity_commutator
    } indicator_ = Indicators::entropy_viscosity_commutator;

    /*
     * The constructor takes a reference to the state vector U_i:
     */

    inline DEAL_II_ALWAYS_INLINE Indicator(const rank1_type& U_i);

    /*
     * When looping over the sparsity row, add the contribution associated
     * with the neighboring state U_j:
     *
     * Note that the second argument (a reference to the corresponding c_ij
     * matrix element) is only used for the entropy viscosity commutator.
     */

    inline DEAL_II_ALWAYS_INLINE void add(const rank1_type &U_j,
                                          const dealii::Tensor<1, dim> &c_ij);

    inline DEAL_II_ALWAYS_INLINE double alpha();

  private:

    /*
     * temporary storage used for the entropy_viscosity_commutator:
     */

    double eta_i = 0.;
    double rho_i = 0.;
    rank2_type f_i;
    rank1_type d_eta_i;

    double left = 0.;
    rank1_type right;

  };


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE Indicator<dim>::Indicator(const rank1_type& U_i)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      rho_i = U_i[0];
      eta_i = ProblemDescription<dim>::entropy(U_i);
      f_i = ProblemDescription<dim>::f(U_i);

      d_eta_i = ProblemDescription<dim>::entropy_derivative(U_i);
      d_eta_i[0] -= eta_i / rho_i;
    }
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Indicator<dim>::add(const rank1_type &U_j, const dealii::Tensor<1, dim> &c_ij)
  {
    if constexpr (indicator_ == Indicators::entropy_viscosity_commutator) {
      const auto &rho_j = U_j[0];
      const auto m_j = ProblemDescription<dim>::momentum(U_j);
      const auto eta_j = ProblemDescription<dim>::entropy(U_j);
      const auto f_j = ProblemDescription<dim>::f(U_j);

      left += (eta_j / rho_j - eta_i / rho_i) * m_j * c_ij;
      for (unsigned int k = 0; k < problem_dimension; ++k)
        right[k] += (f_j[k] - f_i[k]) * c_ij;
    }
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
      return std::abs(numerator) / (denominator + 1.e-12);
    }
  }


} /* namespace grendel */

#endif /* INDICATOR_H */
