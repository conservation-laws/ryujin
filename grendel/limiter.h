#ifndef LIMITER_H
#define LIMITER_H

#include "helper.h"
#include "newton.h"
#include "offline_data.h"
#include "simd.h"

#include "problem_description.h"

#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim, typename Number = double>
  class Limiter
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using ScalarNumber = typename get_value_type<Number>::type;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    using vector_type =
        std::array<dealii::LinearAlgebra::distributed::Vector<Number>, 3>;

    using Bounds = std::array<Number, 3>;

    /**
     * We take a reference to an OfflineData object in order to store
     * references to the beta_ij matrix.
     */
    Limiter(const OfflineData<dim, ScalarNumber> &offline_data);

    /*
     * Options:
     */

    static constexpr enum class Limiters {
      none,
      rho,
      specific_entropy,
      entropy_inequality
    } limiter_ = Limiters::specific_entropy;

    static constexpr bool relax_bounds_ = true;

    static constexpr unsigned int relaxation_order_ = 3;

    /*
     * Accumulate bounds:
     */

    void reset();

    void accumulate(const rank1_type U_i,
                    const rank1_type U_j,
                    const rank1_type U_ij_bar,
                    const bool is_diagonal_entry);

    void reset_variations(const Number variations_i);

    template <typename ITERATOR>
    void accumulate_variations(const Number variations_j, const ITERATOR jt);

    void apply_relaxation(const Number hd_i);

    const Bounds &bounds() const;

    /*
     * Given a state U and an update P this function computes and returns
     * the maximal t, obeying t_min < t < t_max, such that the selected
     * local minimum principles are obeyed.
     */
    template <Limiters limiter = limiter_, typename BOUNDS>
    static Number limit(const BOUNDS bounds,
                        const rank1_type U,
                        const rank1_type P,
                        const Number t_min = Number(0.),
                        const Number t_max = Number(1.));

  private:
    const dealii::SparseMatrix<ScalarNumber> &betaij_matrix_;

    Bounds bounds_;

    Number variations_i;
    Number rho_relaxation_numerator;
    Number rho_relaxation_denominator;

    Number s_interp_max;
  };


  template <int dim, typename Number>
  Limiter<dim, Number>::Limiter(
      const OfflineData<dim, ScalarNumber> &offline_data)
      : betaij_matrix_(offline_data.betaij_matrix())
  {
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::reset()
  {
    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    rho_min = Number(std::numeric_limits<ScalarNumber>::max());
    rho_max = Number(0.);

    rho_relaxation_numerator = Number(0.);
    rho_relaxation_denominator = Number(0.);

    if constexpr (limiter_ == Limiters::specific_entropy) {
      s_min = Number(std::numeric_limits<ScalarNumber>::max());
      s_interp_max = Number(0.);
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::accumulate(const rank1_type U_i,
                                   const rank1_type U_j,
                                   const rank1_type U_ij_bar,
                                   const bool is_diagonal_entry)
  {
    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    const auto rho_ij = U_ij_bar[0];
    rho_min = std::min(rho_min, rho_ij);
    rho_max = std::max(rho_max, rho_ij);



    if constexpr (limiter_ == Limiters::specific_entropy) {
      const auto s = ProblemDescription<dim, Number>::specific_entropy(U_j);
      s_min = std::min(s_min, s);

      if (!is_diagonal_entry) {
        const Number s_interp =
            ProblemDescription<dim, Number>::specific_entropy((U_i + U_j) *
                                                              ScalarNumber(.5));
        s_interp_max = std::max(s_interp_max, s_interp);
      }
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::reset_variations(const Number new_variations_i)
  {
    variations_i = new_variations_i;
  }


  template <int dim, typename Number>
  template <typename ITERATOR>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::accumulate_variations(const Number variations_j,
                                              const ITERATOR jt)
  {
    const auto beta_ij = get_entry(betaij_matrix_, jt);

    /* The numerical constant 8 is up to debate... */
    rho_relaxation_numerator +=
        Number(8.0 * 0.5) * beta_ij * (variations_i + variations_j);

    rho_relaxation_denominator += beta_ij;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::apply_relaxation(Number hd_i)
  {
    if constexpr (!relax_bounds_)
      return;

    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    const Number r_i =
        Number(2.) * dealii::Utilities::fixed_power<relaxation_order_>(
                         std::sqrt(std::sqrt(hd_i)));

    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    const Number rho_relaxation =
        std::abs(rho_relaxation_numerator) /
        (std::abs(rho_relaxation_denominator) + Number(eps));

    rho_min = std::max((Number(1.) - r_i) * rho_min, rho_min - rho_relaxation);
    rho_max = std::min((Number(1.) + r_i) * rho_max, rho_max + rho_relaxation);

    if constexpr (limiter_ == Limiters::specific_entropy) {
      s_min = std::max((Number(1.) - r_i) * s_min,
                       Number(2.) * s_min - s_interp_max);
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline const typename Limiter<dim, Number>::Bounds &
  Limiter<dim, Number>::bounds() const
  {
    return bounds_;
  }

} /* namespace grendel */

#endif /* LIMITER_H */
