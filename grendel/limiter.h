#ifndef LIMITER_H
#define LIMITER_H

#include "helper.h"
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

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    using vector_type =
        std::array<dealii::LinearAlgebra::distributed::Vector<Number>, 4>;

    using ScalarNumber = typename get_value_type<Number>::type;

    using Bounds = std::array<Number, 4>;

    /*
     * Options:
     */

    static constexpr enum class Limiters {
      none,
      rho,
      internal_energy,
      specific_entropy
    } limiter_ = Limiters::specific_entropy;

    static constexpr bool relax_bounds_ = true;

    static constexpr unsigned int relaxation_order_ = 3;

    static constexpr ScalarNumber line_search_eps_ =
        std::is_same<Number, double>::value ? 1.0e-8 : 1.0e-4;

    static constexpr unsigned int line_search_max_iter_ = 10;

    /*
     * Accumulate bounds:
     */

    DEAL_II_ALWAYS_INLINE inline void reset();

    template <typename ITERATOR>
    DEAL_II_ALWAYS_INLINE inline void accumulate(const rank1_type U_i,
                                                 const rank1_type U_j,
                                                 const rank1_type U_ij_bar,
                                                 const ITERATOR jt); // FIXME

    DEAL_II_ALWAYS_INLINE inline void
    apply_relaxation(const Number hd_i, const Number rho_relaxation);

    DEAL_II_ALWAYS_INLINE inline const Bounds &bounds() const;

    /*
     * Compute limiter value l_ij for update P_ij:
     */

    static DEAL_II_ALWAYS_INLINE inline Number
    limit(const Bounds &bounds, const rank1_type U, const rank1_type P_ij);

  private:
    Bounds bounds_;

    Number s_interp_max;
  };


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::reset()
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    rho_min = std::numeric_limits<Number>::max();
    rho_max = 0.;

    if constexpr (limiter_ == Limiters::internal_energy) {
      rho_epsilon_min = std::numeric_limits<Number>::max();
    }

    if constexpr (limiter_ == Limiters::specific_entropy) {
      s_min = std::numeric_limits<Number>::max();
      s_interp_max = 0.;
    }
  }


  template <int dim, typename Number>
  template <typename ITERATOR>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::accumulate(const rank1_type U_i,
                                   const rank1_type U_j,
                                   const rank1_type U_ij_bar,
                                   const ITERATOR jt) // FIXME
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    const auto rho_ij = U_ij_bar[0];
    rho_min = std::min(rho_min, rho_ij);
    rho_max = std::max(rho_max, rho_ij);

    if constexpr (limiter_ == Limiters::internal_energy) {
      const auto rho_epsilon =
          ProblemDescription<dim, Number>::internal_energy(U_ij_bar);
      rho_epsilon_min = std::min(rho_epsilon_min, rho_epsilon);
    }

    if constexpr (limiter_ == Limiters::specific_entropy) {
      const auto s = ProblemDescription<dim, Number>::specific_entropy(U_j);
      s_min = std::min(s_min, s);

      if (jt->row() != jt->column()) { // FIXME
        const Number s_interp =
            ProblemDescription<dim, Number>::specific_entropy((U_i + U_j) / 2.);
        s_interp_max = std::max(s_interp_max, s_interp);
      }
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void
  Limiter<dim, Number>::apply_relaxation(Number hd_i, Number rho_relaxation)
  {
    if constexpr (relax_bounds_) {
      auto &[rho_min, rho_max, rho_epsilon_min, s_min] = bounds_;

      if constexpr (limiter_ == Limiters::none)
        return;

      const Number r_i =
          Number(2.) * dealii::Utilities::fixed_power<relaxation_order_>(
                           std::sqrt(std::sqrt(hd_i)));

      rho_min =
          std::max((Number(1.) - r_i) * rho_min, rho_min - rho_relaxation);
      rho_max =
          std::min((Number(1.) + r_i) * rho_max, rho_max + rho_relaxation);

      if constexpr (limiter_ == Limiters::specific_entropy) {
        s_min = std::max((Number(1.) - r_i) * s_min,
                         Number(2.) * s_min - s_interp_max);
      }
    }
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline const typename Limiter<dim, Number>::Bounds &
  Limiter<dim, Number>::bounds() const
  {
    return bounds_;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number Limiter<dim, Number>::limit(
      const Bounds &bounds, const rank1_type U, const rank1_type P_ij)
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min] = bounds;

    Number l_ij = 1.;

    if constexpr (limiter_ == Limiters::none)
      return l_ij;

    /*
     * First limit rho.
     *
     * See [Guermond, Nazarov, Popov, Thomas] (4.8):
     */

    const auto &U_i_rho = U[0];
    const auto &P_ij_rho = P_ij[0];

    {
      Number t_0 = 1.;

      constexpr ScalarNumber eps_ =
          std::numeric_limits<ScalarNumber>::epsilon();

      const Number temp =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
              U_i_rho + P_ij_rho,
              rho_max,
              std::abs(rho_max - U_i_rho) /
                  (std::abs(P_ij_rho) + eps_ * rho_max),
              t_0);

      t_0 =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
              rho_min,
              U_i_rho + P_ij_rho,
              std::abs(rho_min - U_i_rho) /
                  (std::abs(P_ij_rho) + eps_ * rho_max),
              temp);

      l_ij = std::min(l_ij, t_0);

      // FIXME
      if constexpr (std::is_same<Number, double>::value ||
                    std::is_same<Number, float>::value) {
        Assert(
            (U + l_ij * P_ij)[0] > 0.,
            dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't do that. "
                               "- Negative density."));
      }
    }

    if constexpr (limiter_ == Limiters::rho)
      return l_ij;

    /*
     * Then, limit the internal energy. (We skip this limiting step in case
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.5:
     */

    if constexpr (limiter_ == Limiters::internal_energy) {

      static_assert(std::is_same<Number, double>::value ||
                        std::is_same<Number, float>::value,
                    "Not implemented yet");


      const auto P_ij_m = ProblemDescription<dim, Number>::momentum(P_ij);
      const auto &P_ij_E = P_ij[dim + 1];

      const auto U_i_m = ProblemDescription<dim, Number>::momentum(U);
      const Number &U_i_E = U[dim + 1];

      const Number c = (U_i_E - rho_epsilon_min) * U_i_rho -
                       Number(1. / 2.) * U_i_m.norm_square();

      const Number b = (U_i_E - rho_epsilon_min) * P_ij_rho + P_ij_E * U_i_rho -
                       U_i_m * P_ij_m;

      const Number a =
          P_ij_E * P_ij_rho - Number(1. / 2.) * P_ij_m.norm_square();

      /*
       * Solve the quadratic equation a t^2 + b t + c = 0 by hand. We use the
       * Ciatardauq formula to avoid numerical cancellation and some if
       * statements:
       */

      Number t_0 = 1.;

      const Number discriminant = b * b - Number(4.) * a * c;

      if (discriminant == 0.) {

        const Number x = -b / Number(2.) / a;

        if (x > 0.)
          t_0 = x;

      } else if (discriminant > 0.) {

        const Number x =
            Number(2.) * c / (-b - std::copysign(std::sqrt(discriminant), b));
        const Number y = c / a / x;

        /* Select the smallest positive root: */
        if (x > 0.) {
          if (y > 0. && y < x)
            t_0 = y;
          else
            t_0 = x;
        } else if (y > 0.) {
          t_0 = y;
        }
      }

      l_ij = std::min(l_ij, t_0);

#ifdef DEBUG
      const Number rho_epsilon =
          ProblemDescription<dim, Number>::internal_energy(U + l_ij * P_ij);
      Assert(rho_epsilon - rho_epsilon_min > 0.,
             dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't do that. "
                                "- Negative internal energy."));
#endif

      return l_ij;
    }

    /*
     * And finally, limit the specific entropy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.6 + Section 5.1:
     */

    if constexpr (limiter_ == Limiters::specific_entropy) {
      /*
       * Prepare a Newton secant method:
       */

      Number t_l = 0.;
      Number t_r = l_ij;

      if (t_r <= ScalarNumber(0.) + line_search_eps_)
        return 0.;

      if (t_r < t_l + line_search_eps_) {
        const auto t = t_l < t_r ? t_l : t_r;
        return std::min(l_ij, t);
      }

      constexpr ScalarNumber gamma = ProblemDescription<dim, Number>::gamma;

      for (unsigned int n = 0; n < line_search_max_iter_; ++n) {

        const auto U_r = U + t_r * P_ij;
        const auto rho_r = U_r[0];
        const auto rho_r_gamma = std::pow(rho_r, gamma);
        auto psi_r = ProblemDescription<dim, Number>::internal_energy(U_r) -
                     s_min * rho_r_gamma;

        /* Right state is good, cut it short and return: */
        if (psi_r >= 0. - line_search_eps_)
          return std::min(l_ij, t_r);

        const auto U_l = U + t_l * P_ij;
        const auto rho_l = U_l[0];
        const auto rho_l_gamma = std::pow(rho_l, gamma);
        auto psi_l = ProblemDescription<dim, Number>::internal_energy(U_l) -
                     s_min * rho_l_gamma;

        /*
         * Due to round-off errors it might happen that psi_l is negative
         * (close to eps). In this case we fix the problem by lowering
         * s_min just enough so that psi_l = 0.;
         */
        if (psi_l < 0.) {
          psi_r -= psi_l;
          psi_l = 0.;
          if (psi_r >= ScalarNumber(0.) - line_search_eps_)
            return std::min(l_ij, t_r);
        }

        const auto dpsi_l =
            ProblemDescription<dim, Number>::internal_energy_derivative(U_l) *
                P_ij -
            gamma * rho_l_gamma / rho_l * s_min * P_ij[0];
        const auto dpsi_r =
            ProblemDescription<dim, Number>::internal_energy_derivative(U_r) *
                P_ij -
            gamma * rho_r_gamma / rho_r * s_min * P_ij[0];

        /* Compute divided differences: */

        const Number dd_11 = -dpsi_l;
        const Number dd_12 = (psi_l - psi_r) / (t_r - t_l);
        const Number dd_22 = -dpsi_r;

        const Number dd_112 = (dd_12 - dd_11) / (t_r - t_l);
        const Number dd_122 = (dd_22 - dd_12) / (t_r - t_l);

        /* Update left and right point: */

        const Number discriminant_l = dpsi_l * dpsi_l + 4. * psi_l * dd_112;
        const Number discriminant_r = dpsi_r * dpsi_r + 4. * psi_r * dd_122;

        Assert(discriminant_l >= 0. && discriminant_r >= 0.,
               dealii::ExcMessage("Houston we have a problem!"));

        t_l = t_l - 2. * psi_l / (dpsi_l - std::sqrt(discriminant_l));
        t_r = t_r - 2. * psi_r / (dpsi_r - std::sqrt(discriminant_r));

        /* Handle some pathological cases that happen in regions with
         * constant specific entropy: */
        if (std::isnan(t_l)) {
          return l_ij;
        }
        if (std::isnan(t_r)) {
          return l_ij;
        }

        if (t_r < t_l + line_search_eps_) {
          const auto t = t_l < t_r ? t_l : t_r;
          return std::min(l_ij, t);
        }
      }

      /* t_l is a good state with psi_l > 0. */
      return std::min(l_ij, t_l);
    }

    __builtin_unreachable();
  }

} /* namespace grendel */

#endif /* LIMITER_H */
