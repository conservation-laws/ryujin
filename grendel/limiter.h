#ifndef LIMITER_H
#define LIMITER_H

#include "helper.h"
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
        std::is_same<ScalarNumber, double>::value ? ScalarNumber(1.0e-10)
                                                  : ScalarNumber(1.0e-4);

    static constexpr unsigned int line_search_max_iter_ = 2;

    /*
     * Accumulate bounds:
     */

    void reset();

    void accumulate(const rank1_type U_i,
                    const rank1_type U_j,
                    const rank1_type U_ij_bar,
                    const bool is_diagonal_entry);

    void apply_relaxation(const Number hd_i, const Number rho_relaxation);

    const Bounds &bounds() const;

    /*
     * Compute limiter value l_ij for update P_ij:
     */

    static Number
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

    rho_min = Number(std::numeric_limits<ScalarNumber>::max());
    rho_max = Number(0.);

    if constexpr (limiter_ == Limiters::internal_energy) {
      rho_epsilon_min = Number(std::numeric_limits<ScalarNumber>::max());
    }

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

      if (!is_diagonal_entry) {
        const Number s_interp =
            ProblemDescription<dim, Number>::specific_entropy((U_i + U_j) /
                                                              ScalarNumber(2.));
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

    Number l_ij = Number(1.);

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
      constexpr ScalarNumber eps_ =
          std::numeric_limits<ScalarNumber>::epsilon();

      Number t_0 = Number(1.);

      t_0 = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          rho_max,
          U_i_rho + P_ij_rho,
          std::abs(rho_max - U_i_rho) / (std::abs(P_ij_rho) + eps_ * rho_max),
          t_0);

      t_0 = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          U_i_rho + P_ij_rho,
          rho_min,
          std::abs(rho_min - U_i_rho) / (std::abs(P_ij_rho) + eps_ * rho_max),
          t_0);

      l_ij = std::min(l_ij, t_0);

#ifdef DEBUG
      if constexpr (std::is_same<Number, double>::value ||
                    std::is_same<Number, float>::value) {
        Assert((U + l_ij * P_ij)[0] > 0.,
               dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't do "
                                  "that. - Negative density."));
      } else {
        for (unsigned int k = 0; k < Number::n_array_elements; ++k)
          Assert((U + l_ij * P_ij)[0][k] > 0.,
                 dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't do "
                                    "that. - Negative density."));
      }
#endif
    }

    if constexpr (limiter_ == Limiters::rho)
      return l_ij;

    /*
     * Then, limit the internal energy. (We skip this limiting step in case
     * we already limit the specific entropy because limiting the latter
     * implies that the internal energy stays in bounds).
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
       * Prepare a quadratic Newton method:
       *
       * Given initial limiter values t_l and t_r with psi(t_l) > 0 and
       * psi(t_r) < 0 we try to find t^\ast with psi(t^\ast) \approx 0.
       *
       * Note: If the precondition psi(t_l) > 0 and psi(t_r) < 0 is not
       * fulfilled we are very likely in a region of almost constant state.
       * Here, the limiter value ultimately doesn't matter that much. We
       * nevertheless have to make sure that we do not accidentally produce
       * nonsensical l_ij in the quadratic Newton iteration.
       */

      Number t_l = Number(0.);
      Number t_r = l_ij;

      constexpr ScalarNumber gamma = ProblemDescription<dim, Number>::gamma;

      for (unsigned int n = 0; n < line_search_max_iter_; ++n) {

        const auto U_r = U + t_r * P_ij;
        const auto rho_r = U_r[0];
        const auto rho_r_gamma = grendel::pow(rho_r, gamma);
        /* use a scaled variant of psi:  e - s * rho^gamma */
        auto psi_r = ProblemDescription<dim, Number>::internal_energy(U_r) -
                     s_min * rho_r_gamma;

        /*
         * Shortcut: In the majority of states no Newton step is necessary
         * because Psi(t_r) > 0. Just return in this case:
         */

        if (std::min(Number(0.), psi_r + Number(line_search_eps_)) ==
            Number(0.))
          return std::min(l_ij, t_r);

        /*
         * If psi_r > 0 the right state is fine, force l_ij = t_r by
         * setting t_l = t_r:
         */

        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        const auto U_l = U + t_l * P_ij;
        const auto rho_l = U_l[0];
        const auto rho_l_gamma = grendel::pow(rho_l, gamma);
        /* use a scaled variant of psi:  e - s * rho^gamma */
        auto psi_l = ProblemDescription<dim, Number>::internal_energy(U_l) -
                     s_min * rho_l_gamma;

        /*
         * Shortcut: In the majority of cases only at most one Newton
         * iteration is necessary because we reach Psi(t_l) \approx 0
         * quickly. Just return in this case:
         */

        if (std::max(Number(0.), psi_l - Number(line_search_eps_)) ==
            Number(0.))
          return std::min(l_ij, t_l);

        const auto dpsi_l =
            ProblemDescription<dim, Number>::internal_energy_derivative(U_l) *
                P_ij -
            gamma * rho_l_gamma / rho_l * s_min * P_ij[0];

        const auto dpsi_r =
            ProblemDescription<dim, Number>::internal_energy_derivative(U_r) *
                P_ij -
            gamma * rho_r_gamma / rho_r * s_min * P_ij[0];

        /* Compute divided differences: */

        const auto scaling =
            ScalarNumber(1.) / (t_r - t_l + Number(line_search_eps_));

        const auto dd_11 = -dpsi_l;
        const auto dd_12 = (psi_l - psi_r) * scaling;
        const auto dd_22 = -dpsi_r;

        const auto dd_112 = (dd_12 - dd_11) * scaling;
        const auto dd_122 = (dd_22 - dd_12) * scaling;

        /* Update left and right point: */

        const auto discriminant_l =
            std::abs(dpsi_l * dpsi_l + ScalarNumber(4.) * psi_l * dd_112);
        const auto discriminant_r =
            std::abs(dpsi_r * dpsi_r + ScalarNumber(4.) * psi_r * dd_122);

        t_l -= ScalarNumber(2.) * psi_l /
               (dpsi_l - std::sqrt(discriminant_l) + Number(line_search_eps_));

        t_r -= ScalarNumber(2.) * psi_r /
               (dpsi_r - std::sqrt(discriminant_r) + Number(line_search_eps_));

        /*
         * In pathological cases (where the states are constant) we might
         * end up with pathological t_l and t_r values. Simply clean up t_l
         * and t_r (what we choose as actual value for l_ij in this case
         * doesn't really matter).
         */

        t_l = std::max(Number(0.), t_l);
        t_l = std::min(Number(1.), t_l);
        t_r = std::max(Number(0.), t_r);
        t_r = std::min(Number(1.), t_r);

        /* Ensure that always t_l <= t_r: */

        t_l = std::min(t_l, t_r);
      }

      return std::min(l_ij, t_l);
    }

    __builtin_unreachable();
  }

} /* namespace grendel */

#endif /* LIMITER_H */
