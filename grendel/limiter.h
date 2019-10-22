#ifndef LIMITER_H
#define LIMITER_H

#include "helper.h"
#include "newton.h"
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

    static constexpr ScalarNumber newton_eps_ =
        std::is_same<ScalarNumber, double>::value ? ScalarNumber(1.0e-10)
                                                  : ScalarNumber(1.0e-4);

    static constexpr unsigned int newton_max_iter_ = 2;

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
     * Compute limiter value l_ij for update P:
     *
     * Returns the maximal t, obeying t_min < t < t_max such that the
     * selected local minimum principles are obeyed.
     */
    template <Limiters limiter = limiter_, typename BOUNDS>
    static Number limit(const BOUNDS bounds,
                        const rank1_type U,
                        const rank1_type P,
                        const Number t_min = Number(0.),
                        const Number t_max = Number(1.));

  private:
    Bounds bounds_;

    Number s_interp_max;
  };


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void Limiter<dim, Number>::reset()
  {
    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    rho_min = Number(std::numeric_limits<ScalarNumber>::max());
    rho_max = Number(0.);

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
  Limiter<dim, Number>::apply_relaxation(Number hd_i, Number rho_relaxation)
  {
    if constexpr (!relax_bounds_)
      return;

    auto &[rho_min, rho_max, s_min] = bounds_;

    if constexpr (limiter_ == Limiters::none)
      return;

    const Number r_i =
        Number(2.) * dealii::Utilities::fixed_power<relaxation_order_>(
                         std::sqrt(std::sqrt(hd_i)));

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


  template <int dim, typename Number>
  template <typename Limiter<dim, Number>::Limiters limiter, typename BOUNDS>
  DEAL_II_ALWAYS_INLINE inline Number Limiter<dim, Number>::limit(
      const BOUNDS bounds,
      const rank1_type U,
      const rank1_type P,
      const Number t_min /* = Number(0.) */,
      const Number t_max /* = Number(1.) */)
  {
    Number t_r = t_max;

    if constexpr (limiter_ == Limiters::none)
      return t_r;

    /*
     * First limit the density rho.
     *
     * See [Guermond, Nazarov, Popov, Thomas] (4.8):
     */

    {
      const auto &U_rho = U[0];
      const auto &P_rho = P[0];

      const auto &rho_min = std::get<0>(bounds);
      const auto &rho_max = std::get<1>(bounds);

      constexpr ScalarNumber eps_ =
          std::numeric_limits<ScalarNumber>::epsilon();

      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          rho_max,
          U_rho + t_r * P_rho,
          (std::abs(rho_max - U_rho) + eps_ * rho_min) /
              (std::abs(P_rho) + eps_ * rho_max),
          t_r);

      t_r = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          U_rho + t_r * P_rho,
          rho_min,
          (std::abs(rho_min - U_rho) + eps_ * rho_min) /
              (std::abs(P_rho) + eps_ * rho_max),
          t_r);

      /*
       * It is always t_min <= t <= t_max, but just to be sure, box
       * back into bounds:
       */
      t_r = std::min(t_r, t_max);
      t_r = std::max(t_r, t_min);

#ifdef DEBUG
      const auto new_density = (U + t_r * P)[0];

      if constexpr (std::is_same<Number, double>::value ||
                    std::is_same<Number, float>::value) {
        AssertThrow(new_density > 0.,
                    dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't do "
                                       "that. - Negative density."));
      } else {
        for (unsigned int k = 0; k < Number::n_array_elements; ++k)
          AssertThrow(new_density[k] > 0.,
                      dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                                         "do that. - Negative density."));
      }
#endif
    }

    if constexpr (limiter_ == Limiters::rho)
      return t_r;

    /*
     * Then limit the specific entropy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.6 + Section 5.1:
     */

    Number t_l = t_min;

    {
      /*
       * Prepare a quadratic Newton method:
       *
       * Given initial limiter values t_l and t_r with psi(t_l) > 0 and
       * psi(t_r) < 0 we try to find t^\ast with psi(t^\ast) \approx 0.
       *
       * Here, psi is a 3-convex function obtained by scaling the specific
       * entropy s:
       *
       *   psi = \rho ^ {\gamma + 1} s
       *
       * (s in turn was defined as s =\varepsilon \rho ^{-\gamma}, where
       * \varepsilon = (\rho e) is the internal energy.)
       */

      constexpr ScalarNumber gamma = ProblemDescription<dim, Number>::gamma;
      constexpr ScalarNumber gp1 = gamma + ScalarNumber(1.);

      const auto &s_min = std::get<2>(bounds);

      for (unsigned int n = 0; n < newton_max_iter_; ++n) {

        const auto U_r = U + t_r * P;
        const auto rho_r = U_r[0];
        const auto rho_r_gamma_plus_one = grendel::pow(rho_r, gp1);
        const auto e_r = ProblemDescription<dim, Number>::internal_energy(U_r);

        auto psi_r = rho_r * e_r - s_min * rho_r_gamma_plus_one;

        /*
         * Shortcut: In the majority of states no Newton step is necessary
         * because Psi(t_r) > 0. Just return in this case:
         */

        if (std::min(Number(0.), psi_r) == Number(0.)) {
          t_l = t_r;
          break;
        }

        /*
         * If psi_r > 0 the right state is fine, force returning t_r by
         * setting t_l = t_r:
         */

        t_l = dealii::compare_and_apply_mask<
            dealii::SIMDComparison::greater_than>(psi_r, Number(0.), t_r, t_l);

        const auto U_l = U + t_l * P;
        const auto rho_l = U_l[0];
        const auto rho_l_gamma_plus_one = grendel::pow(rho_l, gp1);
        const auto e_l = ProblemDescription<dim, Number>::internal_energy(U_l);

        auto psi_l = rho_l * e_l - s_min * rho_l_gamma_plus_one;

        /*
         * Shortcut: In the majority of cases only at most one Newton
         * iteration is necessary because we reach Psi(t_l) \approx 0
         * quickly. Just return in this case:
         */

        if (std::max(Number(0.), psi_l - Number(newton_eps_)) == Number(0.))
          break;

        const auto drho = P[0];

        const auto de_l =
            ProblemDescription<dim, Number>::internal_energy_derivative(U_l) *
            P;
        const auto de_r =
            ProblemDescription<dim, Number>::internal_energy_derivative(U_r) *
            P;

        const auto dpsi_l =
            rho_l * de_l -
            (e_l - gp1 * s_min * rho_l_gamma_plus_one / rho_l) * drho;

        const auto dpsi_r =
            rho_r * de_r -
            (e_r - gp1 * s_min * rho_r_gamma_plus_one / rho_r) * drho;

        quadratic_newton_step<true /*convex*/>(
            t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r);
      }

#ifdef DEBUG
      const auto U_new = U + t_l * P;
      const auto new_internal_energy =
          ProblemDescription<dim, Number>::internal_energy(U_new);
      const auto new_specific_entropy =
          ProblemDescription<dim, Number>::specific_entropy(U_new);
      auto psi = ProblemDescription<dim, Number>::internal_energy(U_new) -
                 s_min * grendel::pow(U_new[0], gamma);

      if constexpr (std::is_same<Number, double>::value ||
                    std::is_same<Number, float>::value) {
        AssertThrow(
            new_internal_energy > 0. && new_specific_entropy > 0.,
            dealii::ExcMessage(
                "I'm sorry, Dave. I'm afraid I can't do that. - Negative "
                "internal energy or negative specific entropy encountered."));
        AssertThrow( //
            psi > -100. * newton_eps_,
            dealii::ExcMessage(
                "I'm sorry, Dave. I'm afraid I can't do that. - Local minimum "
                "principle on specific entropy violated."));
      } else {
        for (unsigned int k = 0; k < Number::n_array_elements; ++k) {
          AssertThrow(
              new_internal_energy[k] > 0. && new_specific_entropy[k] > 0.,
              dealii::ExcMessage(
                  "I'm sorry, Dave. I'm afraid I can't do that. - Negative "
                  "internal energy or negative specific entropy encountered."));
          AssertThrow( //
              psi[k] > -100. * newton_eps_,
              dealii::ExcMessage(
                  "I'm sorry, Dave. I'm afraid I can't do that. - Local "
                  "minimum principle on specific entropy violated."));
        }
      }
#endif
    }

    /*
     * The quadratic Newton update does not allow t_r and t_l to go out of
     * bounds. I.e., t_l is good.
     */

    if constexpr (limiter_ == Limiters::specific_entropy)
      return t_l;

    /*
     * Limit based on an entropy inequality:
     */

    if constexpr (limiter_ == Limiters::entropy_inequality) {

      const auto &s_i = std::get<3>(bounds);
      const auto &s_j = std::get<4>(bounds);
    }

    return t_l;

    __builtin_unreachable();
  }

} /* namespace grendel */

#endif /* LIMITER_H */
