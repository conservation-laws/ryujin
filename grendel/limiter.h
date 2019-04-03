#ifndef LIMITER_H
#define LIMITER_H

#include "problem_description.h"

#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim>
  class Limiter
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    /* Let's allocate 5 doubles for limiter bounds: */
    typedef std::array<double, 5> Bounds;

    typedef std::array<dealii::LinearAlgebra::distributed::Vector<double>, 5>
        vector_type;

    /*
     * Options:
     */

    static constexpr enum class Limiters {
      none,
      rho,
      internal_energy,
      specific_entropy
    } limiters_ = Limiters::internal_energy;

    /*
     * Accumulate bounds:
     */

    inline DEAL_II_ALWAYS_INLINE void reset();

    inline DEAL_II_ALWAYS_INLINE void accumulate(const rank1_type &U);

    inline DEAL_II_ALWAYS_INLINE const Bounds &bounds() const;

    /*
     * Compute limiter value l_ij for update P_ij:
     */

    static inline DEAL_II_ALWAYS_INLINE double
    limit(const Bounds &bounds, const rank1_type &U, const rank1_type &P_ij);

  private:
    Bounds bounds_;
  };


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Limiter<dim>::reset()
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min, s_laplace] = bounds_;
    rho_min = std::numeric_limits<double>::max();
    rho_max = 0.;
    rho_epsilon_min = std::numeric_limits<double>::max();
    s_min = std::numeric_limits<double>::max();
    s_laplace = 0.;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Limiter<dim>::accumulate(const rank1_type &U)
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min, s_laplace] = bounds_;

    if constexpr(limiters_ == Limiters::none)
      return;

    const auto rho = U[0];
    rho_min = std::min(rho_min, rho);
    rho_max = std::max(rho_max, rho);

    if constexpr(limiters_ == Limiters::rho)
      return;

    const auto rho_epsilon = ProblemDescription<dim>::internal_energy(U);
    rho_epsilon_min = std::min(rho_epsilon_min, rho_epsilon);

    if constexpr(limiters_ == Limiters::internal_energy)
      return;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE const typename Limiter<dim>::Bounds &
  Limiter<dim>::bounds() const
  {
    return bounds_;
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double Limiter<dim>::limit(
      const Bounds &bounds, const rank1_type &U, const rank1_type &P_ij)
  {
    auto &[rho_min, rho_max, rho_epsilon_min, s_min, s_laplace] = bounds;

    double l_ij = 1.;

    if constexpr(limiters_ == Limiters::none)
      return l_ij;

    /*
     * First limit rho:
     *
     * See [Guermond, Nazarov, Popov, Thomas] (4.8):
     */

    const auto &U_i_rho = U[0];
    const auto &P_ij_rho = P_ij[0];

    {
      double l_ij_rho = 1.;

      constexpr double eps_ = std::numeric_limits<double>::epsilon();

      if (U_i_rho + P_ij_rho < rho_min)
        l_ij_rho = std::abs(rho_min - U_i_rho) /
                   (std::abs(P_ij_rho) + eps_ * rho_max);

      else if (rho_max < U_i_rho + P_ij_rho)
        l_ij_rho = std::abs(rho_max - U_i_rho) /
                   (std::abs(P_ij_rho) + eps_ * rho_max);

      l_ij = std::min(l_ij, l_ij_rho); // ensures that l_ij <= 1

      AssertThrow((U + l_ij * P_ij)[0] > 0.,
                  dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                                     "do that. - Negative density."));
    }

    if constexpr(limiters_ == Limiters::rho)
      return l_ij;

    /*
     * Then, limit the internal energy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.5:
     */

    if constexpr (limiters_ == Limiters::internal_energy) {

      const auto P_ij_m = ProblemDescription<dim>::momentum(P_ij);
      const auto &P_ij_E = P_ij[dim + 1];

      const auto U_i_m = ProblemDescription<dim>::momentum(U);
      const double &U_i_E = U[dim + 1];

      const double c =
          (U_i_E - rho_epsilon_min) * U_i_rho - 1. / 2. * U_i_m.norm_square();

      const double b = (U_i_E - rho_epsilon_min) * P_ij_rho +
                       P_ij_E * U_i_rho - U_i_m * P_ij_m;

      const double a = P_ij_E * P_ij_rho  - 1. / 2. * P_ij_m.norm_square();

      /*
       * Solve the quadratic equation a t^2 + b t + c = 0 by hand. We use the
       * Ciatardauq formula to avoid numerical cancellation and some if
       * statements:
       */

      double l_ij_rhoe = 1.;

      const double discriminant = b * b - 4. * a * c;

      if (discriminant == 0.) {

        const double x = -b / 2. / a;

        if (x > 0)
          l_ij_rhoe = x;

      } else if (discriminant > 0.) {

        const double x =
            2. * c / (-b - std::copysign(std::sqrt(discriminant), b));
        const double y = c / a / x;

        /*
         * Select the smallest positive root:
         */

        if (x > 0.) {
          if (y > 0. && y < x)
            l_ij_rhoe = y;
          else
            l_ij_rhoe = x;
        } else if (y > 0.) {
          l_ij_rhoe = y;
        }
      }

      l_ij = std::min(l_ij, l_ij_rhoe);

      AssertThrow(ProblemDescription<dim>::internal_energy(U + l_ij * P_ij) > 0.,
                  dealii::ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                                     "do that. - Negative internal energy."));

      return l_ij;
    }

    /*
     * And finally, limit the specific entropy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.6:
     */

    if constexpr (limiters_ == Limiters::specific_entropy)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      return l_ij;
    }

    __builtin_unreachable();
  }

} /* namespace grendel */

#endif /* LIMITER_H */
