#ifndef HIGH_ORDER_H
#define HIGH_ORDER_H

#include "problem_description.h"

#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim>
  class HighOrder : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    /*
     * Let's allocate 3 double's for limiter bounds:
     */
    typedef std::array<double, 3> Bounds;

    typedef std::array<dealii::LinearAlgebra::distributed::Vector<double>, 3>
        vector_type;

    HighOrder(const grendel::ProblemDescription<dim> &problem_description,
              const std::string &subsection = "HighOrder");

    virtual ~HighOrder() final = default;

    /*
     * Options:
     */

    static constexpr enum class Indicator {
      none,
      rho,
      internal_energy,
      pressure,
      specific_entropy,
    } indicator_ = Indicator::pressure;

    static constexpr enum class Limiters {
      none,
      rho,
      internal_energy,
      specific_entropy
    } limiters_ = Limiters::internal_energy;

    /*
     * Indicator:
     */

    template <typename Vector, typename Index>
    inline DEAL_II_ALWAYS_INLINE double smoothness_indicator(const Vector &U,
                                                             Index i) const;

    static constexpr double alpha_0 = 0;
    static constexpr unsigned int smoothness_power = 3;

    static inline DEAL_II_ALWAYS_INLINE double psi(const double ratio);

    /*
     * Limiter:
     */

    inline DEAL_II_ALWAYS_INLINE void reset(Bounds &bounds) const;

    inline DEAL_II_ALWAYS_INLINE void accumulate(Bounds &bounds,
                                                 const rank1_type &U) const;

    inline DEAL_II_ALWAYS_INLINE double limit(const Bounds &bounds,
                                              const rank1_type &U,
                                              const rank1_type &P_ij) const;

  protected:
    dealii::SmartPointer<const grendel::ProblemDescription<dim>>
        problem_description_;
    ACCESSOR_READ_ONLY(problem_description)
  };


  template <int dim>
  template <typename Vector, typename Index>
  inline DEAL_II_ALWAYS_INLINE double
  HighOrder<dim>::smoothness_indicator(const Vector &U, Index i) const
  {
    switch (indicator_) {
    case Indicator::none:
      return 1.;

    case Indicator::rho:
      return U[0][i];

    case Indicator::internal_energy:
      return problem_description_->internal_energy(gather(U, i));

    case Indicator::pressure:
      return problem_description_->pressure(gather(U, i));

    case Indicator::specific_entropy:
      return problem_description_->specific_entropy(gather(U, i));
    }
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double
  HighOrder<dim>::psi(const double ratio)
  {
    return std::pow(std::max(ratio - alpha_0, 0.), smoothness_power) /
           std::pow(1 - alpha_0, smoothness_power);
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  HighOrder<dim>::reset(Bounds &bounds) const
  {
    auto &[rho_min, rho_max, rho_epsilon_min] = bounds;
    rho_min = std::numeric_limits<double>::max();
    rho_max = 0.;
    rho_epsilon_min = std::numeric_limits<double>::max();
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  HighOrder<dim>::accumulate(Bounds &bounds, const rank1_type &U) const
  {
    auto &[rho_min, rho_max, rho_epsilon_min] = bounds;

    if constexpr(limiters_ == Limiters::none)
      return;

    const auto rho = U[0];
    rho_min = std::min(rho_min, rho);
    rho_max = std::max(rho_max, rho);

    if constexpr(limiters_ == Limiters::rho)
      return;

    const auto rho_epsilon = problem_description_->internal_energy(U);
    rho_epsilon_min = std::min(rho_epsilon_min, rho_epsilon);


    if constexpr(limiters_ == Limiters::internal_energy)
      return;
  }

  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double HighOrder<dim>::limit(
      const Bounds &bounds, const rank1_type &U, const rank1_type &P_ij) const
  {
    auto &[rho_min, rho_max, rho_epsilon_min] = bounds;

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
    }

    if constexpr(limiters_ == Limiters::rho)
      return l_ij;

    /*
     * Then, limit the internal energy:
     *
     * See [Guermond, Nazarov, Popov, Thomas], Section 4.5:
     */

    const auto P_ij_m = problem_description_->momentum_vector(P_ij);
    const auto &P_ij_E = P_ij[dim + 1];

    const auto U_i_m = problem_description_->momentum_vector(U);
    const double &U_i_E = U[dim + 1];

    {
      double l_ij_rhoe = 1.;

      const double a = P_ij_rho * P_ij_E + 1. / 2. * P_ij_m.norm_square();

      const double b = (U_i_E - rho_epsilon_min) * P_ij_rho - U_i_m * P_ij_m +
                       U_i_rho * P_ij_rho;

      const double c = U_i_rho * U_i_E - 1. / 2. * U_i_m.norm_square() -
                       rho_epsilon_min * U_i_rho;

      /*
       * Solve the quadratic equation a t^2 + b t + c = 0 by hand. We use the
       * Ciatardauq formula to avoid numerical cancellation and some if
       * statements:
       */

      const double discriminant = b * b - 4. * a * c;

      if (discriminant == 0.) {

        l_ij_rhoe = -b / 2. / a;

      } else if (discriminant > 0.) {

        const double x = 2. * c / (-b - std::copysign(discriminant, b));
        const double y = c / a / x;

        if (x > 0 && x < y)
          l_ij_rhoe = x;
        else if (y > 0)
          l_ij_rhoe = y;
      }

      if (l_ij_rhoe < 0.)
        l_ij_rhoe = 1.;

      l_ij = std::min(l_ij, l_ij_rhoe); // ensures that l_ij <= 1
    }

    if constexpr (limiters_ == Limiters::internal_energy)
      return l_ij;

    static_assert(limiters_ != Limiters::specific_entropy,
                  "not implemented, sorry");

    return l_ij;
  }

} /* namespace grendel */

#endif /* HIGH_ORDER_H */
