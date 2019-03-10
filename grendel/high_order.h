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
     * Indicator:
     */

    static constexpr unsigned int smoothness_power = 3;

    template <typename Vector, typename Index>
    inline DEAL_II_ALWAYS_INLINE double smoothness_indicator(const Vector &U,
                                                             Index i) const;

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

  private:
    /* Options: */


    static constexpr enum class Indicator {
      none,
      rho,
      internal_energy,
      pressure,
    } indicator_ = Indicator::rho;

    static constexpr enum class Limiters {
      none,
      rho,
      internal_energy,
      specific_entropy
    } limiters_ = Limiters::rho;

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
    }
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
     */
    {
      /* See [Guermond, Nazarov, Popov, Thomas] (4.8): */

      double l_ij_rho = 1.;

      const auto U_i_rho = U[0];
      const auto P_ij_rho = P_ij[0];

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
     */

    /* See [Guermond, Nazarov, Popov, Thomas], Section 4.5: */

    return l_ij;
  }

} /* namespace grendel */

#endif /* HIGH_ORDER_H */
