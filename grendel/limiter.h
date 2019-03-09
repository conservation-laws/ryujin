#ifndef LIMITER_H
#define LIMITER_H

#include "problem_description.h"

namespace grendel
{

  template <int dim>
  class Limiter : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    /*
     * Let's allocate 3 double's for limiter bounds:
     */
    typedef std::array<double, 3> Bounds;

    Limiter(const grendel::ProblemDescription<dim> &problem_description,
            const std::string &subsection = "Limiter");

    virtual ~Limiter() final = default;

    template <typename Vector, typename Index>
    inline DEAL_II_ALWAYS_INLINE double smoothness_indicator(const Vector &U,
                                                             Index i) const;

    inline DEAL_II_ALWAYS_INLINE void reset(Bounds &bounds) const;

    inline DEAL_II_ALWAYS_INLINE void accumulate(Bounds &bounds,
                                                 const rank1_type &U) const;

    inline DEAL_II_ALWAYS_INLINE double
    limit(Bounds &bounds, const rank1_type &U, const rank1_type &P_ij) const;

  protected:
    dealii::SmartPointer<const grendel::ProblemDescription<dim>>
        problem_description_;
    ACCESSOR_READ_ONLY(problem_description)

  private:
    /* Options: */

    unsigned int smoothness_power_;
    ACCESSOR_READ_ONLY(smoothness_power)

    static constexpr enum class Indicator {
      rho,
      internal_energy,
      pressure,
    } indicator_ = Indicator::rho;

    static constexpr enum class Limiters {
      rho,
      internal_energy,
      specific_entropy
    } limiters_ = Limiters::rho;

  };


  template <int dim>
  template <typename Vector, typename Index>
  inline DEAL_II_ALWAYS_INLINE double
  Limiter<dim>::smoothness_indicator(const Vector &U, Index i) const
  {
    switch (indicator_) {

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
  Limiter<dim>::reset(Bounds &bounds) const
  {
    auto &[rho_min, rho_max, rho_epsilon_min] = bounds;
    rho_min = std::numeric_limits<double>::max();
    rho_max = 0.;
    rho_epsilon_min = std::numeric_limits<double>::max();
  }


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Limiter<dim>::accumulate(Bounds &bounds, const rank1_type &U) const
  {
    auto &[rho_min, rho_max, rho_epsilon_min] = bounds;

    switch (limiters_) {
    case Limiters::internal_energy:
      {
        const auto rho_epsilon = problem_description_->internal_energy(U);
        rho_epsilon_min = std::min(rho_epsilon_min, rho_epsilon);
      }
      [[fallthrough]];
    case Limiters::rho:
      {
        const auto rho = U[0];
        rho_min = std::min(rho_min, rho);
        rho_max = std::max(rho_max, rho);
      }
    }
  }

  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double Limiter<dim>::limit(
      Bounds &bounds, const rank1_type &U, const rank1_type &P_ij) const
  {
    auto &[rho_min, rho_max, rho_epsilon_min] = bounds;

    double l_ij = 1.;

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

    if (limiters_ == Limiters::rho)
      return l_ij;

    /*
     * Then, limit the internal energy:
     */

    /* See [Guermond, Nazarov, Popov, Thomas], Section 4.5: */

    return l_ij;
  }

} /* namespace grendel */

#endif /* LIMITER_H */
