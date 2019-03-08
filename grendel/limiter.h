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

    enum class Indicator { rho, internal_energy } indicator_ = Indicator::rho;

    enum class Limiters { rho, internal_energy } limiters_ = Limiters::rho;

    double eps_;
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
    }
  }

  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Limiter<dim>::reset(Bounds &bounds) const
  {
  }

  template <int dim>
  inline DEAL_II_ALWAYS_INLINE void
  Limiter<dim>::accumulate(Bounds &bounds, const rank1_type &U) const
  {
  }

  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double Limiter<dim>::limit(
      Bounds &bounds, const rank1_type &U, const rank1_type &P_ij) const
  {
    return 1.;
  }

} /* namespace grendel */

#endif /* LIMITER_H */
