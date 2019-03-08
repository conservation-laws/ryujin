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

    Limiter(const grendel::ProblemDescription<dim> &problem_description,
            const std::string &subsection = "Limiter");

    inline DEAL_II_ALWAYS_INLINE double
    smoothness_indicator(const rank1_type &U) const;

    virtual ~Limiter() final = default;

  protected:
    dealii::SmartPointer<const grendel::ProblemDescription<dim>>
        problem_description_;
    ACCESSOR_READ_ONLY(problem_description)

  private:
    /* Options: */

    unsigned int smoothness_index_;

    unsigned int smoothness_power_;
    ACCESSOR_READ_ONLY(smoothness_power)

    double eps_;
  };


  template <int dim>
  inline DEAL_II_ALWAYS_INLINE double
  Limiter<dim>::smoothness_indicator(const rank1_type &U) const
  {
    // Density:
    return U[0];

    // Internal energy:
    // return problem_description.internal_energy(U);
  }

} /* namespace grendel */

#endif /* LIMITER_H */
