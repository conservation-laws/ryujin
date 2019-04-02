#ifndef LIMITER_TEMPLATE_H
#define LIMITER_TEMPLATE_H

#include "limiter.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  Limiter<dim>::Limiter(
      const grendel::ProblemDescription<dim> &problem_description,
      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , problem_description_(&problem_description)
  {
  }
} /* namespace grendel */

#endif /* LIMITER_TEMPLATE_H */
