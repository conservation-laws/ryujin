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
    smoothness_index_ = 0;
    add_parameter("smoothness index",
                  smoothness_index_,
                  "Use the corresponding component of the state vector for the "
                  "smoothness indicator");

    smoothness_power_ = 3;
    add_parameter("smoothness power",
                  smoothness_power_,
                  "Sets the exponent for the smoothness indicator");

    eps_ = std::numeric_limits<double>::epsilon();
  }
} /* namespace grendel */

#endif /* LIMITER_TEMPLATE_H */
