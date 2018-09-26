#ifndef PROBLEM_DESCRIPTION_TEMPLATE_H
#define PROBLEM_DESCRIPTION_TEMPLATE_H

#include "compressible_euler.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  ProblemDescription<dim>::ProblemDescription(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    gamma_ = 1.4;
    add_parameter("gamma", gamma_, "Gamma");

    b_ = 0.0;
    add_parameter("b", b_, "b aka bcovol");
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
