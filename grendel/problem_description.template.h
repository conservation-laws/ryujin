#ifndef PROBLEM_DESCRIPTION_TEMPLATE_H
#define PROBLEM_DESCRIPTION_TEMPLATE_H

#include "problem_description.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  ProblemDescription<dim>::ProblemDescription(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    gamma_ = 5./3.;
    add_parameter("gamma", gamma_, "Gamma");

    b_ = 0.0;
    add_parameter("b", b_, "b aka bcovol");

    cfl_constant_ = 0.35;
    add_parameter("cfl constant", cfl_constant_, "CFL constant C");
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
