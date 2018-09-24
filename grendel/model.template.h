#ifndef MODEL_TEMPLATE_H
#define MODEL_TEMPLATE_H

#include "model.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  Model<dim>::Model(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    gamma_ = 1.4;
    add_parameter("gamma", gamma_, "Gamma");

    b_ = 0.0;
    add_parameter("b", b_, "b aka bcovol");
  }

} /* namespace grendel */

#endif /* MODEL_TEMPLATE_H */
