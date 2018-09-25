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

    eps_ = 1.e-10;
    add_parameter("newton eps", eps_, "Tolerance of the Newton secant solver");

    max_iter_ = 10;
    add_parameter("newton max iter",
                  max_iter_,
                  "Maximal number of iterations for the Newton secant solver");
  }

} /* namespace grendel */

#endif /* MODEL_TEMPLATE_H */
