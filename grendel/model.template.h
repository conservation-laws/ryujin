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
//     add_parameter(
//         &description_,
//         "description",
//         "dipole",
//         Patterns::Selection("dipole|waveguide"),
//         "the problem description to use; valid options are \"dipole\" and \"waveguide\"");
  }

} /* namespace grendel */

#endif /* MODEL_TEMPLATE_H */
