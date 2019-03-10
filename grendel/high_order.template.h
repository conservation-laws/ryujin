#ifndef HIGH_ORDER_TEMPLATE_H
#define HIGH_ORDER_TEMPLATE_H

#include "high_order.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  HighOrder<dim>::HighOrder(
      const grendel::ProblemDescription<dim> &problem_description,
      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , problem_description_(&problem_description)
  {
  }
} /* namespace grendel */

#endif /* HIGH_ORDER_TEMPLATE_H */
