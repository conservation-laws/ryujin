#ifndef RIEMANN_SOLVER_TEMPLATE_H
#define RIEMANN_SOLVER_TEMPLATE_H

#include "riemann_solver.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  RiemannSolver<dim>::RiemannSolver(
    const grendel::ProblemDescription<dim> &problem_description,
    const std::string &subsection)
      : ParameterAcceptor(subsection)
      , problem_description_(&problem_description)
      , gamma_(problem_description.gamma())
      , b_(problem_description.b())
  {
    eps_ = 1.e-10;
    add_parameter("newton eps", eps_, "Tolerance of the Newton secant solver");

    max_iter_ = 10;
    add_parameter("newton max iter",
                  max_iter_,
                  "Maximal number of iterations for the Newton secant solver");
  }

} /* namespace grendel */

#endif /* RIEMANN_SOLVER_TEMPLATE_H */
