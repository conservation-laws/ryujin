#include "riemann_solver.template.h"

namespace grendel
{
  template class grendel::RiemannSolver<1>;
  template class grendel::RiemannSolver<2>;
  template class grendel::RiemannSolver<3>;

  template class grendel::RiemannSolver<1, float>;
  template class grendel::RiemannSolver<2, float>;
  template class grendel::RiemannSolver<3, float>;
} /* namespace grendel */
