#include "riemann_solver.template.h"

#include <deal.II/base/vectorization.h>

namespace grendel
{
  /* instantiations */
  template class grendel::RiemannSolver<DIM, NUMBER>;
  template class grendel::RiemannSolver<DIM, VectorizedArray<NUMBER>>;

} /* namespace grendel */
