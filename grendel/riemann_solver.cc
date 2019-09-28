#include "riemann_solver.template.h"

#include <deal.II/base/vectorization.h>

namespace grendel
{
  template class grendel::RiemannSolver<1>;
  template class grendel::RiemannSolver<2>;
  template class grendel::RiemannSolver<3>;

  template class grendel::RiemannSolver<1, float>;
  template class grendel::RiemannSolver<2, float>;
  template class grendel::RiemannSolver<3, float>;

  template class grendel::RiemannSolver<1, VectorizedArray<double>>;
  template class grendel::RiemannSolver<2, VectorizedArray<double>>;
  template class grendel::RiemannSolver<3, VectorizedArray<double>>;

  template class grendel::RiemannSolver<1, VectorizedArray<float>>;
  template class grendel::RiemannSolver<2, VectorizedArray<float>>;
  template class grendel::RiemannSolver<3, VectorizedArray<float>>;
} /* namespace grendel */
