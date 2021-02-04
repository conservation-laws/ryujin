#include <limiter.h>
#include <simd.template.h>

#include <deal.II/base/vectorization.h>

using namespace ryujin;
using namespace dealii;

/* Explicitly instantiate all variants: */

template class ryujin::Limiter<1>;
template class ryujin::Limiter<2>;
template class ryujin::Limiter<3>;

template class ryujin::Limiter<1, float>;
template class ryujin::Limiter<2, float>;
template class ryujin::Limiter<3, float>;

template class ryujin::Limiter<1, VectorizedArray<double>>;
template class ryujin::Limiter<2, VectorizedArray<double>>;
template class ryujin::Limiter<3, VectorizedArray<double>>;

template class ryujin::Limiter<1, VectorizedArray<float>>;
template class ryujin::Limiter<2, VectorizedArray<float>>;
template class ryujin::Limiter<3, VectorizedArray<float>>;

int main()
{
  return 0;
}

