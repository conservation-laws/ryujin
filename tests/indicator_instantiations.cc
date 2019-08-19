#include <indicator.h>

#include <deal.II/base/vectorization.h>

using namespace grendel;
using namespace dealii;

/* Explicitly instantiate all variants: */

template class grendel::Indicator<1>;
template class grendel::Indicator<2>;
template class grendel::Indicator<3>;

template class grendel::Indicator<1, float>;
template class grendel::Indicator<2, float>;
template class grendel::Indicator<3, float>;

template class grendel::Indicator<1, VectorizedArray<double>>;
template class grendel::Indicator<2, VectorizedArray<double>>;
template class grendel::Indicator<3, VectorizedArray<double>>;

template class grendel::Indicator<1, VectorizedArray<float>>;
template class grendel::Indicator<2, VectorizedArray<float>>;
template class grendel::Indicator<3, VectorizedArray<float>>;

int main()
{
  return 0;
}

