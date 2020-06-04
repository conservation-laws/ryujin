#include <sparse_matrix_simd.template.h>
#include <offline_data.template.h>

using namespace ryujin;
using namespace dealii;

/* Explicitly instantiate all variants: */

template class ryujin::OfflineData<1>;
template class ryujin::OfflineData<2>;
template class ryujin::OfflineData<3>;

template class ryujin::OfflineData<1, float>;
template class ryujin::OfflineData<2, float>;
template class ryujin::OfflineData<3, float>;

int main()
{
  return 0;
}

