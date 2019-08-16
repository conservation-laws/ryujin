#include <offline_data.template.h>

using namespace grendel;
using namespace dealii;

/* Explicitly instantiate all variants: */

template class grendel::OfflineData<1>;
template class grendel::OfflineData<2>;
template class grendel::OfflineData<3>;

template class grendel::OfflineData<1, float>;
template class grendel::OfflineData<2, float>;
template class grendel::OfflineData<3, float>;

int main()
{
  return 0;
}

