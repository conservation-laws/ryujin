#include "initial_values.template.h"

namespace grendel
{
  template class grendel::InitialValues<1>;
  template class grendel::InitialValues<2>;
  template class grendel::InitialValues<3>;

  template class grendel::InitialValues<1, float>;
  template class grendel::InitialValues<2, float>;
  template class grendel::InitialValues<3, float>;

} /* namespace grendel */
