#include "time_step.template.h"

namespace grendel
{
  /* instantiations */
  template class grendel::TimeStep<1>;
  template class grendel::TimeStep<2>;
  template class grendel::TimeStep<3>;

  template class grendel::TimeStep<1, float>;
  template class grendel::TimeStep<2, float>;
  template class grendel::TimeStep<3, float>;

} /* namespace grendel */
