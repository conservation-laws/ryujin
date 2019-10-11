#include "postprocessor.template.h"

namespace grendel
{
  /* instantiations */
  template class grendel::Postprocessor<1>;
  template class grendel::Postprocessor<2>;
  template class grendel::Postprocessor<3>;

  template class grendel::Postprocessor<1, float>;
  template class grendel::Postprocessor<2, float>;
  template class grendel::Postprocessor<3, float>;

} /* namespace grendel */
