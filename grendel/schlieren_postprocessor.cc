#include "schlieren_postprocessor.template.h"

namespace grendel
{
  /* instantiations */
  template class grendel::SchlierenPostprocessor<1>;
  template class grendel::SchlierenPostprocessor<2>;
  template class grendel::SchlierenPostprocessor<3>;

  template class grendel::SchlierenPostprocessor<1, float>;
  template class grendel::SchlierenPostprocessor<2, float>;
  template class grendel::SchlierenPostprocessor<3, float>;

} /* namespace grendel */
