#include "schlieren_postprocessor.template.h"

namespace grendel
{
  /* instantiations */
  template class grendel::SchlierenPostprocessor<1>;
  template class grendel::SchlierenPostprocessor<2>;
  template class grendel::SchlierenPostprocessor<3>;

} /* namespace grendel */
