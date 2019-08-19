#include "offline_data.template.h"

namespace grendel
{
  /* instantiations */
  template class grendel::OfflineData<1>;
  template class grendel::OfflineData<2>;
  template class grendel::OfflineData<3>;

  template class grendel::OfflineData<1, float>;
  template class grendel::OfflineData<2, float>;
  template class grendel::OfflineData<3, float>;

} /* namespace grendel */
