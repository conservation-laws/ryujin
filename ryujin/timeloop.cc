#include "timeloop.template.h"

namespace ryujin
{
  /* instantiations */
  template class TimeLoop<1>;
  template class TimeLoop<2>;
  template class TimeLoop<3>;

  template class TimeLoop<1, float>;
  template class TimeLoop<2, float>;
  template class TimeLoop<3, float>;
} // namespace ryujin
