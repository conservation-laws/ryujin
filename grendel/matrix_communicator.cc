#include "matrix_communicator.template.h"

namespace grendel
{
  /* instantiations */
  template class grendel::MatrixCommunicator<1>;
  template class grendel::MatrixCommunicator<2>;
  template class grendel::MatrixCommunicator<3>;

  template class grendel::MatrixCommunicator<1, float>;
  template class grendel::MatrixCommunicator<2, float>;
  template class grendel::MatrixCommunicator<3, float>;

} /* namespace grendel */
