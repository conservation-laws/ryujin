#ifndef TIME_STEP_TEMPLATE_H
#define TIME_STEP_TEMPLATE_H

#include "time_step.h"

namespace grendel
{
  using namespace dealii;


  template <int dim>
  TimeStep<dim>::TimeStep(const MPI_Comm &mpi_communicator,
                          const grendel::OfflineData<dim> &offline_data,
                          const std::string &subsection /*= "TimeStep"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , offline_data_(&offline_data)
  {
  }

} /* namespace grendel */

#endif /* TIME_STEP_TEMPLATE_H */
