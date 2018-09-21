#ifndef TIME_STEP_H
#define TIME_STEP_H

#include "boilerplate.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>

namespace grendel
{

  template <int dim>
  class TimeStep : public dealii::ParameterAcceptor
  {
  public:
    TimeStep(const MPI_Comm &mpi_communicator,
             const grendel::OfflineData<dim> &offline_data,
             const std::string &subsection = "TimeStep");

    virtual ~TimeStep() final = default;

  protected:

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const grendel::OfflineData<dim>> offline_data_;
    A_RO(offline_data)
  };

} /* namespace grendel */

#endif /* TIME_STEP_H */
