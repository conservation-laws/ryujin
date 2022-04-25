//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <hyperbolic_system.h>

#include <deal.II/base/timer.h>

#include "initial_values.h"
#include "offline_data.h"

namespace ryujin
{
  template <int dim, typename Number = double>
  class DissipationModule final : public dealii::ParameterAcceptor
  {
  public:

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * Constructor.
     */
    DissipationModule(
        const MPI_Comm & /*mpi_communicator*/,
        std::map<std::string, dealii::Timer> & /*computing_timer*/,
        const HyperbolicSystem & /*hyperbolic_system*/,
        const OfflineData<dim, Number> & /*offline_data*/,
        const InitialValues<dim, Number> & /*initial_values*/,
        const std::string & /*subsection*/ = "DissipationModule")
    {
      // do nothing
    }

    void prepare()
    {
      // do nothing
    }


    Number step(vector_type & /*U*/,
                Number /*t*/,
                Number tau,
                unsigned int /*cycle*/) const
    {
      // do nothing
      return tau;
    }

  private:
    mutable unsigned int n_warnings_;
    ACCESSOR_READ_ONLY(n_warnings)

    mutable double n_iterations_velocity_;
    ACCESSOR_READ_ONLY(n_iterations_velocity)

    mutable double n_iterations_internal_energy_;
    ACCESSOR_READ_ONLY(n_iterations_internal_energy)

  };

} // namespace ryujin
