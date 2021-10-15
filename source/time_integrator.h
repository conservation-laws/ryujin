//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include "dissipation_module.h"
#include "euler_module.h"
#include "offline_data.h"

namespace ryujin
{

  /**
   * TODO documentation
   */
  template <int dim, typename Number = double>
  class TimeIntegrator final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * Constructor.
     */
    TimeIntegrator(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const ryujin::OfflineData<dim, Number> &offline_data,
        const ryujin::EulerModule<dim, Number> &euler_module,
        const ryujin::DissipationModule<dim, Number> &dissipation_module,
        const std::string &subsection = "TimeIntegrator");

    /**
     * Prepare time integration. A call to @ref prepare() allocates
     * temporary storage and is necessary before any of the following
     * time-stepping functions can be called.
     */
    void prepare();

    /**
     * @name Functions for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector U perform an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * @note This function switches between different Runge-Kutta methods
     * depending on chosen runtime parameters.
     *
     * @note Depending on chosen run time parameters different CFL
     * adaptation and recovery strategies for invariant domain violations
     * are used.
     */
    Number step(vector_type &U, Number t);

  private:

    //@}
    /**
     * @name Run time options
     */
    //@{

    Number cfl_min_;
    Number cfl_max_;

    //@}

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::EulerModule<dim, Number>> euler_module_;

    dealii::SmartPointer<const ryujin::DissipationModule<dim, Number>>
        dissipation_module_;

    std::vector<SparseMatrixSIMD<Number>> temp_dij;
    std::vector<vector_type> temp_U;

    //@}
  };

} /* namespace ryujin */
