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
     * TODO documentation
     */
    void prepare();

    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * TODO documentation
     */
    Number step(vector_type &U, Number t, Number tau = 0.);

  private:

    //@}
    /**
     * @name Run time options
     */
    //@{

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

    // FIXME: remove
    mutable SparseMatrixSIMD<Number> my_dij;
    mutable vector_type my_U;

    //@}
  };

} /* namespace ryujin */
