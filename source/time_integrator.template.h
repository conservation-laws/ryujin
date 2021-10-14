//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "time_integrator.h"

namespace ryujin
{
  using namespace dealii;

  template <int dim, typename Number>
  TimeIntegrator<dim, Number>::TimeIntegrator(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const ryujin::EulerModule<dim, Number> &euler_module,
      const ryujin::DissipationModule<dim, Number> &dissipation_module,
      const std::string &subsection /*= "TimeIntegrator"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , euler_module_(&euler_module)
      , dissipation_module_(&dissipation_module)
  {
  }


  template <int dim, typename Number>
  void TimeIntegrator<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::prepare()" << std::endl;
#endif
  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step(vector_type &U,
                                           Number t,
                                           Number tau_0 /*= 0*/)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step()" << std::endl;
#endif

    return euler_module_->step(U, t, tau_0);
  }

} /* namespace ryujin */
