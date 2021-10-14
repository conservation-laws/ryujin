//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "time_integrator.h"

#include "euler_module.template.h"

namespace ryujin
{
  using namespace dealii;

  template <int dim, typename Number>
  TimeIntegrator<dim, Number>::TimeIntegrator(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const ryujin::EulerModule<dim, Number> &euler_module,
      const ryujin::DissipationModule<dim, Number> &dissipation_module,
      const std::string &subsection /*= "TimeIntegrator"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
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

    const auto &vector_partitioner = offline_data_->vector_partitioner();
    my_U.reinit(vector_partitioner);

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    my_dij.reinit(sparsity_simd);
  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step(vector_type &U,
                                           Number t,
                                           Number tau_0 /*= 0*/)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step()" << std::endl;
#endif

    Number tau_1 =
        euler_module_->template step<0>(U, {}, {}, {}, my_U, my_dij, tau_0);
    euler_module_->apply_boundary_conditions(my_U, t + tau_1);
    U.swap(my_U);
    return tau_1;
  }

} /* namespace ryujin */
