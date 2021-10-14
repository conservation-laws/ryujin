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

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

    /* SSP-RK3 */

    // FIXME
    temp_U.resize(3);
    temp_dij.resize(3);
    for(unsigned int n = 0; n < 3; ++n) {
      temp_U[n].reinit(vector_partitioner);
      temp_dij[n].reinit(sparsity_simd);
    }

  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step(vector_type &U, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step()" << std::endl;
#endif

    /* Forward Euler step: */

#if 0
    Number tau =
        euler_module_->template step<0>(U, {}, {}, {}, my_U, my_dij);
    euler_module_->apply_boundary_conditions(my_U, t + tau);
    U.swap(my_U);
    return tau;
#endif

    /* SSP-RK3, see @cite Shu1988, Eq. 2.18. */
    {
      /* Step 1: U1 = U_old + tau * L(U_old) at time t + tau */
      Number tau = euler_module_->template step<0>(
          U, {}, {}, {}, temp_U[0], temp_dij[0]);
      euler_module_->apply_boundary_conditions(temp_U[0], t + tau);

      /* Step 2: U2 = 3/4 U_old + 1/4 (U1 + tau L(U1)) at time t + tau */
      euler_module_->template step<0>(
          temp_U[0], {}, {}, {}, temp_U[1], temp_dij[1]);
      temp_U[1].sadd(Number(1. / 4.), Number(3. / 4.), U);
      euler_module_->apply_boundary_conditions(temp_U[1], t + tau);

      /* Step 3: U3 = 1/3 U_old + 2/3 (U2 + tau L(U2)) at time t + 0.5 * tau */
      euler_module_->template step<0>(
          temp_U[1], {}, {}, {}, temp_U[2], temp_dij[2]);
      temp_U[2].sadd(Number(2. / 3.), Number(1. / 3.), U);
      euler_module_->apply_boundary_conditions(temp_U[2], t + 0.5 * tau);

      U.swap(temp_U[2]);
      return tau;
    }
  }
} /* namespace ryujin */
