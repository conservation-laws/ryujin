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
    cfl_min_ = Number(0.45);
    add_parameter(
        "cfl min",
        cfl_min_,
        "Minimal admissible relative CFL constant. How this parameter is used "
        "depends on the chosen CFL recovery strategy");

    cfl_max_ = Number(0.90);
    add_parameter(
        "cfl max",
        cfl_max_,
        "Maximal admissible relative CFL constant. How this parameter is used "
        "depends on the chosen CFL recovery strategy");

    cfl_recovery_strategy_ = CFLRecoveryStrategy::bang_bang_control;
    add_parameter("cfl recovery strategy",
                  cfl_recovery_strategy_,
                  "CFL/invariant domain violation recovery strategy: none, "
                  "bang bang control");

    time_stepping_scheme_ = TimeSteppingScheme::erk_33;
    add_parameter("time stepping scheme",
                  time_stepping_scheme_,
                  "Time stepping scheme: ssprk 33, erk 33");
  }


  template <int dim, typename Number>
  void TimeIntegrator<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::prepare()" << std::endl;
#endif

    /* Resize temporary storage to appropriate sizes: */

    /* SSP-RK3 */
    // FIXME
    temp_U.resize(3);
    temp_dij.resize(2);

    /* Initialize temporary vectors and matrices: */

    const auto &vector_partitioner = offline_data_->vector_partitioner();
    for (auto &it : temp_U)
      it.reinit(vector_partitioner);

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    for (auto &it : temp_dij)
      it.reinit(sparsity_simd);

    /* Reset CFL to canonical starting value: */

    AssertThrow(cfl_min_ > 0., ExcMessage("cfl min must be a positive value"));
    AssertThrow(cfl_max_ >= cfl_min_,
                ExcMessage("cfl max must be greater or equal than cfl min"));

    euler_module_->cfl(cfl_max_);
  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step(vector_type &U, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step()" << std::endl;
#endif

    /* Forward Euler step: */

#if 0
    {
      Number tau =
          euler_module_->template step<0>(U, {}, {}, {}, temp_U[0], temp_dij[0]);
      euler_module_->apply_boundary_conditions(temp_U[0], t + tau);
      U.swap(temp_U[0]);
      return tau;
    }
#endif

#if 0
    {
      /* SSP-RK3, see @cite Shu1988, Eq. 2.18. */

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
#endif

    {
      /* RK(3,3,1) */

      /* Step 1: U1 <- {U, 1} at time t + tau */

      Number tau = euler_module_->template step<0, true>(
          U, {}, {}, {}, temp_U[0], temp_dij[0]);
      euler_module_->apply_boundary_conditions(temp_U[0], t + tau);

      /* Step 2: U2 <- {U1, 2} and {U, -1} at time t + 2 tau */

      euler_module_->template step<1, true>(temp_U[0],
                                            {U},
                                            {temp_dij[0]},
                                            {Number(-1.)},
                                            temp_U[1],
                                            temp_dij[1],
                                            tau);
      euler_module_->apply_boundary_conditions(temp_U[1], t + 2. * tau);

      /* Step 3: U3 <- {U2, 9/4} and {U1, -2} and {U, 3/4} at time t + 3 tau */

      euler_module_->template step<2, false>(temp_U[1],
                                             {U, temp_U[0]},
                                             {temp_dij[0], temp_dij[1]},
                                             {Number(0.75), Number(-2.)},
                                             temp_U[2],
                                             temp_dij[1],
                                             tau);
      euler_module_->apply_boundary_conditions(temp_U[2], t + 3. * tau);

      U.swap(temp_U[2]);
      return 3. * tau;
    }
  }
} /* namespace ryujin */
