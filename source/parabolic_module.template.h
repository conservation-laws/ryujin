//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include "parabolic_module.h"

namespace ryujin
{
  using namespace dealii;

  template <typename Description, int dim, typename Number>
  ParabolicModule<Description, dim, Number>::ParabolicModule(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const InitialValues<Description, dim, Number> &initial_values,
      const std::string &subsection /*= "ParabolicModule"*/)
      : ParameterAcceptor(subsection)
      , id_violation_strategy_(IDViolationStrategy::warn)
      , parabolic_solver_(mpi_communicator,
                          computing_timer,
                          hyperbolic_system,
                          parabolic_system,
                          offline_data,
                          initial_values,
                          subsection)
      , n_restarts_(0)
      , n_warnings_(0)
  {
  }


  template <typename Description, int dim, typename Number>
  void ParabolicModule<Description, dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "ParabolicModule<Description, dim, Number>::prepare()"
              << std::endl;
#endif
    if constexpr (!ParabolicSystem::is_identity)
      parabolic_solver_.prepare();

    cycle_ = 0;
  }


  template <typename Description, int dim, typename Number>
  template <int stages>
  void ParabolicModule<Description, dim, Number>::step(
      const StateVector &old_state_vector,
      const Number old_t,
      std::array<std::reference_wrapper<const StateVector>,
                 stages> /*stage_state_vectors*/,
      const std::array<Number, stages> /*stage_weights*/,
      StateVector &new_state_vector,
      Number tau) const
  {
    if constexpr (ParabolicSystem::is_identity) {
      AssertThrow(
          false,
          dealii::ExcMessage("The parabolic system is the identity. This "
                             "function should have never been called."));
      __builtin_trap();

    } else {

      AssertThrow(stages == 0,
                  dealii::ExcMessage("Although IMEX schemes are implemented, "
                                     "the high order fluxes are not. "));

      const bool reinit_gmg = cycle_++ % 4 == 0;
      parabolic_solver_.backward_euler_step(old_state_vector,
                                            old_t,
                                            new_state_vector,
                                            tau,
                                            id_violation_strategy_,
                                            reinit_gmg);
      n_restarts_ = parabolic_solver_.n_restarts();
      n_warnings_ = parabolic_solver_.n_warnings();
    }
  }


  template <typename Description, int dim, typename Number>
  void ParabolicModule<Description, dim, Number>::print_solver_statistics(
      std::ostream &output) const
  {
    if constexpr (!ParabolicSystem::is_identity) {
      parabolic_solver_.print_solver_statistics(output);
    }
  }

} /* namespace ryujin */
