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
      const vector_type &old_U,
      const Number old_t,
      std::array<std::reference_wrapper<const vector_type>, stages> /*stage_U*/,
      const std::array<Number, stages> /*stage_weights*/,
      vector_type &new_U,
      Number tau) const
  {
    if constexpr (ParabolicSystem::is_identity) {
      AssertThrow(
          false,
          dealii::ExcMessage("The parabolic system is the identity. This "
                             "function should have never been called."));
      __builtin_trap();

    } else {

      static_assert(stages == 0, "high order fluxes are not implemented");

      /* FIXME: This needs to be refactored really really badly. */

      const bool reinit_gmg = cycle_++ % 4 == 0;
      parabolic_solver_.backward_euler_step(
          old_U, old_t, new_U, tau, id_violation_strategy_, reinit_gmg);
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
