//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
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
      const vector_type &,
      const Number,
      std::array<std::reference_wrapper<const vector_type>, stages>,
      const std::array<Number, stages>,
      vector_type &,
      const Number) const
  {
    if constexpr (ParabolicSystem::is_identity) {
      AssertThrow(
          false,
          dealii::ExcMessage("The parabolic system is the identity. This "
                             "function should have never been called."));
      __builtin_trap();

    } else {

      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }
  }


  template <typename Description, int dim, typename Number>
  void ParabolicModule<Description, dim, Number>::crank_nicolson_step(
      const vector_type &old_U,
      const Number t,
      vector_type &new_U,
      const Number tau) const
  {
    if constexpr (ParabolicSystem::is_identity) {
      AssertThrow(
          false,
          dealii::ExcMessage("The parabolic system is the identity. This "
                             "function should have never been called."));
      __builtin_trap();

    } else {

      parabolic_solver_.crank_nicolson_step(old_U, t, new_U, tau, cycle_);
    }
  }

} /* namespace ryujin */
