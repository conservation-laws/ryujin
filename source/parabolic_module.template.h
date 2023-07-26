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
      const ParabolicSystem &parabolic_system,
      const InitialValues<Description, dim, Number> &initial_values,
      const std::string &subsection /*= "ParabolicModule"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , parabolic_system_(&parabolic_system)
      , initial_values_(&initial_values)
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
  }


  template <typename Description, int dim, typename Number>
  template <int stages>
  void ParabolicModule<Description, dim, Number>::step(
      const vector_type &,
      const precomputed_vector_type &,
      std::array<std::reference_wrapper<const vector_type>, stages>,
      std::array<std::reference_wrapper<const precomputed_vector_type>, stages>,
      const std::array<Number, stages>,
      vector_type &,
      Number) const
  {
    if constexpr (ParabolicSystem::is_identity) {
      AssertThrow(
          false,
          dealii::ExcMessage("The parabolic system is the identity. This "
                             "function should have never been called."));
      __builtin_trap();

    } else {

      // fixme
    }
  }


  template <typename Description, int dim, typename Number>
  void ParabolicModule<Description, dim, Number>::crank_nicolson_step(
      const vector_type &,
      const precomputed_vector_type &,
      vector_type &,
      Number) const
  {
    if constexpr (ParabolicSystem::is_identity) {
      AssertThrow(
          false,
          dealii::ExcMessage("The parabolic system is the identity. This "
                             "function should have never been called."));
      __builtin_trap();

    } else {

      // fixme
    }
  }

} /* namespace ryujin */
