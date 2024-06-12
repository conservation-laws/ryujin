//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "mesh_adaptor.h"

namespace ryujin
{
  template <typename Description, int dim, typename Number>
  MeshAdaptor<Description, dim, Number>::MeshAdaptor(
      const MPI_Comm &mpi_communicator,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const std::string &subsection /*= "MeshAdaptor"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , offline_data_(&offline_data)
      , need_mesh_adaptation_(false)
  {
    adaptation_strategy_ = AdaptationStrategy::global_refinement;
    add_parameter(
        "adaptation strategy",
        adaptation_strategy_,
        "The chosen adaptation strategy. Possible values are: \"global\"");

    t_global_refinements_ = {};
    add_parameter("global refinement timepoints",
                  t_global_refinements_,
                  "List of points in (simulation) time at which the mesh will "
                  "be globally refined. Used only for the \"global "
                  "refinement\" adaptation strategy.");
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::prepare(const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MeshAdaptor<dim, Number>::prepare()" << std::endl;
#endif

    /* Remove outdated refinement timestamps: */
    const auto new_end = std::remove_if(
        t_global_refinements_.begin(),
        t_global_refinements_.end(),
        [&](const Number &t_refinement) { return (t >= t_refinement); });
    t_global_refinements_.erase(new_end, t_global_refinements_.end());
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::analyze(
      const StateVector & /*state_vector*/,
      const Number t,
      unsigned int /*cycle*/)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MeshAdaptor<dim, Number>::analyze()" << std::endl;
#endif

    if (adaptation_strategy_ == AdaptationStrategy::global_refinement) {

      /* Remove all refinement points from the vector that lie in the past: */
      const auto new_end = std::remove_if( //
          t_global_refinements_.begin(),
          t_global_refinements_.end(),
          [&](const Number &t_refinement) {
            if (t < t_refinement)
              return false;
            need_mesh_adaptation_ = true;
            return true;
          });
      t_global_refinements_.erase(new_end, t_global_refinements_.end());

      return;
    }

    __builtin_unreachable();
  }


} // namespace ryujin
