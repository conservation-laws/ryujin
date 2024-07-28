//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "mesh_adaptor.h"

#include <deal.II/grid/grid_refinement.h>

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
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
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
        [&](const Number &t_refinement) { return (t > t_refinement); });
    t_global_refinements_.erase(new_end, t_global_refinements_.end());

    // Do not reset state_ and solution_transfer_ objects as they are
    // needed for the subsequent solution transfer.

    /* toggle mesh adaptation flag to off. */
    need_mesh_adaptation_ = false;
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


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::
      mark_cells_for_coarsening_and_refinement(
          dealii::Triangulation<dim> &triangulation) const
  {
    auto &discretization [[maybe_unused]] = offline_data_->discretization();
    Assert(&triangulation == &discretization.triangulation(),
           dealii::ExcInternalError());

    if (adaptation_strategy_ == AdaptationStrategy::global_refinement) {
      for (auto &cell : triangulation.active_cell_iterators())
        cell->set_refine_flag();

      return;
    }

    __builtin_unreachable();
  }
} // namespace ryujin
