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
    add_parameter("adaptation strategy",
                  adaptation_strategy_,
                  "The chosen adaptation strategy. Possible values are: global "
                  "refinement, random adaptation");

    marking_strategy_ = MarkingStrategy::fixed_number;
    add_parameter(
        "marking strategy",
        marking_strategy_,
        "The chosen marking strategy. Possible values are: fixed number");

    time_point_selection_strategy_ =
        TimePointSelectionStrategy::fixed_adaptation_time_points;
    add_parameter("time point selection strategy",
                  time_point_selection_strategy_,
                  "The chosen time point selection strategy. Possible values "
                  "are: fixed adaptation time points");

    /* Options for various marking strategies: */

    enter_subsection("marking strategy");
    fixed_number_refinement_fraction_ = 0.3;
    add_parameter(
        "fixed number: refinement fraction",
        fixed_number_refinement_fraction_,
        "Fixed number strategy: fraction of cells selected for refinement.");

    fixed_number_coarsening_fraction_ = 0.3;
    add_parameter(
        "fixed number: coarsening fraction",
        fixed_number_coarsening_fraction_,
        "Fixed number strategy: fraction of cells selected for coarsening.");
    leave_subsection();

    /* Options for various time point selection strategies: */

    enter_subsection("time point selection strategy");
    adaptation_time_points_ = {};
    add_parameter("adaptation timepoints",
                  adaptation_time_points_,
                  "List of time points in (simulation) time at which we will "
                  "perform a mesh adaptation cycle.");
    leave_subsection();
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::prepare(const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MeshAdaptor<dim, Number>::prepare()" << std::endl;
#endif

    switch (time_point_selection_strategy_) {
    case TimePointSelectionStrategy::fixed_adaptation_time_points: {
      /* Remove outdated refinement timestamps: */
      const auto new_end = std::remove_if(
          adaptation_time_points_.begin(),
          adaptation_time_points_.end(),
          [&](const Number &t_refinement) { return (t > t_refinement); });
      adaptation_time_points_.erase(new_end, adaptation_time_points_.end());
    } break;

    default:
      // do nothing
      break;
    }

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

    switch (time_point_selection_strategy_) {
    case TimePointSelectionStrategy::fixed_adaptation_time_points: {
      /* Remove all refinement points from the vector that lie in the past: */
      const auto new_end = std::remove_if( //
          adaptation_time_points_.begin(),
          adaptation_time_points_.end(),
          [&](const Number &t_refinement) {
            if (t < t_refinement)
              return false;
            need_mesh_adaptation_ = true;
            return true;
          });
      adaptation_time_points_.erase(new_end, adaptation_time_points_.end());
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::
      mark_cells_for_coarsening_and_refinement(
          dealii::Triangulation<dim> &triangulation) const
  {
    auto &discretization [[maybe_unused]] = offline_data_->discretization();
    Assert(&triangulation == &discretization.triangulation(),
           dealii::ExcInternalError());

    switch (adaptation_strategy_) {
    case AdaptationStrategy::global_refinement: {
      /* Simply mark all cells for refinement and return: */
      for (auto &cell : triangulation.active_cell_iterators())
        cell->set_refine_flag();
      return;
    } break;

    case AdaptationStrategy::random_adaptation: {
      return;
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }
  }
} // namespace ryujin
