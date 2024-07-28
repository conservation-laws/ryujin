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


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::prepare_for_interpolation(
      const StateVector &state_vector) const
  {
    /* Set up dealii::<...>::SolutionTransfer object: */
    solution_transfer_ = std::make_unique<
        dealii::parallel::distributed::SolutionTransfer<dim, ScalarVector>>(
        offline_data_->dof_handler());

    const auto &U = std::get<0>(state_vector);

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    const auto &affine_constraints = offline_data_->affine_constraints();

    state_.resize(problem_dimension);
    for (auto &it : state_)
      it.reinit(scalar_partitioner);

    /*
     * We need to copy over to an auxiliary state vector formed by a
     * ScalarVector for each component because dealii::SolutionTransfer
     * cannot work on our StateVector or MultiComponentVector
     */

    for (unsigned int k = 0; k < problem_dimension; ++k) {
      U.extract_component(state_[k], k);
      affine_constraints.distribute(state_[k]);
      state_[k].update_ghost_values();
    }

    std::vector<const ScalarVector *> ptr_state;
    std::transform(state_.begin(),
                   state_.end(),
                   std::back_inserter(ptr_state),
                   [](auto &it) { return &it; });
    solution_transfer_->prepare_for_coarsening_and_refinement(ptr_state);
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::interpolate(
      StateVector &state_vector) const
  {
    Vectors::reinit_state_vector<Description>(state_vector, *offline_data_);
    auto &U = std::get<0>(state_vector);

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();

    std::vector<ScalarVector> interpolated_state;
    interpolated_state.resize(problem_dimension);
    for (auto &it : interpolated_state) {
      it.reinit(scalar_partitioner);
      it.zero_out_ghost_values();
    }

    std::vector<ScalarVector *> ptr_interpolated_state;
    std::transform(interpolated_state.begin(),
                   interpolated_state.end(),
                   std::back_inserter(ptr_interpolated_state),
                   [](auto &it) { return &it; });
    solution_transfer_->interpolate(ptr_interpolated_state);

    /*
     * Read back from interpolated_state_ vector:
     */

    for (unsigned int k = 0; k < problem_dimension; ++k) {
      U.insert_component(interpolated_state[k], k);
    }
    U.update_ghost_values();

    /* Free up some space and delete outdated state vector and transfer. */
    state_.clear();
    solution_transfer_.reset();
  }
} // namespace ryujin
