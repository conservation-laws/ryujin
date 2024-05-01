//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "offline_data.h"

#include <deal.II/distributed/solution_transfer.h>

namespace ryujin
{

  /**
   * A solution transfer class that interpolates a given state U (in
   * conserved variables) to a refined/coarsened mesh. The class first
   * transforms the conserved state into primitive variables and then
   * interpolates/restricts the primitive state field via deal.II's
   * SolutionTransfer mechanism.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class SolutionTransfer final
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = Description::HyperbolicSystem;

    using View = Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using StateVector = View::StateVector;

    using ScalarVector = Vectors::ScalarVector<Number>;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    SolutionTransfer(const OfflineData<dim, Number> &offline_data,
                     const HyperbolicSystem &hyperbolic_system)
        : offline_data_(&offline_data)
        , hyperbolic_system_(hyperbolic_system)
        , solution_transfer_(offline_data.dof_handler())
    {
    }

    /**
     * Read in a state vector (in conserved quantities). The function
     * populates an auxiliary distributed vectors that store the given
     * state in primitive variables and then calls the underlying deal.II
     * SolutionTransfer::prepare_for_coarsening_and_refinement();
     *
     * This function has to be called before the actual grid refinement.
     */
    void prepare_for_interpolation(const StateVector &state_vector)
    {
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
      solution_transfer_.prepare_for_coarsening_and_refinement(ptr_state);
    }

    /**
     * Finalize the state vector transfer by calling
     * SolutionTransfer::interpolate() and repopulating the state vector
     * (in conserved quantities).
     *
     * This function has to be called after the actual grid refinement.
     */
    void interpolate(StateVector &state_vector)
    {
      auto &U = std::get<0>(state_vector);

      const auto &scalar_partitioner = offline_data_->scalar_partitioner();

      U.reinit(offline_data_->vector_partitioner());

      interpolated_state_.resize(problem_dimension);
      for (auto &it : interpolated_state_) {
        it.reinit(scalar_partitioner);
        it.zero_out_ghost_values();
      }

      std::vector<ScalarVector *> ptr_interpolated_state;
      std::transform(interpolated_state_.begin(),
                     interpolated_state_.end(),
                     std::back_inserter(ptr_interpolated_state),
                     [](auto &it) { return &it; });
      solution_transfer_.interpolate(ptr_interpolated_state);

      /*
       * Read back from interpolated_state_ vector:
       */

      for (unsigned int k = 0; k < problem_dimension; ++k) {
        U.insert_component(interpolated_state_[k], k);
      }
      U.update_ghost_values();
    }

  private:
    //@}
    /**
     * @name Internal data
     */
    //@{
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    const HyperbolicSystem &hyperbolic_system_;

    dealii::parallel::distributed::SolutionTransfer<dim, ScalarVector>
        solution_transfer_;

    std::vector<ScalarVector> state_;
    std::vector<ScalarVector> interpolated_state_;
    //@}
  };

} /* namespace ryujin */
