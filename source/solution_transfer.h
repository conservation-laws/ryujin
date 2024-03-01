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
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * @copydoc HyperbolicSystemView
     */
    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension = View::problem_dimension;

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = typename View::state_type;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * Typedef for a MultiComponentVector storing the state U.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

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
    void prepare_for_interpolation(const vector_type &U)
    {
      const auto &scalar_partitioner = offline_data_->scalar_partitioner();
      const auto &affine_constraints = offline_data_->affine_constraints();

      state_.resize(problem_dimension);
      for (auto &it : state_)
        it.reinit(scalar_partitioner);

      /*
       * FIXME: we need to copy over to an auxiliary  state_ vector because
       * dealii::SolutionTransfer cannot work on our MultiComponentVector
       */

      for (unsigned int k = 0; k < problem_dimension; ++k) {
        U.extract_component(state_[k], k);
        affine_constraints.distribute(state_[k]);
        state_[k].update_ghost_values();
      }

      std::vector<const scalar_type *> ptr_state;
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
    void interpolate(vector_type &U)
    {
      const auto &scalar_partitioner = offline_data_->scalar_partitioner();

      U.reinit(offline_data_->vector_partitioner());

      interpolated_state_.resize(problem_dimension);
      for (auto &it : interpolated_state_) {
        it.reinit(scalar_partitioner);
        it.zero_out_ghost_values();
      }

      std::vector<scalar_type *> ptr_interpolated_state;
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

    dealii::parallel::distributed::SolutionTransfer<dim, scalar_type>
        solution_transfer_;

    std::vector<scalar_type> state_;
    std::vector<scalar_type> interpolated_state_;
    //@}
  };

} /* namespace ryujin */
