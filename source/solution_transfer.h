//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/distributed/solution_transfer.h>

namespace ryujin
{

  /**
   * @todo Documentation
   *
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class SolutionTransfer final
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc ProblemDescription::state_type
     */
    using state_type = ProblemDescription::state_type<dim, Number>;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * Constructor.
     */
    SolutionTransfer(const ryujin::OfflineData<dim, Number> &offline_data,
                     const ryujin::ProblemDescription &problem_description)
        : offline_data_(&offline_data)
        , problem_description_(&problem_description)
        , solution_transfer_(offline_data.dof_handler())
    {
    }

    /**
     * @todo Documentation
     */
    void prepare_for_interpolation(const vector_type &U)
    {
      const auto &scalar_partitioner = offline_data_->scalar_partitioner();
      const auto &affine_constraints = offline_data_->affine_constraints();

      state_.resize(problem_dimension);
      for (auto &it : state_)
        it.reinit(scalar_partitioner);

      const unsigned int n_owned = offline_data_->n_locally_owned();

      /* copy over density, velocity and internal energy: */

      for(unsigned int i = 0; i < n_owned; ++i) {
        const auto U_i = U.get_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto v_i = problem_description_->momentum(U_i) / rho_i;
        const auto e_i = problem_description_->internal_energy(U_i) / rho_i;

        state_[0].local_element(i) = rho_i;
        for (unsigned int d = 0; d < dim; ++d)
          state_[1 + d].local_element(i) = v_i[d];
        state_[1 + dim].local_element(i) = e_i;
      }

      for (auto &it : state_) {
        affine_constraints.distribute(it);
        it.update_ghost_values();
      }

      std::vector<const scalar_type *> ptr_state;
      std::transform(state_.begin(),
                     state_.end(),
                     std::back_inserter(ptr_state),
                     [](auto &it) { return &it; });
      solution_transfer_.prepare_for_coarsening_and_refinement(ptr_state);
    }

    /**
     * @todo Documentation
     */
    void interpolate(vector_type &U)
    {
      const auto &scalar_partitioner = offline_data_->scalar_partitioner();

      U.reinit(offline_data_->vector_partitioner());

      interpolated_state_.resize(problem_dimension);
      for (auto &it : interpolated_state_) {
        it.reinit(scalar_partitioner);
        it.zero_out_ghosts();
      }

      std::vector<scalar_type *> ptr_interpolated_state;
      std::transform(interpolated_state_.begin(),
                     interpolated_state_.end(),
                     std::back_inserter(ptr_interpolated_state),
                     [](auto &it) { return &it; });
      solution_transfer_.interpolate(ptr_interpolated_state);

      const unsigned int n_owned = offline_data_->n_locally_owned();

      /* copy over density, velocity and internal energy: */

      for(unsigned int i = 0; i < n_owned; ++i) {

        state_type U_i;

        const auto rho_i = interpolated_state_[0].local_element(i);
        auto E_i = rho_i * interpolated_state_[1 + dim].local_element(i);

        U_i[0] = rho_i;
        for(unsigned int d = 0; d < dim; ++d) {
          const auto v_i = interpolated_state_[1 + d].local_element(i);
          U_i[1 + d] = rho_i * v_i;
          E_i += 0.5 * rho_i * v_i * v_i;
        }

        U_i[1 + dim] = E_i;

        U.write_tensor(U_i, i);
      }

      U.update_ghost_values();
    }

  private:
    //@}
    /**
     * @name Internal data
     */
    //@{
    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::ProblemDescription> problem_description_;

    dealii::parallel::distributed::SolutionTransfer<dim, scalar_type>
        solution_transfer_;

    std::vector<scalar_type> state_;
    std::vector<scalar_type> interpolated_state_;
    //@}
  };

} /* namespace ryujin */
