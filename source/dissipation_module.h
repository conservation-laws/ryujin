//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <hyperbolic_system.h>
#include <parabolic_system.h>

#include "initial_values.h"
#include "offline_data.h"

#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ryujin
{
  /**
   * Implicit theta time-stepping for parabolic systems.
   *
   * @ingroup DissipationModule
   */
  template <int dim, typename Number = double>
  class DissipationModule final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension =
        HyperbolicSystem::problem_dimension<dim>;

    /**
     * Typedef for a MultiComponentVector storing the state U.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * @copydoc Parabolic_system::parabolic_problem_dimension
     */
    static constexpr unsigned int parabolic_problem_dimension =
        ParabolicSystem::parabolic_problem_dimension<dim>;

    /**
     * @copydoc Parabolic_system::parabolic_state_type
     */
    using parabolic_state_type =
        ParabolicSystem::parabolic_state_type<dim, Number>;

    /**
     * @copydoc ParabolicSystem::n_implicit_systems
     */
    static constexpr unsigned int n_implicit_systems =
        ParabolicSystem::n_implicit_systems;

    /**
     * A distributed block vector used for temporary storage of the
     * velocity field.
     */
    using block_vector_type =
        dealii::LinearAlgebra::distributed::BlockVector<Number>;

    /**
     * Constructor.
     */
    DissipationModule(const MPI_Comm &mpi_communicator,
                      std::map<std::string, dealii::Timer> &computing_timer,
                      const ParabolicSystem &parabolic_system,
                      const OfflineData<dim, Number> &offline_data,
                      const InitialValues<dim, Number> &initial_values,
                      const std::string &subsection = "DissipationModule");

    void prepare();

    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector U perform an implicit
     * update of the dissipative parabolic limiting problem and store the
     * result again in U. The function
     *
     *  - returns the chosen time step size tau (for compatibility with the
     *    EulerModule::euler_step() and Euler_Module::step() interfaces).
     *
     *  - performs a time step and populates the vector U by the result (in
     *    place).
     */
    Number step(vector_type &U, Number t, Number tau, unsigned int cycle) const;

  private:
    //@}
    /**
     * @name Private methods for step()
     */
    //@{

    void enforce_boundary_values(Number t) const;


    /**
     * @name Run time options
     */
    //@{

    Number tolerance_;
    bool tolerance_linfty_norm_;

    std::array<bool, n_implicit_systems> use_gmg_;
    ACCESSOR_READ_ONLY(use_gmg)

    std::array<Number, n_implicit_systems> gmg_max_iter_;
    std::array<Number, n_implicit_systems> gmg_smoother_range_;
    std::array<Number, n_implicit_systems> gmg_smoother_max_eig_;
    std::array<unsigned int, n_implicit_systems> gmg_smoother_degree_;
    std::array<unsigned int, n_implicit_systems> gmg_smoother_n_cg_iter_;
    std::array<unsigned int, n_implicit_systems> gmg_min_level_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const ParabolicSystem> parabolic_system_;
    ACCESSOR_READ_ONLY(parabolic_system)
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const InitialValues<dim, Number>> initial_values_;

    mutable unsigned int n_warnings_;
    ACCESSOR_READ_ONLY(n_warnings)

    mutable std::array<double, n_implicit_systems> n_iterations_;
    ACCESSOR_READ_ONLY(n_iterations)

    mutable dealii::MatrixFree<dim, Number> matrix_free_;

    mutable block_vector_type solution_;
    mutable block_vector_type right_hand_side_;

    mutable Number theta_;
    //@}
  };

} // namespace ryujin
