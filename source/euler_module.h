//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "simd.h"

#include "limiter.h"

#include "initial_values.h"
#include "offline_data.h"
#include "problem_description.h"
#include "sparse_matrix_simd.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

namespace ryujin
{

  /**
   * A class signalling a restart, thrown in EulerModule::single_step and
   * caught at various places.
   */
  class Restart final
  {
  };

  /**
   * Explicit (strong stability preserving) time-stepping for the
   * compressible Euler equations described in ProblemDescription.
   *
   * This module is described in detail in @cite KronbichlerMaier2021, Alg.
   * 1.
   *
   * @todo Write out some more documentation
   *
   * @ingroup EulerModule
   */
  template <int dim, typename Number = double>
  class EulerModule final : public dealii::ParameterAcceptor
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
     * @copydoc ProblemDescription::flux_type
     */
    using flux_type = ProblemDescription::flux_type<dim, Number>;

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
    EulerModule(const MPI_Comm &mpi_communicator,
                std::map<std::string, dealii::Timer> &computing_timer,
                const ryujin::OfflineData<dim, Number> &offline_data,
                const ryujin::ProblemDescription &problem_description,
                const ryujin::InitialValues<dim, Number> &initial_values,
                const std::string &subsection = "EulerModule");

    /**
     * Prepare time stepping. A call to @ref prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector U perform an explicit
     * euler step (and store the result in U). The function returns the
     * computed maximal time step size tau_max.
     *
     * The time step is performed with either tau_max (if tau == 0), or tau
     * (if tau != 0). Here, tau_max is the computed maximal time step size
     * and tau is the optional third parameter.
     */
    Number euler_step(vector_type &U, Number t, Number tau = 0.);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * Heun 2nd order step (and store the result in U). The function
     * returns the computed maximal time step size tau_max.
     *
     * The time step is performed with either tau_max (if tau == 0), or tau
     * (if tau != 0). Here, tau_max is the computed maximal time step size
     * and tau is the optional third parameter.
     *
     * See @cite Shu1988, Eq. 2.15.
     */
    Number ssph2_step(vector_type &U, Number t, Number tau = 0.);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * SSP Runge Kutta 3rd order step (and store the result in U). The
     * function returns the computed maximal time step size tau_max.
     *
     * The time step is performed with either tau_max (if tau == 0), or tau
     * (if tau != 0). Here, tau_max is the computed maximal time step size
     * and tau is the optional third parameter.
     *
     * See @cite Shu1988, Eq. 2.18.
     */
    Number ssprk3_step(vector_type &U, Number t, Number tau = 0.);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * This function switches between euler_step(), ssph2_step(), or
     * ssprk3_step() depending on selected approximation order.
     *
     * The time step is performed with either tau_max (if tau == 0), or tau
     * (if tau != 0). Here, tau_max is the computed maximal time step size
     * and tau is the optional third parameter.
     */
    Number step(vector_type &U, Number t, Number tau = 0.);

  private:
    //@}
    /**
     * @name Internally used time-stepping primitives
     */
    //@{

    Number single_step(vector_type &U, Number tau);

    void apply_boundary_conditions(vector_type &U, Number t);

    //@}
    /**
     * @name Run time options
     */
    //@{

    Number cfl_min_;
    Number cfl_max_;

    unsigned int time_step_order_;
    unsigned int limiter_iter_;

    bool enforce_noslip_;

    //@}

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::ProblemDescription> problem_description_;
    dealii::SmartPointer<const ryujin::InitialValues<dim, Number>>
        initial_values_;

    Number cfl_;
    ACCESSOR_READ_ONLY(cfl)

    bool restart_possible_;

    unsigned int n_restarts_;
    ACCESSOR_READ_ONLY(n_restarts)

    scalar_type alpha_;

    scalar_type second_variations_;
    scalar_type specific_entropies_;
    scalar_type evc_entropies_;

    MultiComponentVector<Number, Limiter<dim, Number>::n_bounds> bounds_;

    vector_type r_;

    SparseMatrixSIMD<Number> dij_matrix_;

    SparseMatrixSIMD<Number> lij_matrix_;
    SparseMatrixSIMD<Number> lij_matrix_next_;

    SparseMatrixSIMD<Number, problem_dimension> pij_matrix_;

    vector_type temp_euler_;
    vector_type temp_ssp_;

    //@}
  };

} /* namespace ryujin */
