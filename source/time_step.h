//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef TIME_STEP_H
#define TIME_STEP_H

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
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

namespace ryujin
{
  /**
   * Explicit (strong stability preserving) time-stepping for the
   * conservation law.
   *
   * @ingroup EulerStep
   */
  template <int dim, typename Number = double>
  class TimeStep final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription<dim, Number>::problem_dimension;
    // clang-format on

    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    /**
     * @copydoc ProblemDescription::rank2_type
     */
    using rank2_type = typename ProblemDescription<dim, Number>::rank2_type;

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
    TimeStep(const MPI_Comm &mpi_communicator,
             std::map<std::string, dealii::Timer> &computing_timer,
             const ryujin::OfflineData<dim, Number> &offline_data,
             const ryujin::InitialValues<dim, Number> &initial_values,
             const std::string &subsection = "TimeStep");

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
     * euler step (and store the result in U). The function
     *
     *  - returns the computed maximal time step size tau_max
     *
     *  - performs a time step and populates the vector U_new by the
     *    result. The time step is performed with either tau_max (if tau ==
     *    0), or tau (if tau != 0). Here, tau_max is the computed maximal
     *    time step size and tau is the optional third parameter.
     */
    Number euler_step(vector_type &U, Number t, Number tau = 0.);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * Heun 2nd order step (and store the result in U).
     *
     *  - returns the computed maximal time step size tau_max
     *
     * See @cite Shu_1988, Eq. 2.15.
     */
    Number ssph2_step(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * SSP Runge Kutta 3rd order step (and store the result in U).
     *
     *  - returns the computed maximal time step size tau_max
     *
     * See @cite Shu_1988, Eq. 2.18.
     */
    Number ssprk3_step(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * This function switches between euler_step(), ssph2_step(), or
     * ssprk3_step() depending on selected approximation order.
     */
    Number step(vector_type &U, Number t);

    //@}

    /* Options: */

    static constexpr enum class Order {
      first_order,
      second_order
    } order_ = ORDER;

    static constexpr enum class TimeStepOrder {
      first_order,
      second_order,
      third_order
    } time_step_order_ = TIME_STEP_ORDER;

    static constexpr unsigned int limiter_iter_ = LIMITER_ITER;

  private:
    /**
     * @name Run time options
     */
    //@{

    Number cfl_update_;
    Number cfl_max_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::InitialValues<dim, Number>>
        initial_values_;

    unsigned int n_restarts_;
    ACCESSOR_READ_ONLY(n_restarts)

    scalar_type alpha_;
    ACCESSOR_READ_ONLY(alpha)

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

#endif /* TIME_STEP_H */
