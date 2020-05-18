//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef TIME_STEP_H
#define TIME_STEP_H

#include <compile_time_options.h>

#include "helper.h"
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

namespace grendel
{

  template <int dim, typename Number = double>
  class TimeStep : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;
    using rank2_type = typename ProblemDescription<dim, Number>::rank2_type;

    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;
    using vector_type = std::array<scalar_type, problem_dimension>;

    TimeStep(const MPI_Comm &mpi_communicator,
             std::map<std::string, dealii::Timer> &computing_timer,
             const grendel::OfflineData<dim, Number> &offline_data,
             const grendel::InitialValues<dim, Number> &initial_values,
             const std::string &subsection = "TimeStep");

    virtual ~TimeStep() final = default;

    void prepare();

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
     * [Shu & Osher, Efficient Implementation of Essentially
     * Non-oscillatory Shock-Capturing Schemes JCP 77:439-471 (1988), Eq.
     * 2.15]
     */
    Number ssph2_step(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * SSP Runge Kutta 3rd order step (and store the result in U).
     *
     *  - returns the computed maximal time step size tau_max
     *
     * [Shu & Osher, Efficient Implementation of Essentially
     * Non-oscillatory Shock-Capturing Schemes JCP 77:439-471 (1988), Eq.
     * 2.18]
     */
    Number ssprk3_step(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U perform an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * Depending on the approximation order (first or second order) this
     * function chooses the 2nd order Heun, or the third order Runge Kutta
     * time stepping scheme.
     */
    Number step(vector_type &U, Number t);

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
    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const grendel::InitialValues<dim, Number>>
        initial_values_;

    unsigned int n_restarts_;
    ACCESSOR_READ_ONLY(n_restarts)

    /* Scratch data: */

    scalar_type alpha_;
    ACCESSOR_READ_ONLY(alpha)

    scalar_type second_variations_;

    typename Limiter<dim, Number>::vector_type bounds_;

    vector_type r_;

    SparseMatrixSIMD<Number> dij_matrix_;
    SparseMatrixSIMD<Number> lij_matrix_;

    SparseMatrixSIMD<Number, problem_dimension> pij_matrix_;

    vector_type temp_euler_;
    vector_type temp_ssp_;

    dealii::AlignedVector<Number> specific_entropies;
    dealii::AlignedVector<Number> evc_entropies;
    dealii::AlignedVector<Number> u_and_flux_;

    /* Options: */

    Number cfl_update_;
    Number cfl_max_;
  };

} /* namespace grendel */

#endif /* TIME_STEP_H */
