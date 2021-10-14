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

#include <functional>

namespace ryujin
{

  /**
   * Explicit forward Euler time-stepping for hyperbolic systems with
   * convex limiting.
   *
   * This module is described in detail in @cite ryujin-2021-1, Alg. 1.
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
     * TODO documentation
     *
     * Given a reference to a previous state vector U perform an explicit
     * euler step (and store the result in U). The function returns the
     * computed maximal time step size tau_max.
     *
     * The time step is performed with either tau_max (if tau == 0), or tau
     * (if tau != 0). Here, tau_max is the computed maximal time step size
     * and tau is the optional third parameter.
     */
    template <unsigned int l>
    Number
    step(const vector_type &old_U,
         std::array<std::reference_wrapper<const vector_type>, l> Us,
         std::array<std::reference_wrapper<const SparseMatrixSIMD<Number>>, l>
             dijHs,
         const std::array<Number, l> alphas,
         vector_type &new_U,
         SparseMatrixSIMD<Number> &new_dijH,
         Number tau = 0.) const;

    /**
     * TODO documentation
     */
    void apply_boundary_conditions(vector_type &U, Number t) const;

  private:

   //@}
    /**
     * @name Run time options
     */
    //@{

    unsigned int limiter_iter_;

    bool cfl_with_boundary_dofs_;

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

    unsigned int n_restarts_;
    ACCESSOR_READ_ONLY(n_restarts)

    unsigned int n_warnings_;
    ACCESSOR_READ_ONLY(n_warnings)

    mutable scalar_type alpha_;
    mutable scalar_type second_variations_;
    mutable scalar_type specific_entropies_;
    mutable scalar_type evc_entropies_;

    mutable MultiComponentVector<Number, Limiter<dim, Number>::n_bounds> bounds_;

    mutable vector_type r_;

    mutable SparseMatrixSIMD<Number> dij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_next_;
    mutable SparseMatrixSIMD<Number, problem_dimension> pij_matrix_;

    //@}
  };

} /* namespace ryujin */
