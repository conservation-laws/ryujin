//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef DISSIPATION_MODULE_TEMPLATE_H
#define DISSIPATION_MODULE_TEMPLATE_H

#include "dissipation_module.h"
#include "openmp.h"
#include "scope.h"
#include "simd.h"

#include "indicator.h"
#include "riemann_solver.h"

#include <atomic>

#ifdef VALGRIND_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_START_INSTRUMENTATION
#define CALLGRIND_STOP_INSTRUMENTATION
#endif

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_START(opt)
#define LIKWID_MARKER_STOP(opt)
#endif

#if defined(CHECK_BOUNDS) && !defined(DEBUG)
#define DEBUG
#endif

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  DissipationModule<dim, Number>::DissipationModule(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const ryujin::InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "DissipationModule"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , initial_values_(&initial_values)
  {
    tolerance_ = Number(1.0e-12);
    add_parameter(
        "tolerance", tolerance_, "Tolerance for linear solvers");
  }


  template <int dim, typename Number>
  void DissipationModule<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "DissipationModule<dim, Number>::prepare()" << std::endl;
#endif
  }


  template <int dim, typename Number>
  Number
  DissipationModule<dim, Number>::step(vector_type &U, Number t, Number tau)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "DissipationModule<dim, Number>::step()" << std::endl;
#endif

    CALLGRIND_START_INSTRUMENTATION

    using VA = VectorizedArray<Number>;

    /* Index ranges for the iteration over the sparsity pattern : */

//     constexpr auto simd_length = VA::size();
//     const unsigned int n_export_indices = offline_data_->n_export_indices();
//     const unsigned int n_internal = offline_data_->n_locally_internal();
//     const unsigned int n_owned = offline_data_->n_locally_owned();
//     const unsigned int n_relevant = offline_data_->n_locally_relevant();

    /* References to precomputed matrices and the stencil: */

    /*
     * Step 0: Precompute f(U) and the entropies of U
     */
    {
      Scope scope(computing_timer_, "dissipation step 0 - copy vector");

    }

    CALLGRIND_STOP_INSTRUMENTATION

    return tau;
  }


} /* namespace ryujin */

#endif /* DISSIPATION_MODULE_TEMPLATE_H */
