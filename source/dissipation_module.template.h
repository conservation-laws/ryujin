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

    /* Initialize vectors: */

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    density_n_.reinit(scalar_partitioner);
    internal_energy_n_.reinit(scalar_partitioner);

    momentum_n_.reinit(dim);
    velocity_.reinit(dim);
    for (unsigned int i = 0; i < dim; ++i) {
      momentum_n_.block(i).reinit(scalar_partitioner);
      velocity_.block(i).reinit(scalar_partitioner);
    }
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

    constexpr auto simd_length = VA::size();
    const unsigned int n_relevant = offline_data_->n_locally_relevant();

    /*
     * Step 0: Copy vectors
     *
     * FIXME: Memory access is suboptimal...
     */
    {
      Scope scope(computing_timer_, "time step [N] 0 - copy vectors");

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_0");

      const unsigned int size_regular = n_relevant / simd_length * simd_length;

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        using PD = ProblemDescription<dim, VA>;

        const auto U_i = U.get_vectorized_tensor(i);
        const auto rho_i = U_i[0];
        const auto m_i = PD::momentum(U_i);
        const auto rho_e_i = PD::internal_energy(U_i);

        simd_store(density_n_, rho_i, i);
        for (unsigned int d = 0; d < dim; ++d)
          simd_store(momentum_n_.block(d), m_i[d], i);
        simd_store(internal_energy_n_, rho_e_i, i);
      }

      for (unsigned int i = size_regular; i < n_relevant; ++i) {
        using PD = ProblemDescription<dim, Number>;

        const auto U_i = U.get_tensor(i);
        const auto rho_i = U_i[0];
        const auto m_i = PD::momentum(U_i);
        const auto rho_e_i = PD::internal_energy(U_i);

        density_n_.local_element(i) = rho_i;
        for (unsigned int d = 0; d < dim; ++d)
          momentum_n_.block(d).local_element(i) = m_i[d];
        internal_energy_n_.local_element(i) = rho_e_i;
      }

      LIKWID_MARKER_STOP("time_step_0");
      RYUJIN_PARALLEL_REGION_END
    }

    /*
     * Step 1: Solve velocity update:
     */
    {
      Scope scope(computing_timer_, "time step [N] 1 - update velocities");

      LIKWID_MARKER_START("time_step_n_1");

      LIKWID_MARKER_STOP("time_step_n_1");
    }

    /*
     * Step 2: Solve internal energy update:
     */
    {
      Scope scope(computing_timer_, "time step [N] 2 - update internal energy");

      LIKWID_MARKER_START("time_step_n_2");

      LIKWID_MARKER_STOP("time_step_n_2");
    }

    /*
     * Step 3: Copy vectors
     *
     * FIXME: Memory access is suboptimal...
     */
    {
      Scope scope(computing_timer_, "time step [N] 3 - write back vectors");

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_0");

      const unsigned int size_regular = n_relevant / simd_length * simd_length;

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        using PD = ProblemDescription<dim, VA>;

        auto U_i = U.get_vectorized_tensor(i);
        const auto rho_i = U_i[0];

        /* (5.4b) */
        auto m_i_new = -PD::momentum(U_i);
        for (unsigned int d = 0; d < dim; ++d) {
          m_i_new[d] += 2. * rho_i * simd_load(velocity_.block(d), i);
        }

        /* (5.12)f */
        auto rho_e_i_new = -PD::internal_energy(U_i);
        rho_e_i_new += 2. * rho_i * simd_load(internal_energy_, i);

        /* (5.18) */
        const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

        for (unsigned int d = 0; d < dim; ++d)
          U_i[1 + d] = m_i_new[d];
        U_i[1 + dim] = E_i_new;

        U.write_vectorized_tensor(U_i, i);
      }

      for (unsigned int i = size_regular; i < n_relevant; ++i) {
        using PD = ProblemDescription<dim, Number>;

        auto U_i = U.get_tensor(i);
        const auto rho_i = U_i[0];

        /* (5.4b) */
        auto m_i_new = -PD::momentum(U_i);
        for (unsigned int d = 0; d < dim; ++d) {
          m_i_new[d] += 2. * rho_i * velocity_.block(d).local_element(i);
        }

        /* (5.12)f */
        auto rho_e_i_new = -PD::internal_energy(U_i);
        rho_e_i_new += 2. * rho_i * internal_energy_.local_element(i);

        /* (5.18) */
        const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

        for (unsigned int d = 0; d < dim; ++d)
          U_i[1 + d] = m_i_new[d];
        U_i[1 + dim] = E_i_new;

        U.write_tensor(U_i, i);
      }

      LIKWID_MARKER_STOP("time_step_0");
      RYUJIN_PARALLEL_REGION_END
    }

    CALLGRIND_STOP_INSTRUMENTATION

    return tau;
  }


} /* namespace ryujin */

#endif /* DISSIPATION_MODULE_TEMPLATE_H */
