#ifndef TIME_STEP_TEMPLATE_H
#define TIME_STEP_TEMPLATE_H

#include "scope.h"
#include "time_step.h"

#include "indicator.h"
#include "riemann_solver.h"

#include <boost/range/irange.hpp>

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

namespace grendel
{
  using namespace dealii;


  template <int dim, typename Number>
  TimeStep<dim, Number>::TimeStep(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const grendel::OfflineData<dim, Number> &offline_data,
      const grendel::InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "TimeStep"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , initial_values_(&initial_values)
      , n_restarts_(0)
  {
    cfl_update_ = Number(0.95);
    add_parameter(
        "cfl update", cfl_update_, "relative CFL constant used for update");

    cfl_max_ = Number(1.0);
    add_parameter(
        "cfl max", cfl_max_, "Maximal admissible relative CFL constant");
  }


  template <int dim, typename Number>
  void TimeStep<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeStep<dim, Number>::prepare()" << std::endl;
#endif

    /* Initialize (global) vectors: */

    const auto &partitioner = offline_data_->partitioner();

    second_variations_.reinit(partitioner);
    alpha_.reinit(partitioner);

    for (auto &it : bounds_)
      it.reinit(partitioner);

    for (auto &it : r_)
      it.reinit(partitioner);

    for (auto &it : temp_euler_)
      it.reinit(partitioner);

    for (auto &it : temp_ssp_)
      it.reinit(partitioner);

    /* Initialize local matrices: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

    dij_matrix_.reinit(sparsity_simd);
    lij_matrix_.reinit(sparsity_simd);

    pij_matrix_.reinit(sparsity_simd);
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::euler_step(vector_type &U, Number t, Number tau)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeStep<dim, Number>::euler_step()" << std::endl;
#endif

    CALLGRIND_START_INSTRUMENTATION

    /* Index ranges for the iteration over the sparsity pattern : */

    constexpr auto n_array_elements = VectorizedArray<Number>::n_array_elements;

    const unsigned int n_export_indices = offline_data_->n_export_indices();
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    /* References to precomputed matrices and the stencil: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &lumped_mass_matrix_inverse =
        offline_data_->lumped_mass_matrix_inverse();
    const auto &mass_matrix = offline_data_->mass_matrix();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();

    const auto &boundary_normal_map = offline_data_->boundary_normal_map();
    const Number measure_of_omega_inverse =
        Number(1.) / offline_data_->measure_of_omega();


    /*
     * Step 1: Compute off-diagonal d_ij, and alpha_i
     *
     * The computation of the d_ij is quite costly. So we do a trick to
     * save a bit of computational resources. Instead of computing all d_ij
     * entries for a row of a given local index i, we only compute d_ij for
     * which j > i,
     *
     *        llllrr
     *      l .xxxxx
     *      l ..xxxx
     *      l ...xxx
     *      l ....xx
     *      r ......
     *      r ......
     *
     *  and symmetrize in Step 2.
     *
     *  MM: We could save a bit more computational resources by only
     *  computing entries for which *IN A GLOBAL* enumeration j > i. But
     *  the index translation, subsequent symmetrization, and exchange
     *  sounds a bit too expensive...
     */

    {
      Scope scope(computing_timer_, "time step 1 - compute d_ij, and alpha_i");

      std::atomic_int n_threads_above_n_export_indices = 0;

      GRENDEL_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_1");

      /* Stored thread locally: */
      Indicator<dim, Number> indicator_serial;

      /* Parallel non-vectorized loop: */

      GRENDEL_OMP_FOR
      for (unsigned int i = n_internal; i < n_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom */
        if (row_length == 1)
          continue;

        const auto U_i = gather(U, i);

        const Number mass = lumped_mass_matrix.local_element(i);
        const Number hd_i = mass * measure_of_omega_inverse;

        indicator_serial.reset(U_i);

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const unsigned int j = js[col_idx];

          const auto U_j = gather(U, j);
          const auto c_ij = cij_matrix.get_tensor(i, col_idx);
          const auto beta_ij = betaij_matrix.get_entry(i, col_idx);
          indicator_serial.add(U_j, c_ij, beta_ij);

          /* Only iterate over the upper triangular portion of d_ij */
          if (j <= i)
            continue;

          const auto norm = c_ij.norm();
          const auto n_ij = c_ij / norm;

          const auto [lambda_max, p_star, n_iterations] =
              RiemannSolver<dim, Number>::compute(U_i, U_j, n_ij, hd_i);

          Number d = norm * lambda_max;

          /*
           * In case both dofs are located at the boundary we have to
           * symmetrize.
           */

          if (boundary_normal_map.count(i) != 0 &&
              boundary_normal_map.count(j) != 0) {

            const auto c_ji = cij_matrix.get_transposed_tensor(i, col_idx);
            const auto norm_2 = c_ji.norm();
            const auto n_ji = c_ji / norm_2;

            auto [lambda_max_2, p_star_2, n_iterations_2] =
                RiemannSolver<dim, Number>::compute(U_j, U_i, n_ji, hd_i);
            d = std::max(d, norm_2 * lambda_max_2);
          }

          dij_matrix_.write_entry(d, i, col_idx);
        }

        alpha_.local_element(i) = indicator_serial.alpha(hd_i);
        second_variations_.local_element(i) =
            indicator_serial.second_variations();
      } /* parallel non-vectorized loop */

      /* Stored thread locally: */
      Indicator<dim, VectorizedArray<Number>> indicator_simd;
      bool above_n_export_indices = false;

      /* Parallel SIMD loop: */

      GRENDEL_OMP_FOR
      for (unsigned int i = 0; i < n_internal; i += n_array_elements) {

        if (GRENDEL_UNLIKELY(above_n_export_indices == false &&
                             i >= n_export_indices)) {
          above_n_export_indices = true;
          if(++n_threads_above_n_export_indices == omp_get_num_threads()) {
            /* Synchronize over all MPI processes: */
            alpha_.update_ghost_values_start(/*channel*/ 0);
            second_variations_.update_ghost_values_start(/*channel*/ 1);
          }
        }

        const auto U_i = simd_gather(U, i);
        indicator_simd.reset(U_i);

        const auto mass = simd_gather(lumped_mass_matrix, i);
        const auto hd_i = mass * measure_of_omega_inverse;

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i) + n_array_elements;
        for (unsigned int col_idx = 1; col_idx < row_length;
             ++col_idx, js += n_array_elements) {

          bool all_below_diagonal = true;
          for (unsigned int k = 0; k < n_array_elements; ++k)
            if (js[k] >= i + k) {
              all_below_diagonal = false;
              break;
            }

          const auto U_j = simd_gather(U, js);

          const auto c_ij = cij_matrix.get_vectorized_tensor(i, col_idx);
          const auto beta_ij = betaij_matrix.get_vectorized_entry(i, col_idx);
          indicator_simd.add(U_j, c_ij, beta_ij);

          /* Only iterate over the upper triangular portion of d_ij */
          if (all_below_diagonal)
            continue;

          const auto norm = c_ij.norm();
          const auto n_ij = c_ij / norm;

          const auto [lambda_max, p_star, n_iterations] =
              RiemannSolver<dim, VectorizedArray<Number>>::compute(
                  U_i, U_j, n_ij, hd_i);

          const auto d = norm * lambda_max;

          dij_matrix_.write_vectorized_entry(d, i, col_idx, true);
        }

        simd_scatter(alpha_, indicator_simd.alpha(hd_i), i);
        simd_scatter(second_variations_, indicator_simd.second_variations(), i);
      } /* parallel SIMD loop */

      --n_threads_above_n_export_indices;

      LIKWID_MARKER_STOP("time_step_1");
      GRENDEL_PARALLEL_REGION_END

      if(n_threads_above_n_export_indices < 0) {
        /* Synchronize over all MPI processes: */
        alpha_.update_ghost_values_start(/*channel*/ 0);
        second_variations_.update_ghost_values_start(/*channel*/ 1);
      }
    }

    {
      Scope scope(computing_timer_, "time sync 1 - wait");

      /* Synchronize over all MPI processes: */
      alpha_.update_ghost_values_finish();
      second_variations_.update_ghost_values_finish();
    }

    /*
     * Step 2: Compute diagonal of d_ij, and maximal time-step size.
     */

    std::atomic<Number> tau_max{std::numeric_limits<Number>::infinity()};

    {
      Scope scope(computing_timer_, "time step 2 - compute d_ii, and tau_max");

      /* Parallel region */
      GRENDEL_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_2");

      /* Parallel non-vectorized loop: */

      GRENDEL_OMP_FOR
      for (unsigned int i = 0; i < n_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom */
        if (row_length == 1)
          continue;

        Number d_sum = Number(0.);

        const unsigned int *js = sparsity_simd.columns(i);

        /* skip diagonal: */
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const auto j = *(i < n_internal ? js + col_idx * n_array_elements
                                          : js + col_idx);

          // fill lower triangular part of dij_matrix missing from step 1
          if (j < i) {
            const auto d_ji = dij_matrix_.get_transposed_entry(i, col_idx);
            dij_matrix_.write_entry(d_ji, i, col_idx);
          }

          d_sum -= dij_matrix_.get_entry(i, col_idx);
        }

        /* write diagonal element */
        dij_matrix_.write_entry(d_sum, i, 0);

        const Number mass = lumped_mass_matrix.local_element(i);
        const Number tau = cfl_update_ * mass / (Number(-2.) * d_sum);

        Number current_tau_max = tau_max.load();
        while (current_tau_max > tau &&
               !tau_max.compare_exchange_weak(current_tau_max, tau))
          ;
      } /* parallel non-vectorized loop */

      LIKWID_MARKER_STOP("time_step_2");
      GRENDEL_PARALLEL_REGION_END
    }

    {
      Scope scope(computing_timer_, "time sync 2");

      /* Synchronize over all MPI processes: */
      tau_max.store(Utilities::MPI::min(tau_max.load(), mpi_communicator_));

      AssertThrow(!std::isnan(tau_max) && !std::isinf(tau_max) && tau_max > 0.,
                  ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                             "do that. - We crashed."));

#ifdef DEBUG_OUTPUT
      deallog << "        computed tau_max = " << tau_max << std::endl;
#endif
      tau = (tau == Number(0.) ? tau_max.load() : tau);
#ifdef DEBUG_OUTPUT
      deallog << "        perform time-step with tau = " << tau << std::endl;
#endif

      if (tau * cfl_update_ > tau_max.load() * cfl_max_) {
#ifdef DEBUG_OUTPUT
        deallog << "        insufficient CFL, refuse update and abort stepping"
                << std::endl;
#endif
        U[0] *= std::numeric_limits<Number>::quiet_NaN();
        return tau_max;
      }
    }

    /*
     * Step 3: Low-order update, also compute limiter bounds, R_i
     *
     *   \bar U_ij = 1/2 (U_i + U_j) - 1/2 (f_j - f_i) c_ij / d_ij^L
     *
     *        R_i = \sum_j - c_ij f_j + d_ij^H (U_j - U_i)
     *
     *   Low-order update: += tau / m_i * 2 d_ij^L (\bar U_ij)
     */

    {
      Scope scope(computing_timer_,
                  "time step 3 - l.-o. update, bounds, and r_i");

      std::atomic_int n_threads_above_n_export_indices = 0;

      /* Parallel region */
      GRENDEL_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_3");

      /* Nota bene: This bounds variable is thread local: */
      Limiter<dim, Number> limiter_serial;

      /* Parallel non-vectorized loop: */

      GRENDEL_OMP_FOR
      for (unsigned int i = n_internal; i < n_owned; ++i) {

        /* Skip constrained degrees of freedom */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        const auto U_i = gather(U, i);
        auto U_i_new = U_i;

        const auto f_i = ProblemDescription<dim, Number>::f(U_i);
        const auto alpha_i = alpha_.local_element(i);
        const auto variations_i = second_variations_.local_element(i);

        const Number m_i = lumped_mass_matrix.local_element(i);
        const Number m_i_inv = lumped_mass_matrix_inverse.local_element(i);

        rank1_type r_i;

        /* Clear bounds: */
        limiter_serial.reset();
        limiter_serial.reset_variations(variations_i);

        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
          const auto j = js[col_idx];

          const auto U_j = gather(U, j);
          const auto f_j = ProblemDescription<dim, Number>::f(U_j);
          const auto alpha_j = alpha_.local_element(j);
          const auto variations_j = second_variations_.local_element(j);

          const auto c_ij = cij_matrix.get_tensor(i, col_idx);
          const auto d_ij = dij_matrix_.get_entry(i, col_idx);
          const Number d_ij_inv = Number(1.) / d_ij;

          const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                     Indicator<dim, Number>::Indicators::
                                         entropy_viscosity_commutator
                                 ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                 : d_ij * std::max(alpha_i, alpha_j);

          dealii::Tensor<1, problem_dimension> U_ij_bar;

          for (unsigned int k = 0; k < problem_dimension; ++k) {
            const auto temp = (f_j[k] - f_i[k]) * c_ij;

            r_i[k] += -temp + d_ijH * (U_j - U_i)[k];
            U_ij_bar[k] =
                Number(0.5) * (U_i[k] + U_j[k]) - Number(0.5) * temp * d_ij_inv;
          }

          U_i_new += tau * m_i_inv * Number(2.) * d_ij * U_ij_bar;

          limiter_serial.accumulate(
              U_i, U_j, U_ij_bar, /* is diagonal */ col_idx == 0);
          const auto beta_ij = betaij_matrix.get_entry(i, col_idx);
          limiter_serial.accumulate_variations(variations_j, beta_ij);
        }

        scatter(temp_euler_, U_i_new, i);
        scatter(r_, r_i, i);

        const Number hd_i = m_i * measure_of_omega_inverse;
        limiter_serial.apply_relaxation(hd_i);
        scatter(bounds_, limiter_serial.bounds(), i);
      } /* parallel non-vectorized loop */

      /* Nota bene: This bounds variable is thread local: */
      Limiter<dim, VectorizedArray<Number>> limiter_simd;
      bool above_n_export_indices = false;

      /* Parallel SIMD loop: */

      GRENDEL_OMP_FOR
      for (unsigned int i = 0; i < n_internal; i += n_array_elements) {

        if (GRENDEL_UNLIKELY(above_n_export_indices == false &&
                             i >= n_export_indices)) {
          above_n_export_indices = true;
          if(++n_threads_above_n_export_indices == omp_get_num_threads()) {
            unsigned int channel = 2;
            /* Synchronize over all MPI processes: */
            for (auto &it : r_)
              it.update_ghost_values_start(channel++);
          }
        }

        const auto U_i = simd_gather(U, i);
        auto U_i_new = U_i;

        const auto f_i =
            ProblemDescription<dim, VectorizedArray<Number>>::f(U_i);

        const auto alpha_i = simd_gather(alpha_, i);
        const auto variations_i = simd_gather(second_variations_, i);

        const auto m_i = simd_gather(lumped_mass_matrix, i);
        const auto m_i_inv = simd_gather(lumped_mass_matrix_inverse, i);

        using rank1_type =
            typename ProblemDescription<dim,
                                        VectorizedArray<Number>>::rank1_type;
        rank1_type r_i;

        /* Clear bounds: */
        limiter_simd.reset();
        limiter_simd.reset_variations(variations_i);

        const unsigned int *js = sparsity_simd.columns(i);
        const unsigned int row_length = sparsity_simd.row_length(i);

        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += n_array_elements) {

          const auto U_j = simd_gather(U, js);

          const auto f_j =
              ProblemDescription<dim, VectorizedArray<Number>>::f(U_j);

          const auto alpha_j = simd_gather(alpha_, js);
          const auto variations_j = simd_gather(second_variations_, js);

          const auto c_ij = cij_matrix.get_vectorized_tensor(i, col_idx);

          const auto d_ij = dij_matrix_.get_vectorized_entry(i, col_idx);
          const auto d_ij_inv = Number(1.) / d_ij;

          const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                     Indicator<dim, Number>::Indicators::
                                         entropy_viscosity_commutator
                                 ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                 : d_ij * std::max(alpha_i, alpha_j);

          dealii::Tensor<1, problem_dimension, VectorizedArray<Number>>
              U_ij_bar;

          for (unsigned int k = 0; k < problem_dimension; ++k) {
            const auto temp = (f_j[k] - f_i[k]) * c_ij;

            r_i[k] += -temp + d_ijH * (U_j - U_i)[k];
            U_ij_bar[k] =
                Number(0.5) * (U_i[k] + U_j[k]) - Number(0.5) * temp * d_ij_inv;
          }

          U_i_new += tau * m_i_inv * Number(2.) * d_ij * U_ij_bar;

          limiter_simd.accumulate(
              U_i, U_j, U_ij_bar, /* is diagonal */ col_idx == 0);
          const auto beta_ij = betaij_matrix.get_vectorized_entry(i, col_idx);
          limiter_simd.accumulate_variations(variations_j, beta_ij);
        }

        simd_scatter(temp_euler_, U_i_new, i);
        simd_scatter(r_, r_i, i);

        const auto hd_i = m_i * measure_of_omega_inverse;
        limiter_simd.apply_relaxation(hd_i);

        simd_scatter(bounds_, limiter_simd.bounds(), i);
      } /* parallel SIMD loop */

      --n_threads_above_n_export_indices;

      LIKWID_MARKER_STOP("time_step_3");
      GRENDEL_PARALLEL_REGION_END

      if(n_threads_above_n_export_indices < 0) {
        /* Synchronize over all MPI processes: */
        unsigned int channel = 2;
        for (auto &it : r_)
          it.update_ghost_values_start(channel++);
      }
    }

    {
      Scope scope(computing_timer_, "time sync 3 - wait");

      /* Synchronize over all MPI processes: */
      for (auto &it : r_)
        it.update_ghost_values_finish();
    }

    const unsigned int n_passes =
        (order_ == Order::second_order ? limiter_iter_ : 0);
    for (unsigned int pass = 0; pass < n_passes; ++pass) {

#ifdef DEBUG_OUTPUT
      deallog << "        limiter pass " << pass + 1 << std::endl;
#endif

      if (pass == 0) {
        /*
         * Step 4: Compute P_ij and l_ij (first round):
         *
         *    P_ij = tau / m_i / lambda ( (d_ij^H - d_ij^L) (U_i - U_j) +
         *                                (b_ij R_j - b_ji R_i) )
         */
        Scope scope(computing_timer_, "time step 4 - compute p_ij, and l_ij");

        std::atomic_int n_threads_above_n_export_indices = 0;

        GRENDEL_PARALLEL_REGION_BEGIN
        LIKWID_MARKER_START("time_step_4");

        /* Parallel non-vectorized loop: */

        GRENDEL_OMP_FOR
        for (unsigned int i = n_internal; i < n_owned; ++i) {

          /* Skip constrained degrees of freedom */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          const auto bounds = gather_array(bounds_, i);
          const auto U_i_new = gather(temp_euler_, i);

          const Number m_i_inv = lumped_mass_matrix_inverse.local_element(i);
          const auto alpha_i = alpha_.local_element(i);
          const Number lambda_inv = Number(row_length - 1);
          const auto U_i = gather(U, i);

          const auto r_i = gather(r_, i);

          const unsigned int *js = sparsity_simd.columns(i);
          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
            const auto j = js[col_idx];
            const Number m_j_inv = lumped_mass_matrix_inverse.local_element(j);

            const auto m_ij = mass_matrix.get_entry(i, col_idx);
            const auto b_ij =
                (col_idx == 0 ? Number(1.) : Number(0.)) - m_ij * m_j_inv;
            const auto b_ji =
                (col_idx == 0 ? Number(1.) : Number(0.)) - m_ij * m_i_inv;

            const auto d_ij = dij_matrix_.get_entry(i, col_idx);
            const auto alpha_j = alpha_.local_element(j);
            const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                       Indicator<dim, Number>::Indicators::
                                           entropy_viscosity_commutator
                                   ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                   : d_ij * std::max(alpha_i, alpha_j);

            const auto U_j = gather(U, j);
            const auto r_j = gather(r_, j);

            const auto p_ij =
                tau * m_i_inv * lambda_inv *
                ((d_ijH - d_ij) * (U_j - U_i) + b_ij * r_j - b_ji * r_i);
            pij_matrix_.write_tensor(p_ij, i, col_idx);

            const auto l_ij =
                Limiter<dim, Number>::limit(bounds, U_i_new, p_ij);
            lij_matrix_.write_entry(l_ij, i, col_idx);
          }
        } /* parallel non-vectorized loop */

        /* Parallel SIMD loop: */

        bool above_n_export_indices = false;

        GRENDEL_OMP_FOR
        for (unsigned int i = 0; i < n_internal; i += n_array_elements) {

          if (GRENDEL_UNLIKELY(above_n_export_indices == false &&
                               i >= n_export_indices)) {
            above_n_export_indices = true;
            if(++n_threads_above_n_export_indices == omp_get_num_threads()) {
              /* Synchronize over all MPI processes: */
              lij_matrix_.update_ghost_rows_start(
                  /*channel*/ problem_dimension + 2);
            }
          }

          const auto bounds = simd_gather_array(bounds_, i);
          const auto U_i_new = simd_gather(temp_euler_, i);

          const auto U_i = simd_gather(U, i);

          const auto alpha_i = simd_gather(alpha_, i);

          const auto m_i_inv = simd_gather(lumped_mass_matrix_inverse, i);

          const unsigned int row_length = sparsity_simd.row_length(i);
          const VectorizedArray<Number> lambda_inv = Number(row_length - 1);

          const auto r_i = simd_gather(r_, i);

          const unsigned int *js = sparsity_simd.columns(i);

          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += n_array_elements) {

            const auto m_j_inv = simd_gather(lumped_mass_matrix_inverse, js);

            const auto m_ij = mass_matrix.get_vectorized_entry(i, col_idx);
            const auto b_ij = (col_idx == 0 ? VectorizedArray<Number>(1.)
                                            : VectorizedArray<Number>(0.)) -
                              m_ij * m_j_inv;
            const auto b_ji = (col_idx == 0 ? VectorizedArray<Number>(1.)
                                            : VectorizedArray<Number>(0.)) -
                              m_ij * m_i_inv;

            const auto d_ij = dij_matrix_.get_vectorized_entry(i, col_idx);
            const auto alpha_j = simd_gather(alpha_, js);
            const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                       Indicator<dim, Number>::Indicators::
                                           entropy_viscosity_commutator
                                   ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                   : d_ij * std::max(alpha_i, alpha_j);

            const auto U_j = simd_gather(U, js);
            const auto r_j = simd_gather(r_, js);

            const auto p_ij =
                tau * m_i_inv * lambda_inv *
                ((d_ijH - d_ij) * (U_j - U_i) + b_ij * r_j - b_ji * r_i);
            pij_matrix_.write_vectorized_tensor(p_ij, i, col_idx, true);

            const auto l_ij = Limiter<dim, VectorizedArray<Number>>::limit(
                bounds, U_i_new, p_ij);

            lij_matrix_.write_vectorized_entry(l_ij, i, col_idx, true);
          }
        } /* parallel SIMD loop */

        --n_threads_above_n_export_indices;

        LIKWID_MARKER_STOP("time_step_4");
        GRENDEL_PARALLEL_REGION_END

        if(n_threads_above_n_export_indices < 0) {
          lij_matrix_.update_ghost_rows_start(
              /*channel*/ problem_dimension + 2);
        }

      } else {
        /*
         * Step 5: compute l_ij (second and later rounds):
         */
        Scope scope(computing_timer_, "time step 5 - compute l_ij");

        std::atomic_int n_threads_above_n_export_indices = 0;

        GRENDEL_PARALLEL_REGION_BEGIN
        LIKWID_MARKER_START("time_step_5");

        /* Parallel non-vectorized loop: */

        GRENDEL_OMP_FOR
        for (unsigned int i = n_internal; i < n_owned; ++i) {

          /* Skip constrained degrees of freedom */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          const auto bounds = gather_array(bounds_, i);
          const auto U_i_new = gather(temp_euler_, i);

          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
            auto p_ij = pij_matrix_.get_tensor(i, col_idx);
            const auto l_ij =
                Limiter<dim, Number>::limit(bounds, U_i_new, p_ij);
            lij_matrix_.write_entry(l_ij, i, col_idx);
          }
        } /* parallel non-vectorized loop */

        /* Parallel SIMD loop: */

        bool above_n_export_indices = false;

        GRENDEL_OMP_FOR
        for (unsigned int i = 0; i < n_internal; i += n_array_elements) {

          if (GRENDEL_UNLIKELY(above_n_export_indices == false &&
                               i >= n_export_indices)) {
            above_n_export_indices = true;
            if(++n_threads_above_n_export_indices == omp_get_num_threads()) {
              /* Synchronize over all MPI processes: */
              lij_matrix_.update_ghost_rows_start(
                  /*channel*/ problem_dimension + 2);
            }
          }

          const auto bounds = simd_gather_array(bounds_, i);
          const auto U_i_new = simd_gather(temp_euler_, i);

          const unsigned int row_length = sparsity_simd.row_length(i);
          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {

            const auto p_ij = pij_matrix_.get_vectorized_tensor(i, col_idx);

            const auto l_ij = Limiter<dim, VectorizedArray<Number>>::limit(
                bounds, U_i_new, p_ij);

            lij_matrix_.write_vectorized_entry(l_ij, i, col_idx, true);
          }
        } /* parallel SIMD loop */

        --n_threads_above_n_export_indices;

        LIKWID_MARKER_STOP("time_step_5");
        GRENDEL_PARALLEL_REGION_END

        if(n_threads_above_n_export_indices < 0) {
          lij_matrix_.update_ghost_rows_start(
              /*channel*/ problem_dimension + 2);
        }
      }

      {
        Scope scope(computing_timer_, "time sync 45- wait");

        /* Synchronize over all MPI processes: */
        lij_matrix_.update_ghost_rows_finish();
      }

      /*
       * Step 6: Perform high-order update:
       *
       *   Symmetrize l_ij
       *   High-order update: += l_ij * lambda * P_ij
       */

      {
        Scope scope(computing_timer_,
                    "time step 6 - symmetrize l_ij, h.-o. update");

        GRENDEL_PARALLEL_REGION_BEGIN
        LIKWID_MARKER_START("time_step_6");

        /* Parallel non-vectorized loop: */

        GRENDEL_OMP_FOR
        for (unsigned int i = n_internal; i < n_owned; ++i) {

          /* Skip constrained degrees of freedom */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          auto U_i_new = gather(temp_euler_, i);

          const Number lambda = Number(1.) / Number(row_length - 1);

          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
            auto p_ij = pij_matrix_.get_tensor(i, col_idx);

            const auto l_ji = lij_matrix_.get_transposed_entry(i, col_idx);
            const auto l_ij = std::min(lij_matrix_.get_entry(i, col_idx), l_ji);

            U_i_new += l_ij * lambda * p_ij;
            p_ij *= (1 - l_ij);

            if (pass + 1 < n_passes)
              pij_matrix_.write_tensor(p_ij, i, col_idx);
          }

#ifdef CHECK_BOUNDS
          const auto rho_new = U_i_new[0];
          const auto e_new =
              ProblemDescription<dim, Number>::internal_energy(U_i_new);
          const auto s_new =
              ProblemDescription<dim, Number>::specific_entropy(U_i_new);

          AssertThrowSIMD(rho_new,
                          [](auto val) { return val > Number(0.); },
                          dealii::ExcMessage("Negative density."));

          AssertThrowSIMD(e_new,
                          [](auto val) { return val > Number(0.); },
                          dealii::ExcMessage("Negative internal energy."));

          AssertThrowSIMD(s_new,
                          [](auto val) { return val > Number(0.); },
                          dealii::ExcMessage("Negative specific entropy."));
#endif

          scatter(temp_euler_, U_i_new, i);
        } /* parallel non-vectorized loop */

        /* Parallel vectorized loop: */

        GRENDEL_OMP_FOR
        for (unsigned int i = 0; i < n_internal; i += n_array_elements) {

          auto U_i_new = simd_gather(temp_euler_, i);

          const unsigned int row_length = sparsity_simd.row_length(i);
          const Number lambda = Number(1.) / Number(row_length - 1);

          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {

            VectorizedArray<Number> l_ji =
                lij_matrix_.get_vectorized_transposed_entry(i, col_idx);
            const auto l_ij =
                std::min(lij_matrix_.get_vectorized_entry(i, col_idx), l_ji);

            auto p_ij = pij_matrix_.get_vectorized_tensor(i, col_idx);

            U_i_new += l_ij * lambda * p_ij;
            p_ij *= (VectorizedArray<Number>(1.) - l_ij);

            if (pass + 1 < n_passes)
              pij_matrix_.write_vectorized_tensor(p_ij, i, col_idx, true);
          }

#ifdef CHECK_BOUNDS
          using PD = ProblemDescription<dim, VectorizedArray<Number>>;
          const auto rho_new = U_i_new[0];
          const auto e_new = PD::internal_energy(U_i_new);
          const auto s_new = PD::specific_entropy(U_i_new);

          AssertThrowSIMD(rho_new,
                          [](auto val) { return val > Number(0.); },
                          dealii::ExcMessage("Negative density."));

          AssertThrowSIMD(e_new,
                          [](auto val) { return val > Number(0.); },
                          dealii::ExcMessage("Negative internal energy."));

          AssertThrowSIMD(s_new,
                          [](auto val) { return val > Number(0.); },
                          dealii::ExcMessage("Negative specific entropy."));
#endif

          simd_scatter(temp_euler_, U_i_new, i);
        }

        LIKWID_MARKER_STOP("time_step_6");
        GRENDEL_PARALLEL_REGION_END
      }
    } /* limiter_iter_ */

    /*
     * Step 7: Fix boundary:
     */

    {
      Scope scope(computing_timer_, "time step 7 - fix boundary states");

      LIKWID_MARKER_START("time_step_7");

      const auto on_subranges = [&](const auto it1, const auto it2) {
        for (auto it = it1; it != it2; ++it) {

          const auto i = it->first;

          /* Only iterate over locally owned subset */
          if (i >= n_owned)
            continue;

          /* Skip constrained degrees of freedom */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          auto U_i = gather(temp_euler_, i);

          const auto &[normal, id, position] = it->second;

          /* On boundary 1 remove the normal component of the momentum: */

          if (id == Boundary::slip) {
            auto m = ProblemDescription<dim, Number>::momentum(U_i);
            m -= 1. * (m * normal) * normal;
            for (unsigned int k = 0; k < dim; ++k)
              U_i[k + 1] = m[k];
          }

          /* On boundary 2 enforce initial conditions: */

          if (id == Boundary::dirichlet) {
            U_i = initial_values_->initial_state(position, t + tau);
          }

          scatter(temp_euler_, U_i, i);
        }
      };

      // FIXME: This is currently not parallel:
      on_subranges(boundary_normal_map.begin(), boundary_normal_map.end());

      LIKWID_MARKER_STOP("time_step_7");
    }

    {
      Scope scope(computing_timer_, "time sync 7");

      /* Synchronize over all MPI processes: */
      for (auto &it : temp_euler_)
        it.update_ghost_values();
    }

    /* And finally update the result: */
    U.swap(temp_euler_);

    CALLGRIND_STOP_INSTRUMENTATION

    return tau_max;
  } // namespace grendel


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::ssph2_step(vector_type &U, Number t)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeStep<dim, Number>::ssph2_step()" << std::endl;
#endif

    Number tau_0 = 0.;

  restart_ssph2_step:
    /* This also copies ghost elements: */
    for (unsigned int k = 0; k < problem_dimension; ++k)
      temp_ssp_[k] = U[k];

    /* Step 1: U1 = U_old + tau * L(U_old) */
    Number tau_1 = euler_step(U, t, tau_0);

    AssertThrow(tau_1 >= tau_0,
                ExcMessage("failed to recover from CFL violation"));
    tau_1 = (tau_0 == 0. ? tau_1 : tau_0);

    /* Step 2: U2 = 1/2 U_old + 1/2 (U1 + tau L(U1)) */
    const Number tau_2 = euler_step(U, t, tau_1);

    AssertThrow(tau_2 >= tau_0,
                ExcMessage("failed to recover from CFL violation"));

    if (tau_2 * cfl_max_ < tau_1 * cfl_update_) {
      /* Restart and force smaller time step: */
#ifdef DEBUG_OUTPUT
      deallog << "        insufficient CFL, restart" << std::endl;
#endif
      tau_0 = tau_2 * cfl_update_;
      U.swap(temp_ssp_);
      ++n_restarts_;
      goto restart_ssph2_step;
    }

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(Number(1. / 2.), Number(1. / 2.), temp_ssp_[k]);

    return tau_1;
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::ssprk3_step(vector_type &U, Number t)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeStep<dim, Number>::ssprk3_step()" << std::endl;
#endif

    Number tau_0 = Number(0.);

  restart_ssprk3_step:
    /* This also copies ghost elements: */
    for (unsigned int k = 0; k < problem_dimension; ++k)
      temp_ssp_[k] = U[k];

    /* Step 1: U1 = U_old + tau * L(U_old) */
    Number tau_1 = euler_step(U, tau_0);

    AssertThrow(tau_1 >= tau_0,
                ExcMessage("failed to recover from CFL violation"));
    tau_1 = (tau_0 == 0. ? tau_1 : tau_0);

    /* Step 2: U2 = 3/4 U_old + 1/4 (U1 + tau L(U1)) */
    const Number tau_2 = euler_step(U, t, tau_1);

    AssertThrow(tau_2 >= tau_0,
                ExcMessage("failed to recover from CFL violation"));

    if (tau_2 * cfl_max_ < tau_1 * cfl_update_) {
      /* Restart and force smaller time step: */
#ifdef DEBUG_OUTPUT
      deallog << "        insufficient CFL, restart" << std::endl;
#endif
      tau_0 = tau_2 * cfl_update_;
      U.swap(temp_ssp_);
      ++n_restarts_;
      goto restart_ssprk3_step;
    }

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(Number(1. / 4.), Number(3. / 4.), temp_ssp_[k]);

    /* Step 3: U_new = 1/3 U_old + 2/3 (U2 + tau L(U2)) */
    const Number tau_3 = euler_step(U, t, tau_1);

    AssertThrow(tau_3 >= tau_0,
                ExcMessage("failed to recover from CFL violation"));

    if (tau_3 * cfl_max_ < tau_1 * cfl_update_) {
      /* Restart and force smaller time step: */
#ifdef DEBUG_OUTPUT
      deallog << "        insufficient CFL, restart" << std::endl;
#endif
      tau_0 = tau_3 * cfl_update_;
      U.swap(temp_ssp_);
      ++n_restarts_;
      goto restart_ssprk3_step;
    }

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(Number(2. / 3.), Number(1. / 3.), temp_ssp_[k]);

    return tau_1;
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::step(vector_type &U, Number t)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeStep<dim, Number>::step()" << std::endl;
#endif

    switch (time_step_order_) {
    case TimeStepOrder::first_order:
      return euler_step(U, t);
    case TimeStepOrder::second_order:
      return ssph2_step(U, t);
    case TimeStepOrder::third_order:
      return ssprk3_step(U, t);
    }

    __builtin_unreachable();
  }


} /* namespace grendel */

#endif /* TIME_STEP_TEMPLATE_H */
