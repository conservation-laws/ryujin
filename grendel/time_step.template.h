#ifndef TIME_STEP_TEMPLATE_H
#define TIME_STEP_TEMPLATE_H

#include "time_step.h"

#include "indicator.h"
#include "riemann_solver.h"

#include <boost/range/irange.hpp>

#ifdef CALLGRIND
#include <valgrind/callgrind.h>
#endif

#include <atomic>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif


namespace grendel
{
  using namespace dealii;


  template <int dim, typename Number>
  TimeStep<dim, Number>::TimeStep(
      const MPI_Comm &mpi_communicator,
      dealii::TimerOutput &computing_timer,
      const grendel::OfflineData<dim, Number> &offline_data,
      const grendel::InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "TimeStep"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , initial_values_(&initial_values)
      , lij_matrix_communicator_(
            mpi_communicator, computing_timer, offline_data, lij_matrix_)
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
    deallog << "TimeStep<dim, Number>::prepare()" << std::endl;
    TimerOutput::Scope time(computing_timer_,
                            "time_step - prepare scratch space");

    /* Initialize (global) vectors: */

    const auto &partitioner = offline_data_->partitioner();

    alpha_.reinit(partitioner);

    rho_second_variation_.reinit(partitioner);
    rho_relaxation_.reinit(partitioner);

    for (auto &it : temp_euler_)
      it.reinit(partitioner);

    for (auto &it : temp_ssprk_)
      it.reinit(partitioner);

    for (auto &it : r_)
      it.reinit(partitioner);

    for (auto &it : bounds_)
      it.reinit(partitioner);

    /* Initialize local matrices: */

    const auto &sparsity = offline_data_->sparsity_pattern();

    for (auto &it : pij_matrix_)
      it.reinit(sparsity);

    dij_matrix_.reinit(sparsity);

    lij_matrix_.reinit(sparsity);

    if (Utilities::MPI::n_mpi_processes(mpi_communicator_) > 1)
      lij_matrix_communicator_.prepare();

    transposed_indices.resize(sparsity.n_nonzero_elements(),
                              numbers::invalid_unsigned_int);
    for (unsigned int row = 0; row < sparsity.n_rows(); ++row) {
      for (auto j = sparsity.begin(row); j != sparsity.end(row); ++j)
        transposed_indices[j->global_index()] =
            sparsity.row_position(j->column(), row);
    }
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::euler_step(vector_type &U, Number t, Number tau)
  {
    deallog << "TimeStep<dim, Number>::euler_step()" << std::endl;

#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif

    /* Index ranges for the iteration over the sparsity pattern : */

    constexpr auto n_array_elements = VectorizedArray<Number>::n_array_elements;

    const auto n_internal = offline_data_->n_locally_internal();
    const auto n_owned = offline_data_->n_locally_owned();
    const auto n_relevant = offline_data_->n_locally_relevant();

    /* Index ranges to iterator over dofs in serial: */

    const auto serial_owned = boost::irange<unsigned int>(0, n_owned);
    const auto serial_relevant = boost::irange<unsigned int>(0, n_relevant);

    /*
     * Index ranges for SIMD iteration:
     *
     * simd_internal is actually the only index range over which we iterate
     * vectorized, therefore we compute indices in increments of
     * n_array_elements. The simd_remaining_owned and
     * simd_remaining_relevant index ranges are then used to iterate over
     * the remaining degrees of freedom that do not allow a straightforward
     * vectorization
     */

    const auto simd_internal =
        boost::irange<unsigned int>(0, n_internal, n_array_elements);

    const auto simd_remaining_owned =
        boost::irange<unsigned int>(n_internal, n_owned);
    const auto simd_remaining_relevant =
        boost::irange<unsigned int>(n_internal, n_relevant);

    /* References to precomputed matrices and the stencil: */

    const auto &sparsity = offline_data_->sparsity_pattern();

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &bij_matrix = offline_data_->bij_matrix();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();

    const auto &boundary_normal_map = offline_data_->boundary_normal_map();
    const Number measure_of_omega_inverse =
        Number(1.) / offline_data_->measure_of_omega();

    /*
     * Step 1: Compute off-diagonal d_ij, and alpha_i
     */

    {
      deallog << "        compute d_ij, and alpha_i" << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 1 compute d_ij, and alpha_i");
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START("time_step_1");
#endif

      const auto step_1_simd = [&](auto i1, const auto i2) {
        /* Stored thread locally: */
        Indicator<dim, VectorizedArray<Number>> indicator(*offline_data_);

        for (auto i : boost::make_iterator_range(i1, i2)) {

          const auto U_i = simd_gather(U, i);
          indicator.reset(U_i);

          auto jts = generate_iterators<n_array_elements>(
              [&](auto k) { return sparsity.begin(i + k); });

          /* Skip diagonal. */
          increment_iterators(jts);

          for (; jts[0] != sparsity.end(i); increment_iterators(jts)) {

            const auto js = get_column_indices(jts);
            bool all_above_diagonal = true;
            for (unsigned int k = 0; k < js.size(); ++k)
              if (js[k] < i + k) {
                all_above_diagonal = false;
                break;
              }

            const auto U_j = simd_gather(U, js);

            indicator.add(U_j, jts);

            /* Only iterate over the subdiagonal for d_ij */
            if (all_above_diagonal)
              continue;

            const auto c_ij = gather_get_entry(cij_matrix, jts);
            const auto norm = c_ij.norm();
            const auto n_ij = c_ij / norm;

            const auto [lambda_max, p_star, n_iterations] =
                RiemannSolver<dim, VectorizedArray<Number>>::compute(
                    U_i, U_j, n_ij);

            const auto d = norm * lambda_max;

            /* Set lower diagonal values (the upper will be set in step 2): */
            set_entry(dij_matrix_, jts, d);
          }

          simd_scatter(
              rho_second_variation_, indicator.rho_second_variation(), i);

          const auto mass = simd_get_diag_element(lumped_mass_matrix, i);
          const auto hd_i = mass * measure_of_omega_inverse;
          simd_scatter(alpha_, indicator.alpha(hd_i), i);
        }
      };

      const auto step_1_serial = [&](auto i1, const auto i2) {
        /* Stored thread locally: */
        Indicator<dim, Number> indicator(*offline_data_);

        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const auto U_i = gather(U, i);

          indicator.reset(U_i);

          for (auto jt = ++sparsity.begin(i) /* skip diagonal */;
               jt != sparsity.end(i);
               ++jt) {
            const auto j = jt->column();

            const auto U_j = gather(U, j);
            indicator.add(U_j, jt);

            /* Only iterate over the subdiagonal for d_ij */
            if (j >= i)
              continue;

            const auto c_ij = gather_get_entry(cij_matrix, jt);
            const auto norm = c_ij.norm();
            const auto n_ij = c_ij / norm;

            const auto [lambda_max, p_star, n_iterations] =
                RiemannSolver<dim, Number>::compute(U_i, U_j, n_ij);

            Number d = norm * lambda_max;

            /*
             * In case both dofs are located at the boundary we have to
             * symmetrize.
             */

            if (boundary_normal_map.count(i) != 0 &&
                boundary_normal_map.count(j) != 0) {

              const auto c_ji = gather_get_entry(cij_matrix, j, i);
              const auto norm_2 = c_ji.norm();
              const auto n_ji = c_ji / norm_2;

              auto [lambda_max_2, p_star_2, n_iterations_2] =
                  RiemannSolver<dim, Number>::compute(U_j, U_i, n_ji);
              d = std::max(d, norm_2 * lambda_max_2);
            }

            /* Set lower diagonal values (the upper will be set in step 2): */
            set_entry(dij_matrix_, jt, d);
          }

          rho_second_variation_.local_element(i) =
              indicator.rho_second_variation();

          const Number mass = lumped_mass_matrix.diag_element(i);
          const Number hd_i = mass * measure_of_omega_inverse;
          alpha_.local_element(i) = indicator.alpha(hd_i);
        }
      };

      parallel::apply_to_subranges(
          simd_internal.begin(), simd_internal.end(), step_1_simd, 1024);

      parallel::apply_to_subranges(simd_remaining_relevant.begin(),
                                   simd_remaining_relevant.end(),
                                   step_1_serial,
                                   1024);

      /* Synchronize alpha_ over all MPI processes: */
      rho_second_variation_.update_ghost_values();
      alpha_.update_ghost_values();

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP("time_step_1");
#endif
    }


    /*
     * Step 2: Compute diagonal of d_ij, and maximal time-step size.
     */

    std::atomic<Number> tau_max{std::numeric_limits<Number>::infinity()};

    {
      deallog << "        compute d_ii, and tau_max" << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 2 compute d_ii, and tau_max");
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START("time_step_2");
#endif

      const auto step_2_serial = [&](auto i1, const auto i2) {
        Number tau_max_on_subrange = std::numeric_limits<Number>::infinity();

        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const Number delta_rho_i = rho_second_variation_.local_element(i);

          Number alpha_i = Number(0.);
          (void)alpha_i;
          Number d_sum = Number(0.);
          Number rho_relaxation_numerator = Number(0.);
          Number rho_relaxation_denominator = Number(0.);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            if (j == i)
              continue;

            // fill upper diagonal part of dij_matrix missing from step 1
            if (j > i)
              set_entry(
                  dij_matrix_,
                  jt,
                  get_transposed_entry(dij_matrix_, jt, transposed_indices));

            const Number delta_rho_j = rho_second_variation_.local_element(j);
            const auto beta_ij = get_entry(betaij_matrix, jt);

            /* The numerical constant 8 is up to debate... */
            rho_relaxation_numerator +=
                Number(8.0 * 0.5) * beta_ij * (delta_rho_i + delta_rho_j);
            rho_relaxation_denominator += beta_ij;

            d_sum -= get_entry(dij_matrix_, jt);
          }

          rho_relaxation_.local_element(i) =
              std::abs(rho_relaxation_numerator / rho_relaxation_denominator);

          dij_matrix_.diag_element(i) = d_sum;

          const Number mass = lumped_mass_matrix.diag_element(i);
          const Number tau = cfl_update_ * mass / (Number(-2.) * d_sum);
          tau_max_on_subrange = std::min(tau_max_on_subrange, tau);
        }

        Number current_tau_max = tau_max.load();
        while (current_tau_max > tau_max_on_subrange &&
               !tau_max.compare_exchange_weak(current_tau_max,
                                              tau_max_on_subrange))
          ;
      };

      parallel::apply_to_subranges(
          serial_relevant.begin(), serial_relevant.end(), step_2_serial, 1024);

      /* Synchronize tau_max over all MPI processes: */
      tau_max.store(Utilities::MPI::min(tau_max.load(), mpi_communicator_));

      AssertThrow(!std::isnan(tau_max) && !std::isinf(tau_max) && tau_max > 0.,
                  ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                             "do that. - We crashed."));

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP("time_step_2");
#endif

      deallog << "        computed tau_max = " << tau_max << std::endl;
    }

    tau = (tau == 0 ? tau_max.load() : tau);
    deallog << "        perform time-step with tau = " << tau << std::endl;


    /*
     * Step 3: Low-order update, also compute limiter bounds, R_i and first
     *         part of P_ij
     *
     *   \bar U_ij = 1/2 (U_i + U_j) - 1/2 (f_j - f_i) c_ij / d_ij^L
     *
     *        R_i = \sum_j - c_ij f_j + d_ij^H (U_j - U_i)
     *
     *        P_ij = tau / m_i / lambda (d_ij^H - d_ij^L) (U_i - U_j) + [...]
     *
     *   Low-order update: += tau / m_i * 2 d_ij^L (\bar U_ij)
     */

    {
      deallog << "        low-order update, limiter bounds, r_i, and p_ij"
              << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 3 low-order update, limiter bounds, "
                              "compute r_i, and p_ij (1)");
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START("time_step_3");
#endif

      const auto step_3_simd = [&](auto i1, const auto i2) {
        /* Notar bene: This bounds variable is thread local: */
        Limiter<dim, VectorizedArray<Number>> limiter;

        /* lambda is the same value for all interior dofs */
        const auto size = std::distance(sparsity.begin(*i1), sparsity.end(*i1));
        const Number lambda_inv = Number(size - 1);

        for (const auto i : boost::make_iterator_range(i1, i2)) {

          const auto U_i = simd_gather(U, i);
          auto U_i_new = U_i;

          const auto f_i =
              ProblemDescription<dim, VectorizedArray<Number>>::f(U_i);

          const auto alpha_i = simd_gather(alpha_, i);
          const auto m_i = simd_get_diag_element(lumped_mass_matrix, i);
          const auto m_i_inv = Number(1.) / m_i;

          using rank1_type =
              typename ProblemDescription<dim,
                                          VectorizedArray<Number>>::rank1_type;
          rank1_type r_i;

          /* Clear bounds: */
          limiter.reset();

          auto jts = generate_iterators<n_array_elements>(
              [&](auto k) { return sparsity.begin(i + k); });

          for (; jts[0] != sparsity.end(i); increment_iterators(jts)) {

            const auto js = get_column_indices(jts);
            const auto U_j = simd_gather(U, js);

            const auto f_j =
                ProblemDescription<dim, VectorizedArray<Number>>::f(U_j);
            const auto alpha_j = simd_gather(alpha_, js);

            const auto c_ij = gather_get_entry(cij_matrix, jts);
            const auto d_ij = get_entry(dij_matrix_, jts);
            const auto d_ij_inv = Number(1.) / d_ij;

            const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                       Indicator<dim, Number>::Indicators::
                                           entropy_viscosity_commutator
                                   ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                   : d_ij * std::max(alpha_i, alpha_j);

            const auto p_ij =
                tau * m_i_inv * lambda_inv * (d_ijH - d_ij) * (U_j - U_i);

            dealii::Tensor<1, problem_dimension, VectorizedArray<Number>>
                U_ij_bar;

            for (unsigned int k = 0; k < problem_dimension; ++k) {
              const auto temp = (f_j[k] - f_i[k]) * c_ij;

              r_i[k] += -temp + d_ijH * (U_j - U_i)[k];
              U_ij_bar[k] = Number(0.5) * (U_i[k] + U_j[k]) -
                            Number(0.5) * temp * d_ij_inv;
            }

            U_i_new += tau * m_i_inv * Number(2.) * d_ij * U_ij_bar;

            scatter_set_entry(pij_matrix_, jts, p_ij);

            limiter.accumulate(
                U_i, U_j, U_ij_bar, /* is diagonal */ js[0] == i);
          }

          simd_scatter(temp_euler_, U_i_new, i);
          simd_scatter(r_, r_i, i);

          const auto hd_i = m_i * measure_of_omega_inverse;
          const auto rho_relaxation_i = simd_gather(rho_relaxation_, i);

          limiter.apply_relaxation(hd_i, rho_relaxation_i);
          simd_scatter(bounds_, limiter.bounds(), i);
        }
      };

      const auto step_3_serial = [&](auto i1, const auto i2) {
        /* Notar bene: This bounds variable is thread local: */
        Limiter<dim, Number> limiter;

        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const auto U_i = gather(U, i);
          auto U_i_new = U_i;

          const auto f_i = ProblemDescription<dim, Number>::f(U_i);
          const auto alpha_i = alpha_.local_element(i);
          const Number m_i = lumped_mass_matrix.diag_element(i);
          const Number m_i_inv = Number(1.) / m_i;

          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const Number lambda_inv = Number(size - 1);

          rank1_type r_i;

          /* Clear bounds: */
          limiter.reset();

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            const auto U_j = gather(U, j);
            const auto f_j = ProblemDescription<dim, Number>::f(U_j);
            const auto alpha_j = alpha_.local_element(j);

            const auto c_ij = gather_get_entry(cij_matrix, jt);
            const auto d_ij = get_entry(dij_matrix_, jt);
            const Number d_ij_inv = Number(1.) / d_ij;

            const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                       Indicator<dim, Number>::Indicators::
                                           entropy_viscosity_commutator
                                   ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                   : d_ij * std::max(alpha_i, alpha_j);

            const auto p_ij =
                tau * m_i_inv * lambda_inv * (d_ijH - d_ij) * (U_j - U_i);

            dealii::Tensor<1, problem_dimension> U_ij_bar;

            for (unsigned int k = 0; k < problem_dimension; ++k) {
              const auto temp = (f_j[k] - f_i[k]) * c_ij;

              r_i[k] += -temp + d_ijH * (U_j - U_i)[k];
              U_ij_bar[k] = Number(0.5) * (U_i[k] + U_j[k]) -
                            Number(0.5) * temp * d_ij_inv;
            }

            U_i_new += tau * m_i_inv * Number(2.) * d_ij * U_ij_bar;

            scatter_set_entry(pij_matrix_, jt, p_ij);

            limiter.accumulate(U_i, U_j, U_ij_bar, /* is diagonal */ i == j);
          }

          scatter(temp_euler_, U_i_new, i);
          scatter(r_, r_i, i);

          const Number hd_i = m_i * measure_of_omega_inverse;
          const Number rho_relaxation_i = rho_relaxation_.local_element(i);
          limiter.apply_relaxation(hd_i, rho_relaxation_i);
          scatter(bounds_, limiter.bounds(), i);
        }
      };

      /* Only iterate over locally owned subset! */

      parallel::apply_to_subranges(
          simd_internal.begin(), simd_internal.end(), step_3_simd, 1024);

      parallel::apply_to_subranges(simd_remaining_owned.begin(),
                                   simd_remaining_owned.end(),
                                   step_3_serial,
                                   1024);

      /* Synchronize r_ over all MPI processes: */
      for (auto &it : r_)
        it.update_ghost_values();

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP("time_step_3");
#endif
    }


    /*
     * Step 4: Compute second part of P_ij:
     *
     *        P_ij = [...] + tau / m_i / lambda (b_ij R_j - b_ji R_i)
     */

    if constexpr (order_ == Order::second_order) {
      deallog << "        compute p_ij" << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 4 compute p_ij (2)");
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START("time_step_4");
#endif

      const auto on_subranges = [&](auto i1, const auto i2) {
        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Only iterate over locally owned subset! */
          Assert(i < n_owned, ExcInternalError());

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const Number m_i = lumped_mass_matrix.diag_element(i);
          const Number m_i_inv = Number(1.) / m_i;
          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const Number lambda_inv = Number(size - 1);

          const auto r_i = gather(r_, i);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            const auto b_ij = get_entry(bij_matrix, jt);
            const auto b_ji =
                get_transposed_entry(bij_matrix, jt, transposed_indices);
            auto p_ij = gather_get_entry(pij_matrix_, jt);

            const auto r_j = gather(r_, j);

            p_ij += tau * m_i_inv * lambda_inv * (b_ij * r_j - b_ji * r_i);
            scatter_set_entry(pij_matrix_, jt, p_ij);
          }
        }
      };

      parallel::apply_to_subranges(
          serial_owned.begin(), serial_owned.end(), on_subranges, 1024);

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP("time_step_4");
#endif
    }

    for (unsigned int i = 0;
         i < (order_ == Order::second_order ? limiter_iter_ : 0);
         ++i) {

      deallog << "        limiter pass " << i + 1 << std::endl;

      /*
       * Step 5: compute l_ij:
       */

      {
        deallog << "        compute l_ij" << std::endl;
        TimerOutput::Scope time(computing_timer_, "time_step - 5 compute l_ij");
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_START("time_step_5");
#endif

        const auto step_5_simd = [&](auto i1, const auto i2) {
          for (const auto i : boost::make_iterator_range(i1, i2)) {

            const auto bounds = simd_gather_array(bounds_, i);
            const auto U_i_new = simd_gather(temp_euler_, i);

            auto jts = generate_iterators<n_array_elements>(
                [&](auto k) { return sparsity.begin(i + k); });

            for (; jts[0] != sparsity.end(i); increment_iterators(jts)) {

              const auto p_ij = gather_get_entry(pij_matrix_, jts);
              const auto l_ij = Limiter<dim, VectorizedArray<Number>>::limit(
                  bounds, U_i_new, p_ij);

              set_entry(lij_matrix_, jts, l_ij);
            }
          }
        };

        const auto step_5_serial = [&](auto i1, const auto i2) {
          for (const auto i : boost::make_iterator_range(i1, i2)) {

            /* Skip constrained degrees of freedom */
            if (++sparsity.begin(i) == sparsity.end(i))
              continue;

            const auto bounds = gather_array(bounds_, i);
            const auto U_i_new = gather(temp_euler_, i);

            for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
              auto p_ij = gather_get_entry(pij_matrix_, jt);
              const auto l_ij =
                  Limiter<dim, Number>::limit(bounds, U_i_new, p_ij);
              set_entry(lij_matrix_, jt, l_ij);
            }
          }
        };

        parallel::apply_to_subranges(
            simd_internal.begin(), simd_internal.end(), step_5_simd, 1024);

        parallel::apply_to_subranges(simd_remaining_owned.begin(),
                                     simd_remaining_owned.end(),
                                     step_5_serial,
                                     1024);

#ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP("time_step_5");
#endif
      }

      /*
       * And exchange l_ij:
       */

      if (Utilities::MPI::n_mpi_processes(mpi_communicator_) > 1) {

        deallog << "        exchange l_ij" << std::endl;
        TimerOutput::Scope time(computing_timer_,
                                "time_step - 6 exchange l_ij");
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_START("time_step_6");
#endif

        lij_matrix_communicator_.synchronize();

#ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP("time_step_6");
#endif
      }

      /*
       * Step 7: Perform high-order update:
       *
       *   Symmetrize l_ij
       *   High-order update: += l_ij * lambda * P_ij
       */

      {
        deallog << "        symmetrize l_ij, high-order update" << std::endl;
        TimerOutput::Scope time(computing_timer_,
                                "time_step - 7 symmetrize l_ij, high-order update");
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_START("time_step_7");
#endif

        const auto on_subranges = [&](auto i1, const auto i2) {
          for (const auto i : boost::make_iterator_range(i1, i2)) {

            /* Only iterate over locally owned subset */
            Assert(i < n_owned, ExcInternalError());

            /* Skip constrained degrees of freedom */
            if (++sparsity.begin(i) == sparsity.end(i))
              continue;

            auto U_i_new = gather(temp_euler_, i);

            const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
            const Number lambda = Number(1.) / Number(size - 1);

            for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
              auto p_ij = gather_get_entry(pij_matrix_, jt);

              const auto l_ji =
                  get_transposed_entry(lij_matrix_, jt, transposed_indices);
              const auto l_ij = std::min(get_entry(lij_matrix_, jt), l_ji);

              U_i_new += l_ij * lambda * p_ij;
              p_ij *= (1 - l_ij);
              scatter_set_entry(pij_matrix_, jt, p_ij);
            }

            scatter(temp_euler_, U_i_new, i);
          }
        };

        parallel::apply_to_subranges(
            serial_owned.begin(), serial_owned.end(), on_subranges, 1024);

#ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP("time_step_7");
#endif
      }
    } /* limiter_iter_ */

    /*
     * Step 8: Fix boundary:
     */

    {
      deallog << "        fix up boundary states" << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 8 fix boundary states");
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START("time_step_8");
#endif

      const auto on_subranges = [&](const auto it1, const auto it2) {
        for (auto it = it1; it != it2; ++it) {

          const auto i = it->first;

          /* Only iterate over locally owned subset */
          if (i >= n_owned)
            continue;

          const auto &[normal, id, position] = it->second;

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          auto U_i = gather(temp_euler_, i);

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

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP("time_step_8");
#endif
    }

    /* Synchronize temp over all MPI processes: */
    for (auto &it : temp_euler_)
      it.update_ghost_values();

    /* And finally update the result: */
    U.swap(temp_euler_);

#ifdef CALLGRIND
    CALLGRIND_STOP_INSTRUMENTATION;
#endif

    return tau_max;
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::ssph2_step(vector_type &U, Number t)
  {
    deallog << "TimeStep<dim, Number>::ssph2_step()" << std::endl;

    /* This also copies ghost elements: */
    for (unsigned int k = 0; k < problem_dimension; ++k)
      temp_ssprk_[k] = U[k];

    // Step 1: U1 = U_old + tau * L(U_old)

    const Number tau_1 = euler_step(U, t);

    // Step 2: U2 = 1/2 U_old + 1/2 (U1 + tau L(U1))

    const Number tau_2 = euler_step(U, t, tau_1);

    const Number ratio = cfl_max_ / cfl_update_;
    AssertThrow(ratio * tau_2 >= tau_1,
                ExcMessage("Problem performing SSP H(2) time step: "
                           "CFL condition violated."));

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(Number(1. / 2.), Number(1. / 2.), temp_ssprk_[k]);

    return tau_1;
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::ssprk3_step(vector_type &U, Number t)
  {
    deallog << "TimeStep<dim, Number>::ssprk3_step()" << std::endl;

    /* This also copies ghost elements: */
    for (unsigned int k = 0; k < problem_dimension; ++k)
      temp_ssprk_[k] = U[k];

    // Step 1: U1 = U_old + tau * L(U_old)

    const Number tau_1 = euler_step(U, t);

    // Step 2: U2 = 3/4 U_old + 1/4 (U1 + tau L(U1))

    const Number tau_2 = euler_step(U, t, tau_1);

    const Number ratio = cfl_max_ / cfl_update_;
    AssertThrow(ratio * tau_2 >= tau_1,
                ExcMessage("Problem performing SSP RK(3) time step: "
                           "CFL condition violated."));

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(Number(1. / 4.), Number(3. / 4.), temp_ssprk_[k]);


    // Step 3: U_new = 1/3 U_old + 2/3 (U2 + tau L(U2))

    const Number tau_3 = euler_step(U, t, tau_1);

    AssertThrow(ratio * tau_3 >= tau_1,
                ExcMessage("Problem performing SSP RK(3) time step: "
                           "CFL condition violated."));

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(Number(2. / 3.), Number(1. / 3.), temp_ssprk_[k]);

    return tau_1;
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::step(vector_type &U, Number t)
  {
    deallog << "TimeStep<dim, Number>::step()" << std::endl;

    switch (time_step_order_) {
    case TimeStepOrder::first_order:
      return euler_step(U, t);
    case TimeStepOrder::second_order:
      return ssph2_step(U, t);
    case TimeStepOrder::third_order:
      return ssprk3_step(U, t);
    }
  }


} /* namespace grendel */

#endif /* TIME_STEP_TEMPLATE_H */
