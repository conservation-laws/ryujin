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
    use_ssprk3_ = order_ == Order::second_order;
    add_parameter(
        "use ssprk3", use_ssprk3_, "Use SSPRK3 instead of explicit Euler");

    cfl_update_ = 0.80;
    add_parameter(
        "cfl update", cfl_update_, "relative CFL constant used for update");

    cfl_max_ = 0.90;
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
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::euler_step(vector_type &U, Number t, Number tau)
  {
    deallog << "TimeStep<dim, Number>::euler_step()" << std::endl;

#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif

    const auto &n_locally_owned = offline_data_->n_locally_owned();
    const auto &n_locally_relevant = offline_data_->n_locally_relevant();

    const auto indices_owned = boost::irange<unsigned int>(0, n_locally_owned);
    const auto indices_relevant =
        boost::irange<unsigned int>(0, n_locally_relevant);

    const auto &sparsity = offline_data_->sparsity_pattern();

    const auto &mass_matrix = offline_data_->mass_matrix();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &norm_matrix = offline_data_->norm_matrix();
    const auto &nij_matrix = offline_data_->nij_matrix();
    const auto &bij_matrix = offline_data_->bij_matrix();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();

    const auto &boundary_normal_map = offline_data_->boundary_normal_map();
    const Number measure_of_omega = offline_data_->measure_of_omega();

    /*
     * Step 1: Compute off-diagonal d_ij, and alpha_i
     */

    {
      deallog << "        compute d_ij, and alpha_i" << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 1 compute d_ij, and alpha_i");

      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Stored thread locally: */
        Indicator<dim, Number> indicator(*offline_data_);

        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const auto U_i = gather(U, i);

          indicator.reset(U_i);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            /* Skip diagonal. */
            if (j == i)
              continue;

            const auto U_j = gather(U, j);
            indicator.add(U_j, jt);

            /* Only iterate over the subdiagonal for d_ij */
            if (j >= i)
              continue;

            const auto n_ij = gather_get_entry(nij_matrix, jt);
            const Number norm = get_entry(norm_matrix, jt);

            const auto [lambda_max, p_star, n_iterations] =
                RiemannSolver<dim, Number>::compute(U_i, U_j, n_ij);

            Number d = norm * lambda_max;

            /*
             * In case both dofs are located at the boundary we have to
             * symmetrize.
             */

            if (boundary_normal_map.count(i) != 0 &&
                boundary_normal_map.count(j) != 0) {
              const auto n_ji = gather(nij_matrix, j, i);
              auto [lambda_max_2, p_star_2, n_iterations_2] =
                  RiemannSolver<dim, Number>::compute(U_j, U_i, n_ji);
              const Number norm_2 = norm_matrix(j, i);
              d = std::max(d, norm_2 * lambda_max_2);
            }

            /* Set symmetrized off-diagonal values: */

            set_entry(dij_matrix_, jt, d);
            dij_matrix_(j, i) = d; // FIXME: Suboptimal
          }

          rho_second_variation_.local_element(i) =
              indicator.rho_second_variation();

          const Number mass = lumped_mass_matrix.diag_element(i);
          const Number hd_i = mass / measure_of_omega;
          alpha_.local_element(i) = indicator.alpha(hd_i);
        }
      };

      parallel::apply_to_subranges(
          indices_relevant.begin(), indices_relevant.end(), on_subranges, 4096);

      /* Synchronize alpha_ over all MPI processes: */
      rho_second_variation_.update_ghost_values();
      alpha_.update_ghost_values();
    }


    /*
     * Step 2: Compute diagonal of d_ij, and maximal time-step size.
     */

    std::atomic<Number> tau_max{std::numeric_limits<Number>::infinity()};

    {
      deallog << "        compute d_ii, and tau_max" << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 2 compute d_ii, and tau_max");

      const auto on_subranges = [&](auto i1, const auto i2) {
        Number tau_max_on_subrange = std::numeric_limits<Number>::infinity();

        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const Number delta_rho_i = rho_second_variation_.local_element(i);

          Number alpha_i = 0.;
          Number d_sum = 0.;
          Number rho_relaxation_numerator = 0.;
          Number rho_relaxation_denominator = 0.;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            if constexpr (smoothen_alpha_) {
              const auto m_ij = get_entry(mass_matrix, jt);
              alpha_i += m_ij * alpha_.local_element(j);
            }

            if (j == i)
              continue;

            const Number delta_rho_j = rho_second_variation_.local_element(j);
            const auto beta_ij = get_entry(betaij_matrix, jt);

            /* The numerical constant 8 is up to debate... */
            rho_relaxation_numerator +=
                8.0 * 0.5 * beta_ij * (delta_rho_i + delta_rho_j);
            rho_relaxation_denominator += beta_ij;

            d_sum -= get_entry(dij_matrix_, jt);
          }

          if constexpr (smoothen_alpha_) {
            const Number m_i = lumped_mass_matrix.diag_element(i);
            alpha_.local_element(i) = alpha_i / m_i;
          }

          rho_relaxation_.local_element(i) =
              std::abs(rho_relaxation_numerator / rho_relaxation_denominator);

          dij_matrix_.diag_element(i) = d_sum;

          const Number mass = lumped_mass_matrix.diag_element(i);
          const Number tau = cfl_update_ * mass / (-2. * d_sum);
          tau_max_on_subrange = std::min(tau_max_on_subrange, tau);
        }

        Number current_tau_max = tau_max.load();
        while (current_tau_max > tau_max_on_subrange &&
               !tau_max.compare_exchange_weak(current_tau_max,
                                              tau_max_on_subrange))
          ;
      };

      parallel::apply_to_subranges(
          indices_relevant.begin(), indices_relevant.end(), on_subranges, 4096);

      if constexpr (smoothen_alpha_) {
        /* Synchronize alpha_ over all MPI processes: */
        alpha_.update_ghost_values();
      }

      /* Synchronize tau_max over all MPI processes: */
      tau_max.store(Utilities::MPI::min(tau_max.load(), mpi_communicator_));

      AssertThrow(!std::isnan(tau_max) && !std::isinf(tau_max) && tau_max > 0.,
                  ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                             "do that. - We crashed."));

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

      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Notar bene: This bounds variable is thread local: */
        Limiter<dim, Number> limiter;

        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          /* Only iterate over locally owned subset! */
          Assert(i < n_locally_owned, ExcInternalError());

          const auto U_i = gather(U, i);
          auto U_i_new = U_i;

          const auto f_i = ProblemDescription<dim, Number>::f(U_i);
          const auto alpha_i = alpha_.local_element(i);
          const Number m_i = lumped_mass_matrix.diag_element(i);

          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const Number lambda = 1. / (size - 1.);

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

            const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                       Indicator<dim, Number>::Indicators::
                                           entropy_viscosity_commutator
                                   ? d_ij * (alpha_i + alpha_j) / 2.
                                   : d_ij * std::max(alpha_i, alpha_j);

            const auto p_ij = tau / m_i / lambda * (d_ijH - d_ij) * (U_j - U_i);

            dealii::Tensor<1, problem_dimension> U_ij_bar;

            for (unsigned int k = 0; k < problem_dimension; ++k) {
              const auto temp = (f_j[k] - f_i[k]) * c_ij;

              r_i[k] += -temp + d_ijH * (U_j - U_i)[k];
              U_ij_bar[k] = 1. / 2. * (U_i[k] + U_j[k]) - 1. / 2. * temp / d_ij;
            }

            U_i_new += tau / m_i * 2. * d_ij * U_ij_bar;

            scatter_set_entry(pij_matrix_, jt, p_ij);

            limiter.accumulate(U_i, U_j, U_ij_bar, jt);
          }

          scatter(temp_euler_, U_i_new, i);
          scatter(r_, r_i, i);

          const Number hd_i = m_i / measure_of_omega;
          const Number rho_relaxation_i = rho_relaxation_.local_element(i);
          limiter.apply_relaxation(hd_i, rho_relaxation_i);
          scatter(bounds_, limiter.bounds(), i);
        }
      };

      /* Only iterate over locally owned subset! */
      parallel::apply_to_subranges(
          indices_owned.begin(), indices_owned.end(), on_subranges, 4096);

      /* Synchronize r_ over all MPI processes: */
      for (auto &it : r_)
        it.update_ghost_values();
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

      const auto on_subranges = [&](auto i1, const auto i2) {
        for (const auto i : boost::make_iterator_range(i1, i2)) {

          /* Only iterate over locally owned subset! */
          Assert(i < n_locally_owned, ExcInternalError());

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const Number m_i = lumped_mass_matrix.diag_element(i);
          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const Number lambda = 1. / (size - 1.);

          const auto r_i = gather(r_, i);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            const auto b_ij = get_entry(bij_matrix, jt);
            const auto b_ji = bij_matrix(j, i); // FIXME: Suboptimal
            auto p_ij = gather_get_entry(pij_matrix_, jt);

            const auto r_j = gather(r_, j);

            p_ij += tau / m_i / lambda * (b_ij * r_j - b_ji * r_i);
            scatter_set_entry(pij_matrix_, jt, p_ij);
          }
        }
      };

      parallel::apply_to_subranges(
          indices_owned.begin(), indices_owned.end(), on_subranges, 4096);
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

        const auto on_subranges = [&](auto i1, const auto i2) {
          for (const auto i : boost::make_iterator_range(i1, i2)) {

            /* Only iterate over locally owned subset! */
            Assert(i < n_locally_owned, ExcInternalError());

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
            indices_owned.begin(), indices_owned.end(), on_subranges, 4096);
      }

      /*
       * And symmetrize l_ij:
       */

      {
        deallog << "        symmetrize l_ij" << std::endl;
        TimerOutput::Scope time(computing_timer_,
                                "time_step - 6 symmetrize l_ij");

        if (Utilities::MPI::n_mpi_processes(mpi_communicator_) > 1)
          lij_matrix_communicator_.synchronize();

        {
          const auto on_subranges = [&](auto i1, const auto i2) {
            for (const auto i : boost::make_iterator_range(i1, i2)) {
              for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
                const auto j = jt->column();

                if (j >= i)
                  continue;

                auto l_ij = get_entry(lij_matrix_, jt);
                auto &l_ji = lij_matrix_(j, i); // FIXME: Suboptimal

                const Number min = std::min(l_ij, l_ji);
                l_ji = min;
                set_entry(lij_matrix_, jt, min);
              }
            }
          };

          parallel::apply_to_subranges(indices_relevant.begin(),
                                       indices_relevant.end(),
                                       on_subranges,
                                       4096);
        }
      }

      /*
       * Step 7: Perform high-order update:
       *
       *   High-order update: += l_ij * lambda * P_ij
       */

      {
        deallog << "        high-order update" << std::endl;
        TimerOutput::Scope time(computing_timer_,
                                "time_step - 7 high-order update");

        const auto on_subranges = [&](auto i1, const auto i2) {
          for (const auto i : boost::make_iterator_range(i1, i2)) {

            /* Only iterate over locally owned subset */
            Assert(i < n_locally_owned, ExcInternalError());

            /* Skip constrained degrees of freedom */
            if (++sparsity.begin(i) == sparsity.end(i))
              continue;

            auto U_i_new = gather(temp_euler_, i);

            const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
            const Number lambda = 1. / (size - 1.);

            for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
              auto p_ij = gather_get_entry(pij_matrix_, jt);
              const auto l_ij = get_entry(lij_matrix_, jt);
              U_i_new += l_ij * lambda * p_ij;
              p_ij *= (1 - l_ij);
              scatter_set_entry(pij_matrix_, jt, p_ij);
            }

            scatter(temp_euler_, U_i_new, i);
          }
        };

        parallel::apply_to_subranges(
            indices_owned.begin(), indices_owned.end(), on_subranges, 4096);
      }
    } /* limiter_iter_ */

    /*
     * Step 8: Fix boundary:
     */

    {
      deallog << "        fix up boundary states" << std::endl;
      TimerOutput::Scope time(computing_timer_,
                              "time_step - 8 fix boundary states");

      const auto on_subranges = [&](const auto it1, const auto it2) {
        for (auto it = it1; it != it2; ++it) {

          const auto i = it->first;

          /* Only iterate over locally owned subset */
          if (i >= n_locally_owned)
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
  Number TimeStep<dim, Number>::ssprk_step(vector_type &U, Number t)
  {
    deallog << "TimeStep<dim, Number>::ssprk_step()" << std::endl;

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
                           "Insufficient CFL condition."));

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(1. / 4., 3. / 4., temp_ssprk_[k]);


    // Step 3: U_new = 1/3 U_old + 2/3 (U2 + tau L(U2))

    const Number tau_3 = euler_step(U, t, tau_1);

    AssertThrow(ratio * tau_3 >= tau_1,
                ExcMessage("Problem performing SSP RK(3) time step: "
                           "Insufficient CFL condition."));

    for (unsigned int k = 0; k < problem_dimension; ++k)
      U[k].sadd(2. / 3., 1. / 3., temp_ssprk_[k]);

    return tau_1;
  }


  template <int dim, typename Number>
  Number TimeStep<dim, Number>::step(vector_type &U, Number t)
  {
    deallog << "TimeStep<dim, Number>::step()" << std::endl;

    if (use_ssprk3_) {
      return ssprk_step(U, t);
    } else {
      return euler_step(U, t);
    }
  }


} /* namespace grendel */

#endif /* TIME_STEP_TEMPLATE_H */
