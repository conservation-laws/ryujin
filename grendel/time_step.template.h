#ifndef TIME_STEP_TEMPLATE_H
#define TIME_STEP_TEMPLATE_H

#include "helper.h"
#include "time_step.h"

#include <boost/range/irange.hpp>

#include <atomic>

namespace grendel
{
  using namespace dealii;


  template <int dim>
  TimeStep<dim>::TimeStep(
      const MPI_Comm &mpi_communicator,
      dealii::TimerOutput &computing_timer,
      const grendel::OfflineData<dim> &offline_data,
      const grendel::ProblemDescription<dim> &problem_description,
      const grendel::RiemannSolver<dim> &riemann_solver,
      const grendel::HighOrder<dim> &high_order,
      const std::string &subsection /*= "TimeStep"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , problem_description_(&problem_description)
      , riemann_solver_(&riemann_solver)
      , high_order_(&high_order)
  {
    use_ssprk_ = false;
    add_parameter(
        "use SSP RK",
        use_ssprk_,
        "If enabled, use SSP RK(3) instead of the forward Euler scheme.");
  }


  template <int dim>
  void TimeStep<dim>::prepare()
  {
    deallog << "TimeStep<dim>::prepare()" << std::endl;
    TimerOutput::Scope t(computing_timer_, "time_step - prepare scratch space");

    /* Initialize (global) vectors: */

    const auto &locally_owned = offline_data_->locally_owned();
    const auto &locally_relevant = offline_data_->locally_relevant();

    auto &exemplar = alpha_;
    exemplar.reinit(locally_owned, locally_relevant, mpi_communicator_);

    alpha_second_.reinit(exemplar);
    delta_.reinit(exemplar);

    for (auto &it : temp_euler_)
      it.reinit(exemplar);

    for (auto &it : temp_ssprk_)
      it.reinit(exemplar);

    for (auto &it : r_)
      it.reinit(exemplar);

    for (auto &it : bounds_)
      it.reinit(exemplar);

    /* Initialize local matrices */

    const auto &sparsity_pattern = offline_data_->sparsity_pattern();

    for (auto &it : pij_matrix_)
      it.reinit(sparsity_pattern);

    dij_matrix_.reinit(sparsity_pattern);
    lij_matrix_.reinit(sparsity_pattern);
  }


  template <int dim>
  double TimeStep<dim>::euler_step(vector_type &U, double tau)
  {
    deallog << "TimeStep<dim>::euler_step()" << std::endl;

    const auto &locally_relevant = offline_data_->locally_relevant();
    const auto &locally_owned = offline_data_->locally_owned();
    const auto &sparsity = offline_data_->sparsity_pattern();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &norm_matrix = offline_data_->norm_matrix();
    const auto &nij_matrix = offline_data_->nij_matrix();
    const auto &bij_matrix = offline_data_->bij_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    const auto &boundary_normal_map = offline_data_->boundary_normal_map();

    const auto indices =
        boost::irange<unsigned int>(0, locally_relevant.n_elements());


    /*
     * Step 1: Compute off-diagonal d_ij, also compute smoothness
     *         indicators \alpha_i and \Delta_i:
     *
     *   \alpha_i = \|\sum_j beta_ij (s(U_i) - s(U_j)) \|
     *                / \sum_j \| \beta_ij (s(U_i) - s(U_j)) \|,
     *
     *   \Delta_i = 1. / m_i \sum \beta_ij (- s(U_j)),
     *
     *   where s(.) is a suitable function used for the smoothness
     *   indicator and \beta_ij is the stiffness matrix.
     */

    {
      deallog << "        compute d_ij, alpha_i, and delta_i" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - 1 compute d_ij, alpha_i, and delta_i");

      alpha_.zero_out_ghosts();
      delta_.zero_out_ghosts();

      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {

          const auto i = *it;
          const auto U_i = gather(U, i);
          const auto indicator_i = high_order_->smoothness_indicator(U, i);
          const double m_i = lumped_mass_matrix.diag_element(i);

          double numerator = 0.;
          double denominator = 0.;
          double delta = 0.;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            /*
             * Skip diagonal elements:
             */

            if (j == i)
              continue;

            const auto beta_ij = get_entry(betaij_matrix, jt);
            const auto indicator_j = high_order_->smoothness_indicator(U, j);

            numerator += beta_ij * (indicator_i - indicator_j);
            delta -= beta_ij * indicator_j;

            constexpr double eps_ = 1.e-7;
            denominator +=
                std::abs(beta_ij) * std::abs(indicator_i - indicator_j) +
                eps_ * std::abs(indicator_j);

            /*
             * Only iterate over the subdiagonal for d_ij
             */

            if (j > i)
              continue;

            const auto U_j = gather(U, j);
            const auto n_ij = gather_get_entry(nij_matrix, jt);
            const double norm = get_entry(norm_matrix, jt);

            const auto [lambda_max, p_star, n_iterations] =
                riemann_solver_->compute(U_i, U_j, n_ij);

            double d = norm * lambda_max;

            /*
             * In case both dofs are located at the boundary we have to
             * symmetrize.
             */

            if (boundary_normal_map.count(i) != 0 &&
                boundary_normal_map.count(j) != 0) {
              const auto n_ji = gather(nij_matrix, j, i);
              auto [lambda_max_2, p_star_2, n_iterations_2] =
                  riemann_solver_->compute(U_j, U_i, n_ji);
              const double norm_2 = norm_matrix(j, i);
              d = std::max(d, norm_2 * lambda_max_2);
            }

            /* Set symmetrized off-diagonal values: */

            set_entry(dij_matrix_, jt, d);
            dij_matrix_(j, i) = d; // FIXME: Suboptimal
          }

          alpha_[i] = std::pow(std::abs(numerator) / denominator,
                               HighOrder<dim>::smoothness_power);

          delta_[i] = delta / m_i;
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);

      /* Synchronize alpha_ over all MPI processes: */
      alpha_.update_ghost_values();

      /* Synchronize delta_ over all MPI processes: */
      delta_.update_ghost_values();
    }


    /*
     * Step 2: Compute diagonal of d_ij maximal time-step size, and
     *         alpha_second_i:
     */

    std::atomic<double> tau_max{std::numeric_limits<double>::infinity()};

    {
      deallog << "        compute d_ii, tau_max, and alpha_second_i"
              << std::endl;
      TimerOutput::Scope t(
          computing_timer_,
          "time_step - 2 compute d_ii, tau_max, and alpha_second_i");

      alpha_second_.zero_out_ghosts();

      const auto on_subranges = [&](auto i1, const auto i2) {
        double tau_max_on_subrange = std::numeric_limits<double>::infinity();
        const double cfl = problem_description_->cfl_update();

        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {

          const auto i = *it;

          double d_sum = 0.;

          double delta_max = std::numeric_limits<double>::min();
          double delta_min = std::numeric_limits<double>::max();

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            const auto delta_j = delta_[j];
            delta_max = std::max(delta_max, delta_j);
            delta_min = std::min(delta_min, delta_j);

            if (j == i)
              continue;

            d_sum -= get_entry(dij_matrix_, jt);
          }

          alpha_second_[i] = std::abs(delta_max - delta_min) /
                             (std::abs(delta_max) + std::abs(delta_min));

          dij_matrix_.diag_element(i) = d_sum;

          const double mass = lumped_mass_matrix.diag_element(i);
          const double tau = cfl * mass / (-2. * d_sum);
          tau_max_on_subrange = std::min(tau_max_on_subrange, tau);
        }

        double current_tau_max = tau_max.load();
        while (current_tau_max > tau_max_on_subrange &&
               !tau_max.compare_exchange_weak(current_tau_max,
                                              tau_max_on_subrange))
          ;
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);

      /* Synchronize alpha_second_ over all MPI processes: */
      alpha_second_.update_ghost_values();

      /* Synchronize tau_max over all MPI processes: */
      tau_max.store(Utilities::MPI::min(tau_max.load(), mpi_communicator_));

      AssertThrow(!std::isnan(tau_max) && !std::isinf(tau_max) && tau_max > 0.,
                  ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                             "do that. - We crashed."));

      deallog << "        computed tau_max = " << tau_max << std::endl;
    }

    tau = tau == 0 ? tau_max.load() : tau;
    deallog << "        perform time-step with tau = " << tau << std::endl;


    /*
     * Step 3: Low-order update, also compute limiter bounds, R_i and first
     *         part of P_ij
     *
     *   \bar U_ij = 1/2 d_ij^L (U_i + U_j) - 1/2 (f_j - f_i) c_ij
     *
     *        R_i = \sum_j - c_ij f_j + d_ij^H (U_j - U_i)
     *
     *        P_ij = tau / m_i / lambda (d_ij^H - d_ij^L) (U_i + U_j) + [...]
     *
     *   Low-order update: += tau / m_i * 2 d_ij^L (\bar U_ij - U_i)
     */

    {
      deallog << "        low-order update, limiter bounds, r_i, and p_ij"
              << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - 3 low-order update, limiter bounds, "
                           "compute r_i, and p_ij (1)");

      const auto on_subranges = [&](auto i1, const auto i2) {

        /* Notar bene: This bounds variable is thread local: */
        typename HighOrder<dim>::Bounds bounds;

        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {

          const auto i = *it;

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          const auto U_i = gather(U, i);
          auto U_i_new = U_i;

          const auto f_i = problem_description_->f(U_i);
          const auto alpha_i = alpha_[i]; //std::min(alpha_[i], alpha_second_[i]);
          const double m_i = lumped_mass_matrix.diag_element(i);

          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const double lambda = 1. / (size - 1.);

          rank1_type r_i;

          /* Clear bounds: */
          high_order_->reset(bounds);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {

            const auto j = jt->column();
            const auto U_j = gather(U, j);
            const auto f_j = problem_description_->f(U_j);
            const auto alpha_j = alpha_[j]; //std::min(alpha_[j], alpha_second_[j]);

            const auto c_ij = gather_get_entry(cij_matrix, jt);
            const auto d_ij = get_entry(dij_matrix_, jt);

            const auto d_ijH = d_ij * std::max(alpha_i, alpha_j);

            const auto p_ij = tau / m_i / lambda * (d_ijH - d_ij) * (U_j - U_i);

            dealii::Tensor<1, problem_dimension> U_ij_bar;

            for (unsigned int k = 0; k < problem_dimension; ++k) {
              const auto temp = c_ij * (f_j[k] - f_i[k]);

              r_i[k] += -temp + d_ijH * (U_j - U_i)[k];
              U_ij_bar[k] = 1. / 2. * (U_i[k] + U_j[k]) - 1. / 2. * temp / d_ij;
            }

            U_i_new += tau / m_i * 2. * d_ij * U_ij_bar;

            scatter_set_entry(pij_matrix_, jt, p_ij);

            high_order_->accumulate(bounds, U_ij_bar);
          }

          scatter(temp_euler_, U_i_new, i);

          scatter(r_, r_i, i);

          scatter(bounds_, bounds, i);
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);

      /* Synchronize r_ over all MPI processes: */
      for (auto &it : r_)
        it.update_ghost_values();
    }


    /*
     * Step 4: Compute second part of P_ij, and compute l_ij:
     *
     *        P_ij = [...] + tau / m_i / lambda (b_ij R_j - b_ji R_i)
     */

    if constexpr (order_ == Order::second_order) {
      deallog << "        compute p_ij and l_ij" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - 4 compute p_ij (2), and l_ij");

      const auto on_subranges = [&](auto i1, const auto i2) {

        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {

          const auto i = *it;

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          const auto bounds = gather_array(bounds_, i);
          const auto U_i_new = gather(temp_euler_, i);

          const double m_i = lumped_mass_matrix.diag_element(i);
          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const double lambda = 1. / (size - 1.);

//           const auto r_i = gather(r_, i);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            auto p_ij = gather_get_entry(pij_matrix_, jt);

            const auto j = jt->column();
//             const auto b_ij = get_entry(bij_matrix, jt);
//             const auto b_ji = bij_matrix(j, i); // FIXME: Suboptimal

//             const auto r_j = gather(r_, j);

//             p_ij += tau / m_i / lambda * (b_ij * r_j - b_ji * r_i);
            scatter_set_entry(pij_matrix_, jt, p_ij);

            const auto l_ij = high_order_->limit(bounds, U_i_new, p_ij);
            set_entry(lij_matrix_, jt, l_ij);
          }
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);
    }


    /* And symmetrize l_ij: */

    if constexpr (order_ == Order::second_order) {
      deallog << "        symmetrize l_ij" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - 4 symmetrize l_ij");

      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {
          const auto i = *it;
          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();
            if (j >= i)
              continue;
            auto l_ij = get_entry(lij_matrix_, jt);
            auto &l_ji = lij_matrix_(j, i); // FIXME: Suboptimal

            const double min = std::min(l_ij, l_ji);
            l_ji = min;
            set_entry(lij_matrix_, jt, min);
          }
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);
    }


    /*
     * Step 5: Perform high-order update:
     *
     *   High-order update: += l_ij * lambda * P_ij
     */

    if constexpr (order_ == Order::second_order) {
      deallog << "        high-order update" << std::endl;
      TimerOutput::Scope t(computing_timer_, "time_step - 5 high-order update");

      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {

          const auto i = *it;

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          auto U_i_new = gather(temp_euler_, i);

          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const double lambda = 1. / (size - 1.);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto p_ij = gather_get_entry(pij_matrix_, jt);
            const auto l_ij = get_entry(lij_matrix_, jt);
            U_i_new += l_ij * lambda * p_ij;
          }

          scatter(temp_euler_, U_i_new, i);
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);
    }

    /*
     * Step 6: Fix boundary:
     */

    {
      deallog << "        fix up boundary states" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - 6 fix boundary states");

      const auto on_subranges = [&](const auto it1, const auto it2) {
        for (auto it = it1; it != it2; ++it) {
          const auto i = it->first;
          const auto &[normal, id, position] = it->second;

          if (!locally_owned.is_element(i))
            continue;

          auto U_i = gather(temp_euler_, i);

          /* On boundray 1 remove the normal component of the momentum: */

          if (id == 1) {
            auto m = ProblemDescription<dim>::momentum_vector(U_i);
            m -= 1. * (m * normal) * normal;
            for (unsigned int k = 0; k < dim; ++k)
              U_i[k + 1] = m[k];
          }

          /* On boundray 2 enforce initial conditions: */

          if (id == 2) {
            U_i = problem_description_->initial_state(position, 0.);
          }

          scatter(temp_euler_, U_i, i);
        }
      };

      //FIXME: This is currently not parallel:
      on_subranges(boundary_normal_map.begin(), boundary_normal_map.end());
    }

    /* Synchronize temp over all MPI processes: */
    for (auto &it : temp_euler_)
      it.update_ghost_values();

    /* And finally update the result: */
    U.swap(temp_euler_);

    return tau_max;
  }


  template <int dim>
  double TimeStep<dim>::ssprk_step(vector_type &U)
  {
    deallog << "TimeStep<dim>::ssprk_step()" << std::endl;

    /* This also copies ghost elements: */
    for (unsigned int i = 0; i < problem_dimension; ++i)
      temp_ssprk_[i] = U[i];

    // Step 1: U1 = U_old + tau * L(U_old)

    const double tau_1 = euler_step(U);

    // Step 2: U2 = 3/4 U_old + 1/4 (U1 + tau L(U1))

    const double tau_2 = euler_step(U, tau_1);

    const double ratio =
        problem_description_->cfl_max() / problem_description_->cfl_update();
    AssertThrow(ratio * tau_2 >= tau_1,
                ExcMessage("Problem performing SSP RK(3) time step: "
                           "Insufficient CFL condition."));

    for (unsigned int i = 0; i < problem_dimension; ++i)
      U[i].sadd(1. / 4., 3. / 4., temp_ssprk_[i]);


    // Step 3: U_new = 1/3 U_old + 2/3 (U2 + tau L(U2))

    const double tau_3 = euler_step(U, tau_1);

    AssertThrow(ratio * tau_3 >= tau_1,
                ExcMessage("Problem performing SSP RK(3) time step: "
                           "Insufficient CFL condition."));

    for (unsigned int i = 0; i < problem_dimension; ++i)
      U[i].sadd(2. / 3., 1. / 3., temp_ssprk_[i]);

    return tau_1;
  }


  template <int dim>
  double TimeStep<dim>::step(vector_type &U)
  {
    deallog << "TimeStep<dim>::step()" << std::endl;

    return use_ssprk_ ? ssprk_step(U) : euler_step(U);
  }




} /* namespace grendel */

#endif /* TIME_STEP_TEMPLATE_H */
