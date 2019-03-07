#ifndef TIME_STEP_TEMPLATE_H
#define TIME_STEP_TEMPLATE_H

#include "helper.h"
#include "time_step.h"

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
      const std::string &subsection /*= "TimeStep"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , problem_description_(&problem_description)
      , riemann_solver_(&riemann_solver)
  {
    use_ssprk_ = false;
    add_parameter(
        "use SSP RK",
        use_ssprk_,
        "If enabled, use SSP RK(3) instead of the forward Euler scheme.");

    use_smoothness_indicator_ = false;
    add_parameter("use smoothness indicator",
                  use_smoothness_indicator_,
                  "If enabled, use a smoothness indicator for the high-order "
                  "approximation.");

    use_limiter_ = false;
    add_parameter(
        "use limiter",
        use_limiter_,
        "If enabled, use a convex limiter for the high-order approximation..");

    smoothness_index_ = 0;
    add_parameter("smoothness index",
                  smoothness_index_,
                  "Use the corresponding component of the state vector for the "
                  "smoothness indicator");

    smoothness_power_ = 3;
    add_parameter("smoothness power",
                  smoothness_power_,
                  "Sets the exponent for the smoothness indicator");

    eps_ = std::numeric_limits<double>::epsilon();
  }


  template <int dim>
  void TimeStep<dim>::prepare()
  {
    deallog << "TimeStep<dim>::prepare()" << std::endl;
    TimerOutput::Scope t(computing_timer_, "time_step - prepare scratch space");

    const auto &locally_owned = offline_data_->locally_owned();
    const auto &locally_relevant = offline_data_->locally_relevant();
    const auto &sparsity_pattern = offline_data_->sparsity_pattern();

    f_i_.resize(locally_relevant.n_elements());
    dij_matrix_.reinit(sparsity_pattern);

    if (use_smoothness_indicator_) {
      alpha_i_.reinit(locally_relevant.n_elements());
      for (auto &it : pij_matrix_)
        it.reinit(sparsity_pattern);
    }

    if (use_limiter_) {
      for (auto &it : uij_bar_matrix_)
        it.reinit(sparsity_pattern);
      lij_matrix_.reinit(sparsity_pattern);
    }

    temp_euler_[0].reinit(locally_owned, locally_relevant, mpi_communicator_);
    for (auto &it : temp_euler_)
      it.reinit(temp_euler_[0]);

    for (auto &it : temp_ssprk_)
      it.reinit(temp_euler_[0]);
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
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_normal_map = offline_data_->boundary_normal_map();

    /*
     * Step 1: Compute off-diagonal d_ij and f_i:
     */

    {
      deallog << "        compute d_ij and f_i" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - compute d_ij and f_i");

      /*
       * FIXME Workaround: IndexSet does not have an iterator with
       * complete operater arithmetic (that is required for tbb). We
       * iterate over the local vector f_i_ instead and do a bit of index
       * juggling...
       */

      const auto on_subranges = [&](const auto it1, const auto it2) {
        /* [it1, it2) is an iterator range over f_i_ */

        /* Create an iterator for the index set: */
        const unsigned int pos = std::distance(f_i_.begin(), it1);
        auto set_iterator =
            locally_relevant.at(locally_relevant.nth_index_in_set(pos));

        for (auto it = it1; it != it2; ++it, ++set_iterator) {
          const auto i = *set_iterator;
          const auto U_i = gather(U, i);

          /* Populate f_i_: */
          *it = problem_description_->f(U_i);

          /* Populate off-diagonal dij_: */
          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            /* Iterate over subdiagonal */
            if (j >= i)
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
        }
      };

      parallel::apply_to_subranges(
          f_i_.begin(), f_i_.end(), on_subranges, 4096);
    }

    /*
     * Step 2: Compute diagonal of d_ij and maximal time-step size:
     */

    std::atomic<double> tau_max{std::numeric_limits<double>::infinity()};

    {
      deallog << "        compute diagonal and tau_max" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - compute diagonal and tau_max");

      /* [it1, it2) is an iterator range over f_i_ */
      const auto on_subranges = [&](const auto it1, const auto it2) {
        double tau_max_on_subrange = std::numeric_limits<double>::infinity();

        /* Create an iterator for the index set: */
        const unsigned int pos = std::distance(f_i_.begin(), it1);
        auto set_iterator =
            locally_relevant.at(locally_relevant.nth_index_in_set(pos));

        for (auto it = it1; it != it2; ++it, ++set_iterator) {
          const auto i = *set_iterator;

          /* Compute sum of d_ijs for index i: */
          double d_sum = 0.;
          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();
            if (j == i)
              continue;

            d_sum -= get_entry(dij_matrix_, jt);
          }

          dij_matrix_.diag_element(i) = d_sum;

          const double cfl = problem_description_->cfl_update();
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
          f_i_.begin(), f_i_.end(), on_subranges, 4096);

      /* Synchronize tau_max over all MPI processes: */
      tau_max.store(Utilities::MPI::min(tau_max.load(), mpi_communicator_));

      deallog << "        computed tau_max = " << tau_max << std::endl;
    }

    /*
     * Step 3: Compute smoothness indicator
     *
     * If enabled we compute the smoothness indicator
     *
     *   \alpha_i = \|\sum_j U_i[s] - U_j[s] \| / \sum_j \| U_i[s] - U_j[s] \|
     */

    if (use_smoothness_indicator_) {
      deallog << "        compute smoothness indicator" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - compute smoothness indicator");

      /* [it1, it2) is an iterator range over f_i_ */
      const auto on_subranges = [&](const auto it1, const auto it2) {

        /* Create an iterator for the index set: */
        const unsigned int pos = std::distance(f_i_.begin(), it1);
        auto set_iterator =
            locally_relevant.at(locally_relevant.nth_index_in_set(pos));

        for (auto it = it1; it != it2; ++it, ++set_iterator) {
          const auto i = *set_iterator;
          const auto U_is = U[smoothness_index_][i];

          double numerator = 0.;
          double denominator = 0.;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            if (j == i)
              continue;

            const auto U_js = U[smoothness_index_][j];

            numerator += (U_js - U_is);
            denominator += std::abs(U_js - U_is) + eps_ * std::abs(U_js);
          }

          const unsigned int pos_i = locally_relevant.index_within_set(i);
          alpha_i_[pos_i] =
              std::pow(std::abs(numerator) / denominator, smoothness_power_);
        }
      };

      parallel::apply_to_subranges(
          f_i_.begin(), f_i_.end(), on_subranges, 4096);
    }

    tau = tau == 0 ? tau_max.load() : tau;
    deallog << "        perform time-step with tau = " << tau << std::endl;

    /*
     * Step 4: Perform low-order update and compute limiter:
     *
     *   \bar U_ij = 1/2 d_ij^L (U_i + U_j) - 1/2 (f_j - f_i) c_ij
     *        P_ij = tau / m_i / lambda (d_ij^H - d_ij^L) + [...]
     *
     *   Low-order update: += tau / m_i * 2 d_ij^L (\bar U_ij - U_i)
     */

    {
      TimerOutput::Scope t(computing_timer_,
                           "time_step - low-order update and limiter");

      const auto on_subranges = [&](const auto it1, const auto it2) {
        /* [it1, it2) is an iterator range over f_i_ */

        /* Create an iterator for the index set: */
        const unsigned int pos = std::distance(f_i_.begin(), it1);
        auto set_iterator =
            locally_relevant.at(locally_relevant.nth_index_in_set(pos));

        for (auto it = it1; it != it2; ++it, ++set_iterator) {
          const auto i = *set_iterator;

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          const auto U_i = gather(U, i);
          const double m_i = lumped_mass_matrix.diag_element(i);
          const auto f_i = *it; // This is f_i_[pos_i]
          const unsigned int pos_i = locally_relevant.index_within_set(i);
          const auto alpha_i = use_smoothness_indicator_ ? alpha_i_[pos_i] : 1.;

          dealii::Tensor<1, problem_dimension> Unew_i = U_i;

          const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
          const double lambda = 1. / (size - 1.);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i);
               ++jt) {
            const auto j = jt->column();

            const auto U_j = gather(U, j);

            const unsigned int pos_j = locally_relevant.index_within_set(j);
            const auto f_j = f_i_[pos_j];
            const auto alpha_j =
                use_smoothness_indicator_ ? alpha_i_[pos_j] : 1.;

            const auto c_ij = gather_get_entry(cij_matrix, jt);
            const auto d_ij = get_entry(dij_matrix_, jt);

            dealii::Tensor<1, problem_dimension> U_ij_bar;
            for (unsigned int k = 0; k < problem_dimension; ++k)
              U_ij_bar[k] = 1. / 2. * (U_i[k] + U_j[k]) -
                            1. / 2. * (f_j[k] - f_i[k]) * c_ij / d_ij;

            Unew_i += tau / m_i * 2. * d_ij * (U_ij_bar - U_i);

            if (use_limiter_) {
              scatter_set_entry(uij_bar_matrix_, jt, U_ij_bar);
            }

            if (use_smoothness_indicator_ || use_limiter_) {
              const auto p_ij = tau / m_i / lambda * d_ij *
                                (std::max(alpha_i, alpha_j) - 1.) * (U_j - U_i);
              scatter_set_entry(pij_matrix_, jt, p_ij);
            }
          }

          //FIXME: Call limiter to compute l_ij

          scatter(temp_euler_, Unew_i, i);
        }
      };

      parallel::apply_to_subranges(
          f_i_.begin(), f_i_.end(), on_subranges, 4096);
    }

    /*
     * Step 5: Perform high-order and fix boundary:
     *
     *   P_ij = tau / m_i / lambda (d_ij^H - d_ij^L) + [...]
     *
     *   High-order update: += l_ij * lambda * P_ij
     */

    {
      TimerOutput::Scope t(computing_timer_,
                           "time_step - high-order update and boundary");

      const auto on_subranges = [&](const auto it1, const auto it2) {
        /* [it1, it2) is an iterator range over f_i_ */

        /* Create an iterator for the index set: */
        const unsigned int pos = std::distance(f_i_.begin(), it1);
        auto set_iterator =
            locally_relevant.at(locally_relevant.nth_index_in_set(pos));

        for (auto it = it1; it != it2; ++it, ++set_iterator) {
          const auto i = *set_iterator;

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          auto Unew_i = gather(temp_euler_, i);

          if (use_smoothness_indicator_) {

            const auto size = std::distance(sparsity.begin(i), sparsity.end(i));
            const double lambda = 1. / (size - 1.);

            for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
              const auto p_ij = gather_get_entry(pij_matrix_, jt);
              const auto l_ij = use_limiter_ ? get_entry(lij_matrix_, jt) : 1.;
              Unew_i += l_ij * lambda * p_ij;
            }
          }

          const auto bnm_it = boundary_normal_map.find(i);
          if (bnm_it != boundary_normal_map.end()) {
            const auto [normal, id] = bnm_it->second;

            /* On boundray 1 remove the normal component of the momentum: */
            if (id == 1) {
              auto m = ProblemDescription<dim>::momentum_vector(Unew_i);
              m -= 1. * (m * normal) * normal;
              for (unsigned int i = 0; i < dim; ++i)
                Unew_i[i + 1] = m[i];
            }
          }

          scatter(temp_euler_, Unew_i, i);
        }
      };

      parallel::apply_to_subranges(
          f_i_.begin(), f_i_.end(), on_subranges, 4096);
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
