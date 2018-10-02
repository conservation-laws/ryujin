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
  }


  template <int dim>
  void TimeStep<dim>::setup()
  {
    deallog << "TimeStep<dim>::setup()" << std::endl;
    TimerOutput::Scope t(computing_timer_, "time_step - setup scratch space");

    const auto &locally_relevant = offline_data_->locally_relevant();
    const auto &sparsity_pattern = offline_data_->sparsity_pattern();

    f_i_.resize(locally_relevant.n_elements());
    dij_matrix_.reinit(sparsity_pattern);
  }


  template <int dim>
  double TimeStep<dim>::euler_step(vector_type &U_new, const vector_type &U_old)
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

    /* Clear out matrix entries: */
    dij_matrix_ = 0.;

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
        for (auto it = it1; it != it2; ++it) {
          /* Determine absolute position of it in vector: */
          const unsigned int pos_i = std::distance(f_i_.begin(), it);
          /* Determine global index i from  pos_i: */
          const auto i = locally_relevant.nth_index_in_set(pos_i);

          const auto U_i = gather(U_old, i);

          /* Populate f_i_: */
          *it = problem_description_->f(U_i);

          /* Populate off-diagonal dij_: */
          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            /* Iterate over subdiagonal */
            if (j >= i)
              continue;

            const auto U_j = gather(U_old, j);
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

            {
              // FIXME explain this
              const auto global_index = jt->global_index();
              const typename SparseMatrix<double>::iterator matrix_iterator(
                  &dij_matrix_, global_index);
              matrix_iterator->value() = d;
            }
            dij_matrix_(j, i) = d; // FIXME index access suboptimal
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

      const auto on_subranges = [&](const auto it1, const auto it2) {
        double tau_max_on_subrange = std::numeric_limits<double>::infinity();

        /* [it1, it2) is an iterator range over f_i_ */
        for (auto it = it1; it != it2; ++it) {
          /* Determine absolute position of it in vector: */
          const unsigned int pos_i = std::distance(f_i_.begin(), it);
          /* Determine global index i from  pos_i: */
          const auto i = locally_relevant.nth_index_in_set(pos_i);

          /* Compute sum of d_ijs for index i: */
          double d_sum = 0.;
          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();
            if (j == i)
              continue;

            d_sum -= get_entry(dij_matrix_, jt);
          }

          dij_matrix_.diag_element(i) = d_sum;

          const double cfl = problem_description_->cfl_constant();
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
    }

    /*
     * Step 3: Perform update *yay*
     */

    {
      deallog << "        perform time-step" << std::endl;
      TimerOutput::Scope t(computing_timer_, "time_step - perform time-step");

      const auto on_subranges = [&](const auto it1, const auto it2) {
        /* [it1, it2) is an iterator range over f_i_ */
        for (auto it = it1; it != it2; ++it) {
          const unsigned int pos_i = std::distance(f_i_.begin(), it);
          const auto i = locally_relevant.nth_index_in_set(pos_i);

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          const auto U_i = gather(U_old, i);

          dealii::Tensor<1, problem_dimension> Unew_i = U_i;

          const auto f_i = *it; // This is f_i_[pos_i]
          const double m_i = lumped_mass_matrix.diag_element(i);

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            if (j == i)
              continue;

            const auto U_j = gather(U_old, j);

            const unsigned int pos_j = locally_relevant.index_within_set(j);
            const auto f_j = f_i_[pos_j];

            const auto c_ij = gather_get_entry(cij_matrix, jt);
            const auto d_ij = get_entry(dij_matrix_, jt);

            for (unsigned int k = 0; k < problem_dimension; ++k)
              Unew_i[k] +=
                  tau_max / m_i *
                  (-(f_j[k] - f_i[k]) * c_ij + d_ij * (U_j[k] - U_i[k]));
          }

          /*
           * Treat boundary points:
           */

          const auto bnm_it = boundary_normal_map.find(i);
          if (bnm_it != boundary_normal_map.end()) {
            const auto [normal, id] = bnm_it->second;

            /* On boundray 1 we reflect: */
            if (id == 1) {
              auto m = ProblemDescription<dim>::momentum_vector(Unew_i);
              m -= 1. * (m * normal) * normal;
              for (unsigned int i = 0; i < dim; ++i)
                Unew_i[i + 1] = m[i];
            }
          }

          /*
           * And write to global vector:
           */

          scatter(U_new, Unew_i, i);
        }
      };

      parallel::apply_to_subranges(
          f_i_.begin(), f_i_.end(), on_subranges, 4096);

      /* Synchronize U_new over all MPI processes: */
      for (auto &it : U_new)
        it.update_ghost_values();

      return tau_max;
    }
  }


} /* namespace grendel */

#endif /* TIME_STEP_TEMPLATE_H */
