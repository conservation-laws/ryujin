#ifndef TIME_STEP_TEMPLATE_H
#define TIME_STEP_TEMPLATE_H

#include "time_step.h"

namespace grendel
{
  using namespace dealii;


  /*
   * It's magic
   */
  template <typename T1, std::size_t k, typename T2>
  inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, k>
  gather(const std::array<T1, k> &U, const T2 i)
  {
    dealii::Tensor<1, k> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j][i];
    return result;
  }


  /*
   * It's magic
   */
  template <typename T1, std::size_t k, typename T2, typename T3>
  inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, k>
  gather(const std::array<T1, k> &U, const T2 i, const T3 l)
  {
    dealii::Tensor<1, k> result;
    for (unsigned int j = 0; j < k; ++j)
      result[j] = U[j](i, l);
    return result;
  }


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
    TimerOutput::Scope t(computing_timer_,
                         "time_step - setup scratch space");

    const auto &locally_owned = offline_data_->locally_owned();
    const auto &sparsity_pattern = offline_data_->sparsity_pattern();

    f_i_.reinit(locally_owned.n_elements());
    dij_matrix_.reinit(sparsity_pattern);
  }


  template <int dim>
  std::tuple<typename TimeStep<dim>::vector_type, double>
  TimeStep<dim>::compute_step(const vector_type &U_old, const double t_old)
  {
    deallog << "TimeStep<dim>::compute_step()" << std::endl;

    /*
     * Step 1: Compute d_ij and f_i
     */

    {
      deallog << "        compute d_ij and f_i" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "time_step - compute d_ij and f_i");

      const auto &locally_owned = offline_data_->locally_owned();
      const auto &sparsity_pattern = offline_data_->sparsity_pattern();
      const auto &norm_matrix_ = offline_data_->norm_matrix();
      const auto &nij_matrix_ = offline_data_->nij_matrix();

      /*
       * FIXME: Workaround: IndexSet does not have an iterator with
       * complete operater arithmetic (that is required for tbb). We
       * iterate over the local vector f_i_ instead and do a bit of index
       * juggling...
       */

      const auto on_subranges = [&](const auto it1, const auto it2) {
        /* [it1, it2) is an iterator range over f_i_ */
        for (auto it = it1; it != it2; ++it) {
          /* Determine absolute position of it in vector: */
          const unsigned int pos = std::distance(f_i_.begin(), it);
          const auto i = locally_owned.nth_index_in_set(pos);

          const auto U_i = gather(U_old, i);

          /* Populate f_i */
          *it = problem_description_->f(U_i);

          for (auto jt = sparsity_pattern.begin(i);
               jt != sparsity_pattern.end(i);
               ++jt) {
            const auto j = jt->column();

            const auto U_j = gather(U_old, j);
            const auto n_ij = gather(nij_matrix_, i, j);

            // benchmark

            const auto [lambda_max, n_iterations] =
                riemann_solver_->lambda_max(U_i, U_j, n_ij);
          }
        }
      };

      parallel::apply_to_subranges(
          f_i_.begin(), f_i_.end(), on_subranges, 4096);
    }
  }

} /* namespace grendel */

#endif /* TIME_STEP_TEMPLATE_H */
