//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "dissipation_module.h"
#include "introspection.h"
#include "openmp.h"
#include "scope.h"
#include "simd.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <atomic>

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  DissipationModule<dim, Number>::DissipationModule(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const ParabolicSystem &parabolic_system,
      const OfflineData<dim, Number> &offline_data,
      const InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "DissipationModule"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , parabolic_system_(&parabolic_system)
      , offline_data_(&offline_data)
      , initial_values_(&initial_values)
      , n_warnings_(0)
  {
    tolerance_ = Number(1.0e-12);
    add_parameter("tolerance", tolerance_, "Tolerance for linear solvers");

    tolerance_linfty_norm_ = false;
    add_parameter("tolerance linfty norm",
                  tolerance_linfty_norm_,
                  "Use the l_infty norm instead of the l_2 norm for the "
                  "stopping criterion");

    std::fill_n(use_gmg_.begin(), n_implicit_systems, false);
    add_parameter("multigrid",
                  use_gmg_,
                  "Use a geometric multigrid for the specified linear systems");

    enter_subsection("Multigrid");

    std::fill_n(gmg_max_iter_.begin(), n_implicit_systems, 12); // 8
    add_parameter("max iter",
                  gmg_max_iter_,
                  "Maximal number of CG iterations with GMG smoother before "
                  "falling back to diagonal preconditiong");

    std::fill_n(gmg_smoother_range_.begin(), n_implicit_systems, 8.); // 15.
    add_parameter("chebyshev range",
                  gmg_smoother_range_,
                  "Chebyshev smoother: eigenvalue range parameter");

    std::fill_n(gmg_smoother_max_eig_.begin(), n_implicit_systems, 2.0);
    add_parameter("chebyshev max eig",
                  gmg_smoother_max_eig_,
                  "Chebyshev smoother: maximal eigenvalue");

    std::fill_n(gmg_smoother_degree_.begin(), n_implicit_systems, 3);
    add_parameter("chebyshev degree",
                  gmg_smoother_degree_,
                  "Chebyshev smoother: degree");

    std::fill_n(gmg_smoother_n_cg_iter_.begin(), n_implicit_systems, 10);
    add_parameter("chebyshev cg iter",
                  gmg_smoother_n_cg_iter_,
                  "Chebyshev smoother: number of CG iterations to approximate "
                  "eigenvalue");

    std::fill_n(gmg_min_level_.begin(), n_implicit_systems, 0);
    add_parameter("min level",
                  gmg_min_level_,
                  "Minimal mesh level to be visited in the geometric multigrid "
                  "cycle where the coarse grid solver (Chebyshev) is called");

    leave_subsection();
  }


  template <int dim, typename Number>
  void DissipationModule<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "DissipationModule<dim, Number>::prepare()" << std::endl;
#endif

    /* Nothing to do: */
    if constexpr (n_implicit_systems == 0)
      return;

    /* Initialize matrix free context and vectors: */

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::none;

    matrix_free_.reinit(offline_data_->discretization().mapping(),
                        offline_data_->dof_handler(),
                        offline_data_->affine_constraints(),
                        offline_data_->discretization().quadrature_1d(),
                        additional_data);

    const auto &scalar_partitioner =
        matrix_free_.get_dof_info(0).vector_partitioner;

    solution_.reinit(parabolic_problem_dimension);
    right_hand_side_.reinit(parabolic_problem_dimension);
    for (unsigned int i = 0; i < parabolic_problem_dimension; ++i) {
      solution_.block(i).reinit(scalar_partitioner);
      right_hand_side_.block(i).reinit(scalar_partitioner);
    }

    /* Initialize multigrid: */

    if (!std::any_of(
            use_gmg_.begin(), use_gmg_.end(), [](auto value) { return value; }))
      return;

    __builtin_trap(); // FIXME
  }


  template <int dim, typename Number>
  Number DissipationModule<dim, Number>::step(vector_type &U,
                                              Number /*t*/,
                                              Number tau,
                                              unsigned int /*cycle*/) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "DissipationModule<dim, Number>::step()" << std::endl;
#endif

    /* Nothing to do: */
    if constexpr (n_implicit_systems == 0)
      return tau;

#ifdef DEBUG_OUTPUT
    std::cout << "        perform time-step with tau = " << tau << std::endl;
    std::cout << "        (theta: " << theta_ << ")" << std::endl;
#endif

    CALLGRIND_START_INSTRUMENTATION

    using VA = VectorizedArray<Number>;

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &affine_constraints = offline_data_->affine_constraints();

    /* Index ranges for the iteration over the sparsity pattern : */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    constexpr auto simd_length = VA::size();
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    /*
     * Step 0:
     *
     * Build right hand sides and initialize solution vectors.
     */
    {
      Scope scope(computing_timer_,
                  "time step [N] 0 - initialize vectors and build rhs");

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_n_0");

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          const auto m_i = load_value<T>(lumped_mass_matrix, i);
          const auto U_i = U.template get_tensor<T>(i);

          const auto &[V_i, rhs_i] =
              parabolic_system_->compute_parabolic_state_and_rhs(U_i, m_i);

          for (unsigned int d = 0; d < parabolic_problem_dimension; ++d) {
            store_value<T>(solution_.block(d), V_i[d], i);
            store_value<T>(right_hand_side_.block(d), rhs_i[d], i);
          }
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      LIKWID_MARKER_STOP("time_step_n_0");
      RYUJIN_PARALLEL_REGION_END
    }


    return tau;
  }

} // namespace ryujin
