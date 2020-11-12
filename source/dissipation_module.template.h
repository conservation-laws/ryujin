//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef DISSIPATION_MODULE_TEMPLATE_H
#define DISSIPATION_MODULE_TEMPLATE_H

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
      const ryujin::ProblemDescription &problem_description,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const ryujin::InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "DissipationModule"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , problem_description_(&problem_description)
      , offline_data_(&offline_data)
      , initial_values_(&initial_values)
      , n_iterations_velocity_(0.)
      , n_iterations_internal_energy_(0.)
  {
    tolerance_ = Number(1.0e-12);
    add_parameter("tolerance", tolerance_, "Tolerance for linear solvers");

    tolerance_linfty_norm_ = true;
    add_parameter("tolerance linfty norm",
                  tolerance_linfty_norm_,
                  "Use the l_infty norm instead of the l_2 norm for the "
                  "stopping criterion");

    shift_ = Number(0.0);
    add_parameter(
        "shift", shift_, "Implicit shift applied to the Crank Nicolson scheme");
  }


  template <int dim, typename Number>
  void DissipationModule<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "DissipationModule<dim, Number>::prepare()" << std::endl;
#endif

    /* Initialize vectors: */

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

    velocity_.reinit(dim);
    velocity_rhs_.reinit(dim);
    for (unsigned int i = 0; i < dim; ++i) {
      velocity_.block(i).reinit(scalar_partitioner);
      velocity_rhs_.block(i).reinit(scalar_partitioner);
    }

    internal_energy_.reinit(scalar_partitioner);
    internal_energy_rhs_.reinit(scalar_partitioner);

    density_.reinit(scalar_partitioner);

    const unsigned int n_levels =
        offline_data_->dof_handler().get_triangulation().n_global_levels();
    MGLevelObject<IndexSet> relevant_sets(0, n_levels - 1);
    for (unsigned int level = 0; level < n_levels; ++level)
      DoFTools::extract_locally_relevant_level_dofs(
          offline_data_->dof_handler(), level, relevant_sets[level]);
    mg_constrained_dofs_.initialize(offline_data_->dof_handler(),
                                    relevant_sets);
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(Boundary::dirichlet);
    boundary_ids.insert(Boundary::no_slip);
    mg_constrained_dofs_.make_zero_boundary_constraints(
        offline_data_->dof_handler(), boundary_ids);

    level_matrix_free_.resize(0, n_levels - 1);
    level_density_.resize(0, n_levels - 1);
    for (unsigned int level = 0; level < n_levels; ++level) {
      additional_data.mg_level = level;
      AffineConstraints<double> constraints(relevant_sets[level]);
      constraints.add_lines(mg_constrained_dofs_.get_boundary_indices(level));
      constraints.merge(mg_constrained_dofs_.get_level_constraints(level));
      constraints.close();
      level_matrix_free_[level].reinit(
          offline_data_->discretization().mapping(),
          offline_data_->dof_handler(),
          constraints,
          offline_data_->discretization().quadrature_1d(),
          additional_data);
      level_matrix_free_[level].initialize_dof_vector(level_density_[level]);
    }

    mg_transfer_.build(offline_data_->dof_handler(),
                       mg_constrained_dofs_,
                       level_matrix_free_);
  }


  template <int dim, typename Number>
  template <typename VectorType>
  void DissipationModule<dim, Number>::internal_energy_vmult(
      VectorType &dst, const VectorType &src) const
  {
    /* Apply action of m_i rho_i e_i: */

    using VA = VectorizedArray<Number>;
    constexpr auto simd_length = VA::size();

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const unsigned int n_owned = offline_data_->n_locally_owned();
    const unsigned int size_regular = n_owned / simd_length * simd_length;

    RYUJIN_PARALLEL_REGION_BEGIN

    RYUJIN_OMP_FOR
    for (unsigned int i = 0; i < size_regular; i += simd_length) {
      const auto m_i = simd_load(lumped_mass_matrix, i);
      const auto rho_i = simd_load(density_, i);
      const auto e_i = simd_load(src, i);
      simd_store(dst, m_i * rho_i * e_i, i);
    }

    RYUJIN_PARALLEL_REGION_END

    for (unsigned int i = size_regular; i < n_owned; ++i) {
      const auto m_i = lumped_mass_matrix.local_element(i);
      const auto rho_i = density_.local_element(i);
      const auto e_i = src.local_element(i);
      dst.local_element(i) = m_i * rho_i * e_i;
    }

    /* Apply action of diffusion operator \sum_j beta_ij e_j: */

    const auto integrator =
        [this](const auto &data, auto &dst, const auto &src, const auto range) {
          constexpr auto order_fe = Discretization<dim>::order_finite_element;
          constexpr auto order_quad = Discretization<dim>::order_quadrature;
          FEEvaluation<dim, order_fe, order_quad, 1, Number> energy(data);
          const auto factor =
              theta_ * tau_ * problem_description_->cv_inverse_kappa();

          for (unsigned int cell = range.first; cell < range.second; ++cell) {
            energy.reinit(cell);
#if DEAL_II_VERSION_GTE(9, 3, 0)
            energy.gather_evaluate(src, EvaluationFlags::gradients);
#else
            energy.gather_evaluate(src, false, true);
#endif
            for (unsigned int q = 0; q < energy.n_q_points; ++q) {
              energy.submit_gradient(factor * energy.get_gradient(q), q);
            }
#if DEAL_II_VERSION_GTE(9, 3, 0)
            energy.integrate_scatter(EvaluationFlags::gradients, dst);
#else
            energy.integrate_scatter(false, true, dst);
#endif
          }
        };

    matrix_free_.template cell_loop<scalar_type, scalar_type>(
        integrator, dst, src, /* zero destination */ false);

    /* Fix up constrained degrees of freedom: */

    const auto &boundary_map = offline_data_->boundary_map();

    for (auto entry : boundary_map) {
      const auto i = entry.first;
      if (i >= n_owned)
        continue;

      const auto &[normal, id, position] = entry.second;
      if (id == Boundary::dirichlet) {
        dst.local_element(i) = src.local_element(i);
      }
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

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &affine_constraints = offline_data_->affine_constraints();

    /* Index ranges for the iteration over the sparsity pattern : */

    constexpr auto simd_length = VA::size();
    const unsigned int n_owned = offline_data_->n_locally_owned();
    const unsigned int size_regular = n_owned / simd_length * simd_length;

    DiagonalMatrix<dim, Number> diagonal_matrix;

    /*
     * Set time step size and record the time t_{n+1/2} for the computed
     * velocity.
     *
     * This is a but ugly: tau_ is internally used in velocity_vmult() and
     * internal_energy_vmult().
     */

    tau_ = tau;
    theta_ = Number(0.5) + shift_ * tau; // FIXME
    t_interp_ = t + theta_ * tau;

    /*
     * Step 0:
     *
     * Build right hand side for the velocity update.
     * Also initialize solution vectors for internal energy and velocity
     * update.
     */
    {
      Scope scope(computing_timer_, "time step [N] 0 - build velocities rhs");

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_0");

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto U_i = U.get_vectorized_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto M_i = problem_description_->momentum(U_i);
        const auto rho_e_i = problem_description_->internal_energy(U_i);
        const auto m_i = simd_load(lumped_mass_matrix, i);

        simd_store(density_, rho_i, i);
        /* (5.4a) */
        for (unsigned int d = 0; d < dim; ++d) {
          simd_store(velocity_.block(d), M_i[d] / rho_i, i);
          simd_store(velocity_rhs_.block(d), m_i * (M_i[d]), i);
        }
        simd_store(internal_energy_, rho_e_i / rho_i, i);
      }

      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        const auto U_i = U.get_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto M_i = problem_description_->momentum(U_i);
        const auto rho_e_i = problem_description_->internal_energy(U_i);
        const auto m_i = lumped_mass_matrix.local_element(i);

        density_.local_element(i) = rho_i;
        /* (5.4a) */
        for (unsigned int d = 0; d < dim; ++d) {
          velocity_.block(d).local_element(i) = M_i[d] / rho_i;
          velocity_rhs_.block(d).local_element(i) = m_i * M_i[d];
        }
        internal_energy_.local_element(i) = rho_e_i / rho_i;
      }

      /*
       * Set up "strongly enforced" boundary conditions that are not stored
       * in the AffineConstraints map. In this case we enforce boundary
       * values by imposing them strongly in the iteration by setting the
       * initial vector and the right hand side to the right value:
       */

      const auto &boundary_map = offline_data_->boundary_map();

      for (auto entry : boundary_map) {
        const auto i = entry.first;
        if (i >= n_owned)
          continue;

        const auto &[normal, id, position] = entry.second;

        if (id == Boundary::slip) {
          /* Remove normal component of velocity: */
          Tensor<1, dim, Number> V_i;
          Tensor<1, dim, Number> RHS_i;
          for (unsigned int d = 0; d < dim; ++d) {
            V_i[d] = velocity_.block(d).local_element(i);
            RHS_i[d] = velocity_rhs_.block(d).local_element(i);
          }
          V_i -= 1. * (V_i * normal) * normal;
          RHS_i -= 1. * (RHS_i * normal) * normal;
          for (unsigned int d = 0; d < dim; ++d) {
            velocity_.block(d).local_element(i) = V_i[d];
            velocity_rhs_.block(d).local_element(i) = RHS_i[d];
          }

        } else if (id == Boundary::no_slip) {

          /* Set velocity to zero: */
          for (unsigned int d = 0; d < dim; ++d) {
            velocity_.block(d).local_element(i) = Number(0.);
            velocity_rhs_.block(d).local_element(i) = Number(0.);
          }

        } else if (id == Boundary::dirichlet) {

          /* Prescribe velocity: */
          const auto U_i =
              initial_values_->initial_state(position, t + theta_ * tau_);
          const auto rho_i = problem_description_->density(U_i);
          const auto V_i = problem_description_->momentum(U_i) / rho_i;
          const auto e_i = problem_description_->internal_energy(U_i) / rho_i;

          for (unsigned int d = 0; d < dim; ++d) {
            velocity_.block(d).local_element(i) = V_i[d];
            velocity_rhs_.block(d).local_element(i) = V_i[d];
          }

          internal_energy_.local_element(i) = e_i;
        }
      }

      /*
       * Zero out constrained degrees of freedom due to periodic boundary
       * conditions. These boundary conditions are enforced by modifying
       * the stencil - consequently we have to remove constrained dofs from
       * the linear system.
       */

      affine_constraints.set_zero(density_);
      affine_constraints.set_zero(internal_energy_);
      for (unsigned int d = 0; d < dim; ++d) {
        affine_constraints.set_zero(velocity_.block(d));
        affine_constraints.set_zero(velocity_rhs_.block(d));
      }

      /* Prepare preconditioner: */

      diagonal_matrix.reinit(lumped_mass_matrix, density_, affine_constraints);

        MGLevelObject<typename PreconditionChebyshev<
            VelocityMatrix<dim, Number>,
            LinearAlgebra::distributed::BlockVector<Number>,
            DiagonalMatrix<dim, Number>>::AdditionalData>
            smoother_data(0, level_matrix_free_.max_level());

        level_velocity_matrices_.resize(0, level_matrix_free_.max_level());
        mg_transfer_.interpolate_to_mg(
            offline_data_->dof_handler(), level_density_, density_);

        for (unsigned int level = level_matrix_free_.min_level();
             level <= level_matrix_free_.max_level();
             ++level) {
          level_velocity_matrices_[level].initialize(*problem_description_,
                                                     *offline_data_,
                                                     level_matrix_free_[level],
                                                     level_density_[level],
                                                     theta_ * tau_,
                                                     level);
          level_velocity_matrices_[level].compute_diagonal(
              smoother_data[level].preconditioner);
          smoother_data[level].smoothing_range = 15.;
          smoother_data[level].degree = 3;
          smoother_data[level].eig_cg_n_iterations = 15;
        }
        mg_smoother_.initialize(level_velocity_matrices_, smoother_data);

      LIKWID_MARKER_STOP("time_step_0");
    }

    /*
     * Step 1: Solve velocity update:
     */
    {
      Scope scope(computing_timer_, "time step [N] 1 - update velocities");

      LIKWID_MARKER_START("time_step_n_1");

      VelocityMatrix<dim, Number> velocity_operator;
      velocity_operator.initialize(*problem_description_,
                                   *offline_data_,
                                   matrix_free_,
                                   density_,
                                   theta_ * tau_);

      MGCoarseGridApplySmoother<LinearAlgebra::distributed::BlockVector<Number>>
          mg_coarse;
      mg_coarse.initialize(mg_smoother_);
      mg::Matrix<LinearAlgebra::distributed::BlockVector<Number>> mg_matrix(
          level_velocity_matrices_);

      Multigrid<LinearAlgebra::distributed::BlockVector<Number>> mg(
          mg_matrix, mg_coarse, mg_transfer_, mg_smoother_, mg_smoother_);
      PreconditionMG<dim,
                     LinearAlgebra::distributed::BlockVector<Number>,
                     MGTransferVelocity<dim, Number>>
          preconditioner(offline_data_->dof_handler(), mg, mg_transfer_);

      SolverControl solver_control(1000,
                                   (tolerance_linfty_norm_
                                        ? velocity_rhs_.linfty_norm()
                                        : velocity_rhs_.l2_norm()) *
                                       tolerance_);
      SolverCG<block_vector_type> solver(solver_control);
      solver.solve(velocity_operator, velocity_, velocity_rhs_, preconditioner);

      /* update exponential moving average */
      n_iterations_velocity_ =
          0.9 * n_iterations_velocity_ + 0.1 * solver_control.last_step();

      LIKWID_MARKER_STOP("time_step_n_1");
    }

    /*
     * Step 2: Build internal energy right hand side:
     */
    {
      Scope scope(computing_timer_,
                  "time step [N] 2 - build internal energy rhs");

      LIKWID_MARKER_START("time_step_n_2");

      /* Compute m_i K_i^{n+1/2}:  (5.5) */
      matrix_free_.template cell_loop<scalar_type, block_vector_type>(
          [this](const auto &data,
                 auto &dst,
                 const auto &src,
                 const auto cell_range) {
            constexpr auto order_fe = Discretization<dim>::order_finite_element;
            constexpr auto order_quad = Discretization<dim>::order_quadrature;
            FEEvaluation<dim, order_fe, order_quad, dim, Number> velocity(data);
            FEEvaluation<dim, order_fe, order_quad, 1, Number> energy(data);

            const auto mu = problem_description_->mu();
            const auto lambda = problem_description_->lambda();

            for (unsigned int cell = cell_range.first; cell < cell_range.second;
                 ++cell) {
              velocity.reinit(cell);
              energy.reinit(cell);
#if DEAL_II_VERSION_GTE(9, 3, 0)
              velocity.gather_evaluate(src, EvaluationFlags::gradients);
#else
              velocity.gather_evaluate(src, false, true);
#endif

              for (unsigned int q = 0; q < velocity.n_q_points; ++q) {

                const auto symmetric_gradient =
                    velocity.get_symmetric_gradient(q);
                const auto divergence = trace(symmetric_gradient);

                auto S = 2. * mu * symmetric_gradient;
                for (unsigned int d = 0; d < dim; ++d)
                  S[d][d] += (lambda - 2. / 3. * mu) * divergence;

                energy.submit_value(symmetric_gradient * S, q);
              }
#if DEAL_II_VERSION_GTE(9, 3, 0)
              energy.integrate_scatter(EvaluationFlags::values, dst);
#else
              energy.integrate_scatter(true, false, dst);
#endif
            }
          },
          internal_energy_rhs_,
          velocity_,
          /* zero destination */ true);

      using VA = VectorizedArray<Number>;
      constexpr auto simd_length = VA::size();

      const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const unsigned int size_regular = n_owned / simd_length * simd_length;

      RYUJIN_PARALLEL_REGION_BEGIN

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto rhs_i = simd_load(internal_energy_rhs_, i);
        const auto m_i = simd_load(lumped_mass_matrix, i);
        const auto rho_i = simd_load(density_, i);
        const auto e_i = simd_load(internal_energy_, i);
        /* rhs_i already contains m_i K_i^{n+1/2} */
        simd_store(
            internal_energy_rhs_, m_i * rho_i * e_i + theta_ * tau_ * rhs_i, i);
      }

      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        const auto rhs_i = internal_energy_rhs_.local_element(i);
        const auto m_i = lumped_mass_matrix.local_element(i);
        const auto rho_i = density_.local_element(i);
        const auto e_i = internal_energy_.local_element(i);
        /* rhs_i already contains m_i K_i^{n+1/2} */
        internal_energy_rhs_.local_element(i) =
            m_i * rho_i * e_i + theta_ * tau_ * rhs_i;
      }

      /*
       * Set up "strongly enforced" boundary conditions that are not stored
       * in the AffineConstraints map: We enforce Neumann conditions (i.e.,
       * insulating boundary conditions) everywhere except for Dirichlet
       * boundaries where we have to enforce prescribed conditions:
       */

      const auto &boundary_map = offline_data_->boundary_map();

      for (auto entry : boundary_map) {
        const auto i = entry.first;
        if (i >= n_owned)
          continue;

        const auto &[normal, id, position] = entry.second;

        if (id == Boundary::dirichlet) {
          /* Prescribe internal energy: */
          const auto U_i =
              initial_values_->initial_state(position, t + theta_ * tau_);
          const auto rho_i = problem_description_->density(U_i);
          const auto e_i = problem_description_->internal_energy(U_i) / rho_i;
          internal_energy_rhs_.local_element(i) = e_i;
        }
      }

      /*
       * Zero out constrained degrees of freedom due to periodic boundary
       * conditions. These boundary conditions are enforced by modifying
       * the stencil - consequently we have to remove constrained dofs from
       * the linear system.
       */
      affine_constraints.set_zero(internal_energy_rhs_);

      LIKWID_MARKER_STOP("time_step_n_2");
    }

    /*
     * Step 3: Solve internal energy update:
     */
    {
      Scope scope(computing_timer_, "time step [N] 3 - update internal energy");

      LIKWID_MARKER_START("time_step_n_3");

      LinearOperator<scalar_type, scalar_type> internal_energy_operator;
      internal_energy_operator.vmult = [this](scalar_type &dst,
                                              const scalar_type &src) {
        internal_energy_vmult<scalar_type>(dst, src);
      };

      SolverControl solver_control(1000,
                                   (tolerance_linfty_norm_
                                        ? internal_energy_rhs_.linfty_norm()
                                        : internal_energy_rhs_.l2_norm()) *
                                       tolerance_);

      SolverCG<scalar_type> solver(solver_control);
      solver.solve(internal_energy_operator,
                   internal_energy_,
                   internal_energy_rhs_,
                   diagonal_matrix);

      /* update exponential moving average */
      n_iterations_internal_energy_ = 0.9 * n_iterations_internal_energy_ +
                                      0.1 * solver_control.last_step();

      LIKWID_MARKER_STOP("time_step_n_3");
    }

    /*
     * Step 4: Copy vectors
     *
     * FIXME: Memory access is suboptimal...
     */
    {
      const auto alpha = Number(1.) / theta_;

      Scope scope(computing_timer_, "time step [N] 4 - write back vectors");

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_4");

      const unsigned int size_regular = n_owned / simd_length * simd_length;

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        auto U_i = U.get_vectorized_tensor(i);
        const auto rho_i = problem_description_->density(U_i);

        /* (5.4b) */
        auto m_i_new =
            (Number(1.) - alpha) * problem_description_->momentum(U_i);
        for (unsigned int d = 0; d < dim; ++d) {
          m_i_new[d] += alpha * rho_i * simd_load(velocity_.block(d), i);
        }

        /* (5.12)f */
        auto rho_e_i_new =
            (Number(1.0) - alpha) * problem_description_->internal_energy(U_i);
        rho_e_i_new += alpha * rho_i * simd_load(internal_energy_, i);

        /* (5.18) */
        const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

        for (unsigned int d = 0; d < dim; ++d)
          U_i[1 + d] = m_i_new[d];
        U_i[1 + dim] = E_i_new;

        U.write_vectorized_tensor(U_i, i);
      }

      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        auto U_i = U.get_tensor(i);
        const auto rho_i = problem_description_->density(U_i);

        /* (5.4b) */
        auto m_i_new =
            (Number(1.) - alpha) * problem_description_->momentum(U_i);
        for (unsigned int d = 0; d < dim; ++d) {
          m_i_new[d] += alpha * rho_i * velocity_.block(d).local_element(i);
        }

        /* (5.12)f */
        auto rho_e_i_new =
            (Number(1.) - alpha) * problem_description_->internal_energy(U_i);
        rho_e_i_new += alpha * rho_i * internal_energy_.local_element(i);

        /* (5.18) */
        const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

        for (unsigned int d = 0; d < dim; ++d)
          U_i[1 + d] = m_i_new[d];
        U_i[1 + dim] = E_i_new;

        U.write_tensor(U_i, i);
      }

      U.update_ghost_values();

      LIKWID_MARKER_STOP("time_step_4");
    }

    CALLGRIND_STOP_INSTRUMENTATION

    return tau;
  }


} /* namespace ryujin */

#endif /* DISSIPATION_MODULE_TEMPLATE_H */
