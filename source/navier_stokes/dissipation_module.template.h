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
      const HyperbolicSystem &hyperbolic_system,
      const OfflineData<dim, Number> &offline_data,
      const InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "DissipationModule"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , hyperbolic_system_(&problem_description)
      , offline_data_(&offline_data)
      , initial_values_(&initial_values)
      , n_warnings_(0)
      , n_iterations_velocity_(0.)
      , n_iterations_internal_energy_(0.)
  {
    use_gmg_velocity_ = false;
    add_parameter("multigrid velocity",
                  use_gmg_velocity_,
                  "Use geometric multigrid for velocity component");

    gmg_max_iter_vel_ = 12;
    add_parameter("multigrid velocity - max iter",
                  gmg_max_iter_vel_,
                  "Maximal number of CG iterations with GMG smoother");

    gmg_smoother_range_vel_ = 8.;
    add_parameter("multigrid velocity - chebyshev range",
                  gmg_smoother_range_vel_,
                  "Chebyshev smoother: eigenvalue range parameter");

    gmg_smoother_max_eig_vel_ = 2.0;
    add_parameter("multigrid velocity - chebyshev max eig",
                  gmg_smoother_max_eig_vel_,
                  "Chebyshev smoother: maximal eigenvalue");

    use_gmg_internal_energy_ = false;
    add_parameter("multigrid energy",
                  use_gmg_internal_energy_,
                  "Use geometric multigrid for internal energy component");

    gmg_max_iter_en_ = 15;
    add_parameter("multigrid energy - max iter",
                  gmg_max_iter_en_,
                  "Maximal number of CG iterations with GMG smoother");

    gmg_smoother_range_en_ = 15.;
    add_parameter("multigrid energy - chebyshev range",
                  gmg_smoother_range_en_,
                  "Chebyshev smoother: eigenvalue range parameter");

    gmg_smoother_max_eig_en_ = 2.0;
    add_parameter("multigrid energy - chebyshev max eig",
                  gmg_smoother_max_eig_en_,
                  "Chebyshev smoother: maximal eigenvalue");

    gmg_smoother_degree_ = 3;
    add_parameter("multigrid - chebyshev degree",
                  gmg_smoother_degree_,
                  "Chebyshev smoother: degree");

    gmg_smoother_n_cg_iter_ = 10;
    add_parameter("multigrid - chebyshev cg iter",
                  gmg_smoother_n_cg_iter_,
                  "Chebyshev smoother: number of CG iterations to approximate "
                  "eigenvalue");

    gmg_min_level_ = 0;
    add_parameter("multigrid - min level",
                  gmg_min_level_,
                  "Minimal mesh level to be visited in the geometric multigrid "
                  "cycle where the coarse grid solver (Chebyshev) is called");

    tolerance_ = Number(1.0e-12);
    add_parameter("tolerance", tolerance_, "Tolerance for linear solvers");

    tolerance_linfty_norm_ = false;
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

    /* Initialize multigrid: */

    if (!use_gmg_velocity_ && !use_gmg_internal_energy_)
      return;

    const unsigned int n_levels =
        offline_data_->dof_handler().get_triangulation().n_global_levels();
    const unsigned int min_level = std::min(gmg_min_level_, n_levels - 1);
    MGLevelObject<IndexSet> relevant_sets(0, n_levels - 1);
    for (unsigned int level = 0; level < n_levels; ++level)
      dealii::DoFTools::extract_locally_relevant_level_dofs(
          offline_data_->dof_handler(), level, relevant_sets[level]);
#if DEAL_II_VERSION_GTE(9, 3, 0)
    mg_constrained_dofs_.initialize(offline_data_->dof_handler(),
                                    relevant_sets);
#else
    mg_constrained_dofs_.initialize(offline_data_->dof_handler());
#endif
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(Boundary::dirichlet);
    boundary_ids.insert(Boundary::no_slip);
    mg_constrained_dofs_.make_zero_boundary_constraints(
        offline_data_->dof_handler(), boundary_ids);

    typename MatrixFree<dim, float>::AdditionalData additional_data_level;
    additional_data_level.tasks_parallel_scheme =
        MatrixFree<dim, float>::AdditionalData::none;

    level_matrix_free_.resize(min_level, n_levels - 1);
    level_density_.resize(min_level, n_levels - 1);
    for (unsigned int level = min_level; level < n_levels; ++level) {
      additional_data_level.mg_level = level;
      AffineConstraints<double> constraints(relevant_sets[level]);
      // constraints.add_lines(mg_constrained_dofs_.get_boundary_indices(level));
      // constraints.merge(mg_constrained_dofs_.get_level_constraints(level));
      constraints.close();
      level_matrix_free_[level].reinit(
          offline_data_->discretization().mapping(),
          offline_data_->dof_handler(),
          constraints,
          offline_data_->discretization().quadrature_1d(),
          additional_data_level);
      level_matrix_free_[level].initialize_dof_vector(level_density_[level]);
    }

    mg_transfer_velocity_.build(
        offline_data_->dof_handler(), mg_constrained_dofs_, level_matrix_free_);
    mg_transfer_energy_.build(offline_data_->dof_handler(), level_matrix_free_);
  }


  template <int dim, typename Number>
  Number DissipationModule<dim, Number>::step(vector_type &U,
                                              Number t,
                                              Number tau,
                                              unsigned int cycle) const
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
     * This is a bit ugly: tau_ is internally used in velocity_vmult() and
     * internal_energy_vmult().
     */

    tau_ = tau;
    theta_ = Number(0.5) + shift_ * tau; // FIXME
#ifdef DEBUG_OUTPUT
    std::cout << "        perform time-step with tau = " << tau << std::endl;
    std::cout << "        (shift: " << shift_ << ")" << std::endl;
#endif

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
      LIKWID_MARKER_START("time_step_n_0");

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto U_i = U.template get_tensor<VA>(i);
        const auto rho_i = hyperbolic_system_->density(U_i);
        const auto M_i = hyperbolic_system_->momentum(U_i);
        const auto rho_e_i = hyperbolic_system_->internal_energy(U_i);
        const auto m_i = load_value<VA>(lumped_mass_matrix, i);

        store_value<VA>(density_, rho_i, i);
        /* (5.4a) */
        for (unsigned int d = 0; d < dim; ++d) {
          store_value<VA>(velocity_.block(d), M_i[d] / rho_i, i);
          store_value<VA>(velocity_rhs_.block(d), m_i * (M_i[d]), i);
        }
        store_value<VA>(internal_energy_, rho_e_i / rho_i, i);
      }

      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        const auto U_i = U.get_tensor(i);
        const auto rho_i = hyperbolic_system_->density(U_i);
        const auto M_i = hyperbolic_system_->momentum(U_i);
        const auto rho_e_i = hyperbolic_system_->internal_energy(U_i);
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

        const auto normal = std::get<0>(entry.second);
        const auto id = std::get<3>(entry.second);
        const auto position = std::get<4>(entry.second);

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
          const auto rho_i = hyperbolic_system_->density(U_i);
          const auto V_i = hyperbolic_system_->momentum(U_i) / rho_i;
          const auto e_i = hyperbolic_system_->internal_energy(U_i) / rho_i;

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

      /*
       * Update MG matrices all 4 time steps; this is a balance because more
       * refreshes will render the approximation better, at some additional
       * cost.
       */
      if (use_gmg_velocity_ && (cycle % 4 == 1)) {
        MGLevelObject<typename PreconditionChebyshev<
            VelocityMatrix<dim, float, Number>,
            LinearAlgebra::distributed::BlockVector<float>,
            DiagonalMatrix<dim, float>>::AdditionalData>
            smoother_data(level_matrix_free_.min_level(),
                          level_matrix_free_.max_level());

        level_velocity_matrices_.resize(level_matrix_free_.min_level(),
                                        level_matrix_free_.max_level());
        mg_transfer_velocity_.interpolate_to_mg(
            offline_data_->dof_handler(), level_density_, density_);

        for (unsigned int level = level_matrix_free_.min_level();
             level <= level_matrix_free_.max_level();
             ++level) {
          level_velocity_matrices_[level].initialize(*hyperbolic_system_,
                                                     *offline_data_,
                                                     level_matrix_free_[level],
                                                     level_density_[level],
                                                     theta_ * tau_,
                                                     level);
          level_velocity_matrices_[level].compute_diagonal(
              smoother_data[level].preconditioner);
          if (level == level_matrix_free_.min_level()) {
            smoother_data[level].degree = numbers::invalid_unsigned_int;
            smoother_data[level].eig_cg_n_iterations = 500;
            smoother_data[level].smoothing_range = 1e-3;
          } else {
            smoother_data[level].degree = gmg_smoother_degree_;
            smoother_data[level].eig_cg_n_iterations = gmg_smoother_n_cg_iter_;
            smoother_data[level].smoothing_range = gmg_smoother_range_vel_;
            if (gmg_smoother_n_cg_iter_ == 0)
              smoother_data[level].max_eigenvalue = gmg_smoother_max_eig_vel_;
          }
        }
        mg_smoother_velocity_.initialize(level_velocity_matrices_,
                                         smoother_data);
      }

      LIKWID_MARKER_STOP("time_step_n_0");
    }

    /* Compute the global minimum of the internal energy: */

    // .begin() and .end() denote the locally owned index range:
    auto e_min_old =
        *std::min_element(internal_energy_.begin(), internal_energy_.end());
    e_min_old = Utilities::MPI::min(e_min_old, mpi_communicator_);

    /*
     * Step 1: Solve velocity update:
     */
    {
      Scope scope(computing_timer_, "time step [N] 1 - update velocities");

      LIKWID_MARKER_START("time_step_n_1");

      VelocityMatrix<dim, Number, Number> velocity_operator;
      velocity_operator.initialize(*hyperbolic_system_,
                                   *offline_data_,
                                   matrix_free_,
                                   density_,
                                   theta_ * tau_);

      const auto tolerance_velocity =
          (tolerance_linfty_norm_ ? velocity_rhs_.linfty_norm()
                                  : velocity_rhs_.l2_norm()) *
          tolerance_;

      /*
       * Multigrid might lack robustness for some cases, so in case it takes
       * too many iterations we better switch to the more robust plain
       * conjugate gradient method.
       */
      try {
        if (!use_gmg_velocity_)
          throw SolverControl::NoConvergence(0, 0.);

        using bvt_float = LinearAlgebra::distributed::BlockVector<float>;

        MGCoarseGridApplySmoother<bvt_float> mg_coarse;
        mg_coarse.initialize(mg_smoother_velocity_);

        mg::Matrix<bvt_float> mg_matrix(level_velocity_matrices_);

        Multigrid<bvt_float> mg(mg_matrix,
                                mg_coarse,
                                mg_transfer_velocity_,
                                mg_smoother_velocity_,
                                mg_smoother_velocity_,
                                level_velocity_matrices_.min_level(),
                                level_velocity_matrices_.max_level());

        const auto &dof_handler = offline_data_->dof_handler();
        PreconditionMG<dim, bvt_float, MGTransferVelocity<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer_velocity_);

        SolverControl solver_control(gmg_max_iter_vel_, tolerance_velocity);
        SolverCG<block_vector_type> solver(solver_control);
        solver.solve(
            velocity_operator, velocity_, velocity_rhs_, preconditioner);

        /* update exponential moving average */
        n_iterations_velocity_ =
            0.9 * n_iterations_velocity_ + 0.1 * solver_control.last_step();

      } catch (SolverControl::NoConvergence &) {

        SolverControl solver_control(1000, tolerance_velocity);
        SolverCG<block_vector_type> solver(solver_control);
        solver.solve(
            velocity_operator, velocity_, velocity_rhs_, diagonal_matrix);

        /* update exponential moving average, counting also GMG iterations */
        n_iterations_velocity_ *= 0.9;
        n_iterations_velocity_ +=
            0.1 * (use_gmg_velocity_ ? gmg_max_iter_vel_ : 0) +
            0.1 * solver_control.last_step();
      }

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

            const auto mu = hyperbolic_system_->mu();
            const auto lambda = hyperbolic_system_->lambda();

            for (unsigned int cell = cell_range.first; cell < cell_range.second;
                 ++cell) {
              velocity.reinit(cell);
              energy.reinit(cell);
#if DEAL_II_VERSION_GTE(9, 3, 0)
              velocity.gather_evaluate(src, EvaluationFlags::gradients);
#else
              velocity.read_dof_values(src);
              velocity.evaluate(false, true);
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
        const auto rhs_i = load_value<VA>(internal_energy_rhs_, i);
        const auto m_i = load_value<VA>(lumped_mass_matrix, i);
        const auto rho_i = load_value<VA>(density_, i);
        const auto e_i = load_value<VA>(internal_energy_, i);
        /* rhs_i already contains m_i K_i^{n+1/2} */
        store_value<VA>(
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

        const auto id = std::get<3>(entry.second);
        const auto position = std::get<4>(entry.second);

        if (id == Boundary::dirichlet) {
          /* Prescribe internal energy: */
          const auto U_i =
              initial_values_->initial_state(position, t + theta_ * tau_);
          const auto rho_i = hyperbolic_system_->density(U_i);
          const auto e_i = hyperbolic_system_->internal_energy(U_i) / rho_i;
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

      /*
       * Update MG matrices all 4 time steps; this is a balance because more
       * refreshes will render the approximation better, at some additional
       * cost.
       */
      if (use_gmg_internal_energy_ && (cycle % 4 == 1)) {
        MGLevelObject<typename PreconditionChebyshev<
            EnergyMatrix<dim, float, Number>,
            LinearAlgebra::distributed::Vector<float>>::AdditionalData>
            smoother_data(level_matrix_free_.min_level(),
                          level_matrix_free_.max_level());

        level_energy_matrices_.resize(level_matrix_free_.min_level(),
                                      level_matrix_free_.max_level());

        for (unsigned int level = level_matrix_free_.min_level();
             level <= level_matrix_free_.max_level();
             ++level) {
          level_energy_matrices_[level].initialize(
              *offline_data_,
              level_matrix_free_[level],
              level_density_[level],
              theta_ * tau_ * hyperbolic_system_->cv_inverse_kappa(),
              level);
          level_energy_matrices_[level].compute_diagonal(
              smoother_data[level].preconditioner);
          if (level == level_matrix_free_.min_level()) {
            smoother_data[level].degree = numbers::invalid_unsigned_int;
            smoother_data[level].eig_cg_n_iterations = 500;
            smoother_data[level].smoothing_range = 1e-3;
          } else {
            smoother_data[level].degree = gmg_smoother_degree_;
            smoother_data[level].eig_cg_n_iterations = gmg_smoother_n_cg_iter_;
            smoother_data[level].smoothing_range = gmg_smoother_range_en_;
            if (gmg_smoother_n_cg_iter_ == 0)
              smoother_data[level].max_eigenvalue = gmg_smoother_max_eig_en_;
          }
        }
        mg_smoother_energy_.initialize(level_energy_matrices_, smoother_data);
      }

      LIKWID_MARKER_STOP("time_step_n_2");
    }

    /*
     * Step 3: Solve internal energy update:
     */
    {
      Scope scope(computing_timer_, "time step [N] 3 - update internal energy");

      LIKWID_MARKER_START("time_step_n_3");

      EnergyMatrix<dim, Number, Number> energy_operator;
      energy_operator.initialize(*offline_data_,
                                 matrix_free_,
                                 density_,
                                 theta_ * tau_ *
                                     hyperbolic_system_->cv_inverse_kappa());

      const auto tolerance_internal_energy =
          (tolerance_linfty_norm_ ? internal_energy_rhs_.linfty_norm()
                                  : internal_energy_rhs_.l2_norm()) *
          tolerance_;

      try {
        if (!use_gmg_internal_energy_)
          throw SolverControl::NoConvergence(0, 0.);

        using vt_float = LinearAlgebra::distributed::Vector<float>;
        MGCoarseGridApplySmoother<vt_float> mg_coarse;
        mg_coarse.initialize(mg_smoother_energy_);
        mg::Matrix<vt_float> mg_matrix(level_energy_matrices_);

        Multigrid<vt_float> mg(mg_matrix,
                               mg_coarse,
                               mg_transfer_energy_,
                               mg_smoother_energy_,
                               mg_smoother_energy_,
                               level_energy_matrices_.min_level(),
                               level_energy_matrices_.max_level());

        const auto &dof_handler = offline_data_->dof_handler();
        PreconditionMG<dim, vt_float, MGTransferEnergy<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer_energy_);

        SolverControl solver_control(gmg_max_iter_en_,
                                     tolerance_internal_energy);
        SolverCG<scalar_type> solver(solver_control);
        solver.solve(energy_operator,
                     internal_energy_,
                     internal_energy_rhs_,
                     preconditioner);

        /* update exponential moving average */
        n_iterations_internal_energy_ = 0.9 * n_iterations_internal_energy_ +
                                        0.1 * solver_control.last_step();

      } catch (SolverControl::NoConvergence &) {

        SolverControl solver_control(1000, tolerance_internal_energy);
        SolverCG<scalar_type> solver(solver_control);
        solver.solve(energy_operator,
                     internal_energy_,
                     internal_energy_rhs_,
                     diagonal_matrix);

        /* update exponential moving average, counting also GMG iterations */
        n_iterations_internal_energy_ *= 0.9;
        n_iterations_internal_energy_ +=
            0.1 * (use_gmg_internal_energy_ ? gmg_max_iter_en_ : 0) +
            0.1 * solver_control.last_step();
      }

      /*
       * Check for local minimum principle on internal energy:
       */
      {
        // .begin() and .end() denote the locally owned index range:
        auto e_min_new =
            *std::min_element(internal_energy_.begin(), internal_energy_.end());
        e_min_new = Utilities::MPI::min(e_min_new, mpi_communicator_);

        constexpr Number eps = std::numeric_limits<Number>::epsilon();
        if (e_min_new < e_min_old * (1. - 1000. * eps)) {
          n_warnings_++;
          if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0)
            std::cout << "[INFO] Dissipation module: Insufficient CFL: "
                         "Invariant domain violation detected"
                      << std::endl;
        }
      }

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
      LIKWID_MARKER_START("time_step_n_4");

      const unsigned int size_regular = n_owned / simd_length * simd_length;

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        auto U_i = U.template get_tensor<VA>(i);
        const auto rho_i = hyperbolic_system_->density(U_i);

        /* (5.4b) */
        auto m_i_new = (Number(1.) - alpha) * hyperbolic_system_->momentum(U_i);
        for (unsigned int d = 0; d < dim; ++d) {
          m_i_new[d] += alpha * rho_i * load_value<VA>(velocity_.block(d), i);
        }

        /* (5.12)f */
        auto rho_e_i_new =
            (Number(1.0) - alpha) * hyperbolic_system_->internal_energy(U_i);
        rho_e_i_new += alpha * rho_i * load_value<VA>(internal_energy_, i);

        /* (5.18) */
        const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

        for (unsigned int d = 0; d < dim; ++d)
          U_i[1 + d] = m_i_new[d];
        U_i[1 + dim] = E_i_new;

        U.template write_tensor<VA>(U_i, i);
      }

      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        auto U_i = U.get_tensor(i);
        const auto rho_i = hyperbolic_system_->density(U_i);

        /* (5.4b) */
        auto m_i_new = (Number(1.) - alpha) * hyperbolic_system_->momentum(U_i);
        for (unsigned int d = 0; d < dim; ++d) {
          m_i_new[d] += alpha * rho_i * velocity_.block(d).local_element(i);
        }

        /* (5.12)f */
        auto rho_e_i_new =
            (Number(1.) - alpha) * hyperbolic_system_->internal_energy(U_i);
        rho_e_i_new += alpha * rho_i * internal_energy_.local_element(i);

        /* (5.18) */
        const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

        for (unsigned int d = 0; d < dim; ++d)
          U_i[1 + d] = m_i_new[d];
        U_i[1 + dim] = E_i_new;

        U.write_tensor(U_i, i);
      }

      U.update_ghost_values();

      LIKWID_MARKER_STOP("time_step_n_4");
    }

    CALLGRIND_STOP_INSTRUMENTATION

    return tau;
  }


} // namespace ryujin
