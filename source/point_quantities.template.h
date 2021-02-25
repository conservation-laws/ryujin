//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#pragma once

#include "openmp.h"
#include "point_quantities.h"
#include "scope.h"
#include "scratch_data.h"
#include "simd.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>

DEAL_II_NAMESPACE_OPEN
template <int rank, int dim, typename Number>
bool operator<(const Tensor<rank, dim, Number> &left,
               const Tensor<rank, dim, Number> &right)
{
  return std::lexicographical_compare(
      left.begin_raw(), left.end_raw(), right.begin_raw(), right.end_raw());
}
DEAL_II_NAMESPACE_CLOSE

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  PointQuantities<dim, Number>::PointQuantities(
      const MPI_Comm &mpi_communicator,
      const ryujin::ProblemDescription &problem_description,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "PointQuantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , problem_description_(&problem_description)
      , offline_data_(&offline_data)
  {
    add_parameter("interior manifolds",
                  interior_manifolds_,
                  "List of level set functions describing interior manifolds. "
                  "The description is used to only output point values for "
                  "vertices belonging to a certain level set.");

    boundary_manifolds_.push_back({"upper_boundary", "y - 1.0"});
    boundary_manifolds_.push_back({"lower_boundary", "y"});
    add_parameter("boundary manifolds",
                  boundary_manifolds_,
                  "List of level set functions describing boundary. The "
                  "description is used to only output point values for "
                  "boundary vertices belonging to a certain level set.");
  }


  template <int dim, typename Number>
  void PointQuantities<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "PointQuantities<dim, Number>::prepare()" << std::endl;
#endif

    /* Initialize matrix free context: */

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags_boundary_faces =
        (update_values | update_gradients | update_JxW_values |
         update_normal_vectors);
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::none;

    matrix_free_.reinit(offline_data_->discretization().mapping(),
                        offline_data_->dof_handler(),
                        offline_data_->affine_constraints(),
                        offline_data_->discretization().quadrature_1d(),
                        additional_data);

    /* Initialize vectors: */

    const auto &scalar_partitioner =
        matrix_free_.get_dof_info(0).vector_partitioner;

    velocity_.reinit(dim);
    vorticity_.reinit(dim == 2 ? 1 : dim);
    boundary_stress_.reinit(dim);
    for (unsigned int i = 0; i < dim; ++i) {
      velocity_.block(i).reinit(scalar_partitioner);
      if constexpr (dim == 3)
        vorticity_.block(i).reinit(scalar_partitioner);
      boundary_stress_.block(i).reinit(scalar_partitioner);
    }
    if constexpr (dim == 2)
      vorticity_.block(0).reinit(scalar_partitioner);

    lumped_boundary_mass_.reinit(scalar_partitioner);

    /* Compute lumped boundary mass matrix: */

    {
      constexpr auto order_fe = Discretization<dim>::order_finite_element;
      constexpr auto order_quad = Discretization<dim>::order_quadrature;

      FEFaceEvaluation<dim, order_fe, order_quad, 1, Number> phi(matrix_free_);

      lumped_boundary_mass_ = 0.;

      const auto begin = matrix_free_.n_inner_face_batches();
      const auto size = matrix_free_.n_boundary_face_batches();
      for (unsigned int face = begin; face < begin + size; ++face) {
        const auto id = matrix_free_.get_boundary_id(face);

        /* only compute on slip and no_slip boundaries */
        if (id != Boundary::slip && id != Boundary::no_slip)
          continue;
        phi.reinit(face);
        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(1., q);
        }

#if DEAL_II_VERSION_GTE(9, 3, 0)
        phi.integrate_scatter(EvaluationFlags::values, lumped_boundary_mass_);
#else
        phi.integrate_scatter(true, false, lumped_boundary_mass_);
#endif
      }

      lumped_boundary_mass_.compress(VectorOperation::add);
    }

    /* Create interior maps: */

    const unsigned int n_owned = offline_data_->n_locally_owned();

    interior_maps_.clear();
    std::transform(
        interior_manifolds_.begin(),
        interior_manifolds_.end(),
        std::back_inserter(interior_maps_),
        [this, n_owned](auto it) {
          const auto &[name, expression] = it;
          FunctionParser<dim> level_set_function(expression);

          std::map<types::global_dof_index, Point<dim>> map;

          const auto &dof_handler = offline_data_->dof_handler();
          const auto &scalar_partitioner = offline_data_->scalar_partitioner();

          for (auto &cell : dof_handler.active_cell_iterators()) {

            /* skip non-local cells */
            if (!cell->is_locally_owned())
              continue;

            for (auto v : GeometryInfo<dim>::vertex_indices()) {

              const auto position = cell->vertex(v);

              /* only record points sufficiently close to the level set */
              if (std::abs(level_set_function.value(position)) > 1.e-12)
                continue;

              const auto global_index = cell->vertex_dof_index(v, 0);
              const auto index =
                  scalar_partitioner->global_to_local(global_index);

              /* skip nonlocal */
              if (index >= n_owned)
                continue;

              /* skip constrained */
              if (offline_data_->affine_constraints().is_constrained(
                      offline_data_->scalar_partitioner()->local_to_global(
                          index)))
                continue;

              map.insert({index, position});
            }
          }
          return std::make_tuple(name, map);
        });

    /* Create boundary maps: */

    boundary_maps_.clear();
    std::transform(
        boundary_manifolds_.begin(),
        boundary_manifolds_.end(),
        std::back_inserter(boundary_maps_),
        [this, n_owned](auto it) {
          const auto &[name, expression] = it;
          FunctionParser<dim> level_set_function(expression);

          std::multimap<types::global_dof_index, boundary_description> map;

          for (const auto &entry : offline_data_->boundary_map()) {
            /* skip nonlocal */
            if (entry.first >= n_owned)
              continue;

            /* skip constrained */
            if (offline_data_->affine_constraints().is_constrained(
                    offline_data_->scalar_partitioner()->local_to_global(
                        entry.first)))
              continue;

            const auto position = std::get<2>(entry.second);
            if (std::abs(level_set_function.value(position)) < 1.e-12)
              map.insert(entry);
          }
          return std::make_tuple(name, map);
        });
  }


  template <int dim, typename Number>
  void PointQuantities<dim, Number>::compute(const vector_type &U,
                                             const Number t,
                                             std::string name,
                                             unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "PointQuantities<dim, Number>::compute()" << std::endl;
#endif

    using VA = VectorizedArray<Number>;
    constexpr auto simd_length = VA::size();
    const unsigned int n_owned = offline_data_->n_locally_owned();
    const unsigned int size_regular = n_owned / simd_length * simd_length;

    /*
     * Step 0: Copy velocity:
     */

    {
      RYUJIN_PARALLEL_REGION_BEGIN
      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto U_i = U.get_vectorized_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto M_i = problem_description_->momentum(U_i);

        for (unsigned int d = 0; d < dim; ++d) {
          simd_store(velocity_.block(d), M_i[d] / rho_i, i);
        }
      }
      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        const auto U_i = U.get_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto M_i = problem_description_->momentum(U_i);

        for (unsigned int d = 0; d < dim; ++d) {
          velocity_.block(d).local_element(i) = M_i[d] / rho_i;
        }
      }

      velocity_.update_ghost_values();
    }

    /*
     * Step 1: Compute vorticity:
     */

    if (interior_maps_.size() != 0) {
      /*
       * Nota bene: This computes "m_i V_i", i.e., the result has to be
       * divided by the lumped mass matrix (or multiplied with the inverse
       * of the full mass matrix).
       */

      matrix_free_.template cell_loop<block_vector_type, block_vector_type>(
          [](const auto &data,
             auto &dst,
             const auto &src,
             const auto cell_range) {
            constexpr auto order_fe = Discretization<dim>::order_finite_element;
            constexpr auto order_quad = Discretization<dim>::order_quadrature;
            FEEvaluation<dim, order_fe, order_quad, dim, Number> velocity(data);
            FEEvaluation<dim, order_fe, order_quad, dim == 2 ? 1 : dim, Number>
                vorticity(data);

            for (unsigned int cell = cell_range.first; cell < cell_range.second;
                 ++cell) {
              velocity.reinit(cell);
              vorticity.reinit(cell);
#if DEAL_II_VERSION_GTE(9, 3, 0)
              velocity.gather_evaluate(src, EvaluationFlags::gradients);
#else
              velocity.read_dof_values(src);
              velocity.evaluate(false, true);
#endif
              for (unsigned int q = 0; q < velocity.n_q_points; ++q) {
                const auto curl = velocity.get_curl(q);
                vorticity.submit_value(curl, q);
              }
#if DEAL_II_VERSION_GTE(9, 3, 0)
              vorticity.integrate_scatter(EvaluationFlags::values, dst);
#else
              vorticity.integrate(true, false);
              vorticity.distribute_local_to_global(dst);
#endif
            }
          },
          vorticity_,
          velocity_,
          /* zero destination */ true);

      vorticity_.compress(VectorOperation::add);

      /* Fix up boundary: */

      for (auto it : offline_data_->boundary_map()) {
        const auto i = it.first;
        if (i >= n_owned)
          continue;

        const auto [normal, id, _] = it.second;

        /* Only retain the normal component of the curl on the boundary: */

        if (id == Boundary::slip || id == Boundary::no_slip) {
          if constexpr (dim == 2) {
            vorticity_.block(0).local_element(i) = 0.;
          } else if constexpr (dim == 3) {
            Tensor<1, dim, Number> curl_v_i;
            for (unsigned int d = 0; d < dim; ++d)
              curl_v_i[d] = vorticity_.block(d).local_element(i);
            curl_v_i = (curl_v_i * normal) * normal;
            for (unsigned int d = 0; d < dim; ++d)
              vorticity_.block(d).local_element(i) = curl_v_i[d];
          }
        }
      }
    }

    /*
     * Step 2: Boundary stress:
     */

    if (boundary_maps_.size() != 0) {
      /*
       * Nota bene: This computes "m_i Sn_i", i.e., the result has to be
       * divided by the lumped mass matrix (or multiplied with the inverse
       * of the full mass matrix).
       */

      constexpr auto order_fe = Discretization<dim>::order_finite_element;
      constexpr auto order_quad = Discretization<dim>::order_quadrature;

      FEFaceEvaluation<dim, order_fe, order_quad, dim, Number> velocity(
          matrix_free_);

      boundary_stress_ = 0.;

      const auto mu = problem_description_->mu();
      const auto lambda = problem_description_->lambda();

      const auto begin = matrix_free_.n_inner_face_batches();
      const auto size = matrix_free_.n_boundary_face_batches();
      for (unsigned int face = begin; face < begin + size; ++face) {
        const auto id = matrix_free_.get_boundary_id(face);

        /* only compute on slip and no_slip boundaries */
        if (id != Boundary::slip && id != Boundary::no_slip)
          continue;

        velocity.reinit(face);
#if DEAL_II_VERSION_GTE(9, 3, 0)
        velocity.gather_evaluate(velocity_, EvaluationFlags::gradients);
#else
        velocity.read_dof_values(velocity_);
        velocity.evaluate(false, true);
#endif
        for (unsigned int q = 0; q < velocity.n_q_points; ++q) {
          const auto normal = velocity.get_normal_vector(q);

          const auto symmetric_gradient = velocity.get_symmetric_gradient(q);
          const auto divergence = trace(symmetric_gradient);
          auto S = 2. * mu * symmetric_gradient;
          for (unsigned int d = 0; d < dim; ++d)
            S[d][d] += (lambda - 2. / 3. * mu) * divergence;
          velocity.submit_value(S * (-normal), q);
        }

#if DEAL_II_VERSION_GTE(9, 3, 0)
        velocity.integrate_scatter(EvaluationFlags::values, boundary_stress_);
#else
        velocity.integrate(true, false);
        velocity.distribute_local_to_global(boundary_stress_);
#endif
      }

      boundary_stress_.compress(VectorOperation::add);
    }

    /*
     * Step 3: Collect all interior points of interest and output to log file:
     */

    for (const auto &it : interior_maps_) {
      const auto &[description, interior_map] = it;

      using entry = std::tuple<Point<dim>, // position
                               Number,     // lumped mass
                               rank1_type, // state
                               Number,     // pressure
                               curl_type>; // vorticity

      std::vector<entry> entries;
      for (const auto &it : interior_map) {

        /* Only record locally owned degrees of freedom */
        const auto i = it.first;
        if (i >= n_owned)
          continue;

        const auto &position = it.second;

        const auto U_i = U.get_tensor(i);
        const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
        const auto m_i = lumped_mass_matrix.local_element(i);
        const auto P_i = problem_description_->pressure(U_i);

        curl_type V_i;
        for (unsigned int d = 0; d < (dim == 2 ? 1 : dim); ++d) {
          V_i[d] = vorticity_.block(d).local_element(i);
        }

        entries.push_back({position, m_i, U_i, P_i, V_i / m_i});
      }

      std::vector<entry> all_entries;
      {
        const auto received =
            Utilities::MPI::gather(mpi_communicator_, entries);

        for (auto &&it : received)
          std::move(
              std::begin(it), std::end(it), std::back_inserter(all_entries));

        std::sort(all_entries.begin(), all_entries.end());
      }

      if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
        std::ofstream output(name + "-" + description + "-" +
                             Utilities::to_string(cycle, 6) + ".log");

        output << std::scientific << std::setprecision(14);
        output << "# state and pressure at time t = " << t << std::endl;
        output << "# position\tlumped mass\tstate (rho,M,E)"
               << "\tpressure\tvorticity" << std::endl;

        for (const auto &entry : all_entries) {
          const auto &[position, m_i, U_i, P_i, V_i] = entry;
          output << position << "\t" << m_i << "\t"   //
                 << U_i << "\t" << P_i << "\t" << V_i //
                 << std::endl;
        }
      }
    } /* interior_maps_ */

    /*
     * Step 4: Collect all boundary points of interest and output to log file:
     */

    for (const auto &it : boundary_maps_) {
      const auto &[description, boundary_map] = it;

      using entry = std::tuple<Point<dim>,              // position
                               Number,                  // lumped boundary mass
                               Tensor<1, dim, Number>,  // normal
                               rank1_type,              // state
                               Number,                  // pressure
                               Tensor<1, dim, Number>>; // stress

      std::vector<entry> entries;
      for (const auto &it : boundary_map) {

        /* Only record locally owned degrees of freedom */
        const auto i = it.first;
        if (i >= n_owned)
          continue;

        const auto &[normal, id, position] = it.second;

        const auto U_i = U.get_tensor(i);
        const auto m_i = lumped_boundary_mass_.local_element(i);
        const auto P_i = problem_description_->pressure(U_i);

        Tensor<1, dim, Number> Sn_i;
        for (unsigned int d = 0; d < dim; ++d) {
          Sn_i[d] = boundary_stress_.block(d).local_element(i);
        }

        entries.push_back({position, m_i, normal, U_i, P_i, Sn_i / m_i});
      }

      std::vector<entry> all_entries;
      {
        const auto received =
            Utilities::MPI::gather(mpi_communicator_, entries);

        for (auto &&it : received)
          std::move(
              std::begin(it), std::end(it), std::back_inserter(all_entries));

        std::sort(all_entries.begin(), all_entries.end());
      }

      if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
        std::ofstream output(name + "-" + description + "-" +
                             Utilities::to_string(cycle, 6) + ".log");

        output << std::scientific << std::setprecision(14);
        output << "# state and pressure at time t = " << t << std::endl;
        output << "# position\tlumped boundary mass\tnormal\t"
               << "state (rho,M,E)\tpressure\tstress" << std::endl;

        for (const auto &entry : all_entries) {
          const auto &[position, m_i, normal, U_i, P_i, Sn_i] = entry;
          output << position << "\t" << m_i << "\t" << normal << "\t" //
                 << U_i << "\t" << P_i << "\t" << Sn_i                //
                 << std::endl;
        }
      }
    } /* boundary_maps_ */
  }

} /* namespace ryujin */
