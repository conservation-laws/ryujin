//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "openmp.h"
#include "point_quantities.h"
#include "scope.h"
#include "scratch_data.h"
#include "simd.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>

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

    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(offline_data_->dof_handler(),
                                            relevant_dofs);
    const auto scalar_partitioner =
        std::make_shared<Utilities::MPI::Partitioner>(
            offline_data_->dof_handler().locally_owned_dofs(),
            relevant_dofs,
            offline_data_->dof_handler().get_communicator());

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
      const QGauss<dim - 1> quad(Discretization<dim>::order_quadrature);
      FEFaceValues<dim> fe_face_values(
          offline_data_->discretization().mapping(),
          offline_data_->dof_handler().get_fe(),
          quad,
          update_values | update_JxW_values);
      Vector<Number> cell_rhs(
          offline_data_->dof_handler().get_fe().dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(cell_rhs.size());

      for (const auto &cell :
           offline_data_->dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face : GeometryInfo<dim>::face_indices())
            if (cell->at_boundary(face) &&
                (cell->face(face)->boundary_id() == Boundary::slip ||
                 cell->face(face)->boundary_id() == Boundary::no_slip)) {
              fe_face_values.reinit(cell, face);
              for (unsigned int i = 0; i < cell_rhs.size(); ++i) {
                double sum = 0;
                for (unsigned int q = 0; q < quad.size(); ++q)
                  sum +=
                      fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);
                cell_rhs(i) = sum;
              }
              cell->get_dof_indices(dof_indices);
              offline_data_->affine_constraints().distribute_local_to_global(
                  cell_rhs, dof_indices, lumped_boundary_mass_);
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
          simd_store(velocity_.block(d), M_i[d] * (Number(1.) / rho_i), i);
        }
      }
      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        const auto U_i = U.get_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto M_i = problem_description_->momentum(U_i);

        for (unsigned int d = 0; d < dim; ++d) {
          velocity_.block(d).local_element(i) = M_i[d] * (Number(1.) / rho_i);
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
      vorticity_ = 0;
      const QGauss<dim> quad(Discretization<dim>::order_quadrature);
      constexpr unsigned int n_q_points =
          Utilities::pow(Discretization<dim>::order_quadrature, dim);
      FEValues<dim> fe_values(offline_data_->discretization().mapping(),
                              offline_data_->dof_handler().get_fe(),
                              quad,
                              update_values | update_gradients |
                                  update_JxW_values);
      std::array<Vector<Number>, dim> cell_rhs;
      for (auto c : cell_rhs)
        c.reinit(offline_data_->dof_handler().get_fe().dofs_per_cell);
      constexpr int curl_dim = (dim > 2 ? dim : 1);
      std::vector<Tensor<1, curl_dim>> curls(n_q_points);
      std::vector<types::global_dof_index> dof_indices(cell_rhs[0].size());

      for (const auto &cell :
           offline_data_->dof_handler().active_cell_iterators())
        if (cell->is_locally_owned()) {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);

          /* Interpolate gradient to q-points including constraints */
          for (unsigned int d = 0; d < dim; ++d)
            offline_data_->affine_constraints().get_dof_values(
                velocity_.block(d),
                dof_indices.data(),
                cell_rhs[d].begin(),
                cell_rhs[d].end());
          for (unsigned int q = 0; q < n_q_points; ++q) {
            Tensor<2, dim> u_grad;
            for (unsigned int i = 0; i < cell_rhs[0].size(); ++i) {
              const auto &shape_grad = fe_values.shape_grad(i, q);
              for (unsigned int d = 0; d < dim; ++d)
                u_grad[d] += shape_grad * cell_rhs[d](i);
            }
            if (dim == 2)
              curls[q][0] = u_grad[1][0] - u_grad[0][1];
            else if (dim == 3) {
              curls[q][0] = u_grad[2][1] - u_grad[1][2];
              curls[q][1] = u_grad[0][2] - u_grad[2][0];
              curls[q][2] = u_grad[1][0] - u_grad[0][1];
            }
            curls[q] *= fe_values.JxW(q);
          }

          /* Multiply by test function and sum over quadrature points */
          for (unsigned int i = 0; i < cell_rhs[0].size(); ++i) {
            Tensor<1, curl_dim> sum;
            const double *shape_i = &fe_values.shape_value(i, 0);
            for (unsigned int q = 0; q < n_q_points; ++q)
              sum += shape_i[q] * curls[q];
            for (unsigned int d = 0; d < curl_dim; ++d)
              cell_rhs[d][i] = sum[d];
          }
          for (unsigned int d = 0; d < curl_dim; ++d)
            offline_data_->affine_constraints().distribute_local_to_global(
                cell_rhs[d], dof_indices, vorticity_.block(d));
        }

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
       * Nota bene: This computes "m^\partial_i Sn_i", i.e., the result has
       * to be divided by the lumped boundary mass matrix (or multiplied
       * with the inverse of the full boundary mass matrix).
       */

      boundary_stress_ = 0.;

      const auto mu = problem_description_->mu();
      const auto lambda = problem_description_->lambda();

      const QGauss<dim - 1> quad(Discretization<dim>::order_quadrature);
      constexpr unsigned int n_q_points =
          Utilities::pow(Discretization<dim>::order_quadrature, dim - 1);
      FEFaceValues<dim> fe_face_values(
          offline_data_->discretization().mapping(),
          offline_data_->dof_handler().get_fe(),
          quad,
          update_values | update_JxW_values | update_normal_vectors);
      std::array<Vector<Number>, dim> cell_rhs;
      for (auto c : cell_rhs)
        c.reinit(offline_data_->dof_handler().get_fe().dofs_per_cell);
      std::vector<Tensor<1, dim>> stresses(n_q_points);
      std::vector<types::global_dof_index> dof_indices(cell_rhs.size());

      for (const auto &cell :
           offline_data_->dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face : GeometryInfo<dim>::face_indices())
            if (cell->at_boundary(face) &&
                (cell->face(face)->boundary_id() == Boundary::slip ||
                 cell->face(face)->boundary_id() == Boundary::no_slip)) {
              fe_face_values.reinit(cell, face);

              cell->get_dof_indices(dof_indices);

              /* Interpolate gradient to q-points including constraints */
              for (unsigned int d = 0; d < dim; ++d)
                offline_data_->affine_constraints().get_dof_values(
                    velocity_.block(d),
                    dof_indices.data(),
                    cell_rhs[d].begin(),
                    cell_rhs[d].end());
              for (unsigned int q = 0; q < n_q_points; ++q) {
                Tensor<2, dim> u_grad;
                for (unsigned int i = 0; i < cell_rhs[0].size(); ++i) {
                  const auto &shape_grad = fe_face_values.shape_grad(i, q);
                  for (unsigned int d = 0; d < dim; ++d)
                    u_grad[d] += shape_grad * cell_rhs[d](i);
                }

                const auto symmetric_gradient = symmetrize(u_grad);
                const auto divergence = trace(symmetric_gradient);
                auto S = 2. * mu * symmetric_gradient;
                for (unsigned int d = 0; d < dim; ++d)
                  S[d][d] += (lambda - 2. / 3. * mu) * divergence;
                const auto result = S * (-fe_face_values.normal_vector(q) *
                                         fe_face_values.JxW(q));
                stresses[q] = result;
              }

              /* Multiply by test function and sum over quadrature points */
              for (unsigned int i = 0; i < cell_rhs[0].size(); ++i) {
                Tensor<1, dim> sum;
                const double *shape_i = &fe_face_values.shape_value(i, 0);
                for (unsigned int q = 0; q < n_q_points; ++q)
                  sum += shape_i[q] * stresses[q];
                for (unsigned int d = 0; d < dim; ++d)
                  cell_rhs[d][i] = sum[d];
              }
              for (unsigned int d = 0; d < dim; ++d)
                offline_data_->affine_constraints().distribute_local_to_global(
                    cell_rhs[d], dof_indices, boundary_stress_.block(d));
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
                               state_type, // state
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
                               state_type,              // state
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
