//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef POINT_QUANTITIES_TEMPLATE_H
#define POINT_QUANTITIES_TEMPLATE_H

#include "openmp.h"
#include "point_quantities.h"
#include "scope.h"
#include "scratch_data.h"
#include "simd.h"

#include <deal.II/base/work_stream.h>
#include <deal.II/matrix_free/fe_evaluation.h>

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
    add_parameter(
        "output planes",
        output_planes_,
        "A vector of hyperplanes described by an origin, normal vector and a "
        "tolerance. The description is used to only output point values for "
        "vertices belonging to a cell cut by the cutplane. Example declaration "
        "of two hyper planes in 3D, one normal to the x-axis and one normal to "
        "the y-axis: \"0,0,0 : 1,0,0 : 0.01 ; 0,0,0 : 0,1,0 : 0,01\"");
  }


  template <int dim, typename Number>
  void PointQuantities<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "PointQuantities<dim, Number>::prepare()" << std::endl;
#endif

    /* Initialize matrix free context: */

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags_boundary_faces =
        (update_values | update_gradients | update_JxW_values |
         update_normal_vectors);

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
    pressure_.reinit(scalar_partitioner);

    /*
     * Collect local dof indices and associated point locations for the
     * prescribed cut planes:
     */
    {
      cutplane_map_.clear();

      const auto &partitioner = offline_data_->scalar_partitioner();
      const auto &discretization = offline_data_->discretization();
      const unsigned int dofs_per_cell =
          discretization.finite_element().dofs_per_cell;

      const auto local_assemble_system =
          [&](const auto &cell, auto &scratch, auto &copy) {
            /* iterate over locally owned cells and the ghost layer */

            auto &is_artificial = copy.is_artificial_;
            auto &local_dof_indices = copy.local_dof_indices_;
            auto &local_boundary_map = copy.local_boundary_map_;
            auto &fe_values = scratch.fe_values_;

            is_artificial = cell->is_artificial();
            if (is_artificial)
              return;

            fe_values.reinit(cell);

            local_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);

            /* clear out copy data: */
            local_boundary_map.clear();

            unsigned int id = 0;
            /* Record every matching cutplane: */
            for (const auto &plane : output_planes_) {
              const auto &[origin, normal, tolerance] = plane;

              unsigned int above = 0;
              unsigned int below = 0;
              bool cut = false;

              for (auto v : GeometryInfo<dim>::vertex_indices()) {
                const auto vertex = cell->vertex(v);
                const auto distance = (vertex - Point<dim>(origin)) * normal;
                if (distance > -tolerance)
                  above++;
                if (distance < tolerance)
                  below++;
                if (above > 0 && below > 0) {
                  cut = true;
                  break;
                }
              }

              if (cut) {
                /* Record all vertex indices: */

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                  /*
                   * This is a bloody hack: Use "vertex_dof_index" to retrieve
                   * the vertex associated to the current degree of freedom.
                   */
                  Point<dim> position;
                  const auto global_index = local_dof_indices[j];
                  for (auto v : GeometryInfo<dim>::vertex_indices())
                    if (cell->vertex_dof_index(v, 0) == global_index) {
                      position = cell->vertex(v);
                      break;
                    }

                  const auto index = partitioner->global_to_local(global_index);

                  /* Insert a dummy value for boundary normal */
                  local_boundary_map.insert(
                      {index, {dealii::Tensor<1, dim>(), id, position}});
                } /* for j */
              }
              ++id;
            } /* plane */
          };

      const auto copy_local_to_global = [&](const auto &copy) {
        const auto &is_artificial = copy.is_artificial_;
        const auto &local_boundary_map = copy.local_boundary_map_;

        if (is_artificial)
          return;

        for (const auto entry : local_boundary_map) {
          const auto &index = entry.first;
          const auto &[normal, id , position] = entry.second;
          cutplane_map_[id][index] = position;
        }
      };

      cutplane_map_.resize(output_planes_.size());

      const auto &dof_handler = offline_data_->dof_handler();
      WorkStream::run(dof_handler.begin_active(),
                      dof_handler.end(),
                      local_assemble_system,
                      copy_local_to_global,
                      AssemblyScratchData<dim>(discretization),
                      AssemblyCopyData<dim, Number>());
    }
  }


  template <int dim, typename Number>
  void PointQuantities<dim, Number>::compute(const vector_type &U,
                                             std::string name,
                                             Number t,
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
     * Step 0: Copy velocity and compute pressure:
     */

    {
      RYUJIN_PARALLEL_REGION_BEGIN
      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto U_i = U.get_vectorized_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto M_i = problem_description_->momentum(U_i);
        const auto P_i = problem_description_->pressure(U_i);

        for (unsigned int d = 0; d < dim; ++d) {
          simd_store(velocity_.block(d), M_i[d] / rho_i, i);
        }
        simd_store(pressure_, P_i, i);
      }
      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        const auto U_i = U.get_tensor(i);
        const auto rho_i = problem_description_->density(U_i);
        const auto M_i = problem_description_->momentum(U_i);
        const auto P_i = problem_description_->pressure(U_i);

        for (unsigned int d = 0; d < dim; ++d) {
          velocity_.block(d).local_element(i) = M_i[d] / rho_i;
        }
        pressure_.local_element(i) = P_i;
      }
    }

    /*
     * Step 1: Compute vorticity:
     */

    {
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
              velocity.gather_evaluate(src, false, true);
#endif
              for (unsigned int q = 0; q < velocity.n_q_points; ++q) {
                const auto curl = velocity.get_curl(q);
                vorticity.submit_value(curl, q);
              }
#if DEAL_II_VERSION_GTE(9, 3, 0)
              vorticity.integrate_scatter(EvaluationFlags::values, dst);
#else
              vorticity.integrate_scatter(true, false, dst);
#endif
            }
          },
          vorticity_,
          velocity_,
          /* zero destination */ true);

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

    {
      /* We simply integrate over all boundary faces by hand: */

      constexpr auto order_fe = Discretization<dim>::order_finite_element;
      constexpr auto order_quad = Discretization<dim>::order_quadrature;

      FEFaceEvaluation<dim, order_fe, order_quad, dim, Number> velocity(
          matrix_free_);
      FEFaceEvaluation<dim, order_fe, order_quad, 1, Number> pressure(
          matrix_free_);

      boundary_stress_ = 0.;

      const auto mu = problem_description_->mu();
      const auto lambda = problem_description_->lambda();

      const auto begin = matrix_free_.n_inner_face_batches();
      const auto size = matrix_free_.n_boundary_face_batches();
      for (unsigned int face = begin; face < begin + size; ++face) {
        const auto id = matrix_free_.get_boundary_id(face);

        /* only compute on slip and no_slip boundary conditions */
        if (id != Boundary::slip && id != Boundary::no_slip)
          continue;

        velocity.reinit(face);
        pressure.reinit(face);
#if DEAL_II_VERSION_GTE(9, 3, 0)
        velocity.gather_evaluate(velocity_, EvaluationFlags::gradients);
        pressure.gather_evaluate(pressure_, EvaluationFlags::values);
#else
        velocity.gather_evaluate(velocity_, false, true);
        pressure.gather_evaluate(pressure_, true, false);
#endif
        for (unsigned int q = 0; q < velocity.n_q_points; ++q) {
          const auto normal = velocity.get_normal_vector(q);

          const auto symmetric_gradient = velocity.get_symmetric_gradient(q);
          const auto divergence = trace(symmetric_gradient);

          const auto p = pressure.get_value(q);

          // S = (2 mu nabla^S(v) + (lambda - 2/3*mu) div(v) Id) - p * Id
          auto S = 2. * mu * symmetric_gradient;
          for (unsigned int d = 0; d < dim; ++d)
            S[d][d] += (lambda - 2. / 3. * mu) * divergence - p;

          velocity.submit_value(-S * normal, q);
        }

#if DEAL_II_VERSION_GTE(9, 3, 0)
        velocity.integrate_scatter(EvaluationFlags::values, boundary_stress_);
#else
        velocity.integrate_scatter(true, false, boundary_stress_);
#endif
      }
    }

    /*
     * Divide by lumped mass matrix:
     */

    {
      const auto &lumped_mass_matrix_inverse =
          offline_data_->lumped_mass_matrix_inverse();

      RYUJIN_PARALLEL_REGION_BEGIN
      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto m_i_inv = simd_load(lumped_mass_matrix_inverse, i);
        for (unsigned int d = 0; d < dim; ++d) {
          const auto f_i = simd_load(boundary_stress_.block(d), i);
          simd_store(boundary_stress_.block(d), m_i_inv * f_i, i);
          if constexpr (dim == 3) {
            const auto v_i = simd_load(vorticity_.block(d), i);
            simd_store(vorticity_.block(d), m_i_inv * v_i, i);
          }
        }
        if constexpr (dim == 2) {
          const auto v_i = simd_load(vorticity_.block(0), i);
          simd_store(vorticity_.block(0), m_i_inv * v_i, i);
        }
      }
      RYUJIN_PARALLEL_REGION_END

      for (unsigned int i = size_regular; i < n_owned; ++i) {
        const auto m_i_inv = lumped_mass_matrix_inverse.local_element(i);
        for (unsigned int d = 0; d < dim; ++d) {
          boundary_stress_.block(d).local_element(i) *= m_i_inv;
          if constexpr (dim == 3)
            vorticity_.block(d).local_element(i) *= m_i_inv;
        }
        if constexpr (dim == 2)
          vorticity_.block(0).local_element(i) *= m_i_inv;
      }

      vorticity_.update_ghost_values();
      boundary_stress_.update_ghost_values();
    }

    /*
     * Collect all boundary points of interest and output to log file:
     */

    {
      const auto &boundary_map = offline_data_->boundary_map();

      using entry = std::tuple<dealii::Point<dim> /*position*/,
                               dealii::Tensor<1, dim, Number> /*normal*/,
                               rank1_type /*state*/,
                               dealii::Tensor<1, dim, Number> /*stress*/>;

      std::vector<entry> entries;
      for (const auto &it : boundary_map) {

        /* Only record locally owned degrees of freedom */
        const auto i = it.first;
        if (i >= n_owned)
          continue;

        const auto &[normal, id, position] = it.second;

        const auto U_i = U.get_tensor(i);
        Tensor<1, dim, Number> Sn_i;
        for (unsigned int d = 0; d < dim; ++d)
          Sn_i[d] = boundary_stress_.block(d).local_element(i);

        entries.push_back({position, normal, U_i, Sn_i});
      }

      const auto all = Utilities::MPI::gather(mpi_communicator_, entries);

      if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
        std::ofstream output(name + "-boundary_values-" +
                             Utilities::to_string(cycle, 6) + ".log");

        output << std::scientific << std::setprecision(14) << t;
        output << "# t = " << t << std::endl;
        output << "position\tnormal\tstate (rho,M,E)\tstress" << std::endl;

        for (const auto &contribution : all) {
          for (const auto &entry : contribution) {
            const auto &[position, normal, U_i, Sn_i] = entry;
            output << position << "\t" << normal << "\t" << U_i << "\t" << Sn_i
                   << std::endl;
          }
        }
      }
    }

    // TODO
  }

} /* namespace ryujin */

#endif /* POINT_QUANTITIES_TEMPLATE_H */
