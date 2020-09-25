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
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;

    matrix_free_.reinit(offline_data_->discretization().mapping(),
                        offline_data_->dof_handler(),
                        offline_data_->affine_constraints(),
                        offline_data_->discretization().quadrature_1d(),
                        additional_data);

    /* Initialize vectors: */

    const auto &scalar_partitioner =
        matrix_free_.get_dof_info(0).vector_partitioner;

    velocity_.reinit(dim);
    velocity_interp_.reinit(dim);
    vorticity_.reinit(dim == 2 ? 1 : dim);
    boundary_stress_.reinit(dim);
    boundary_stress_interp_.reinit(dim);
    for (unsigned int i = 0; i < dim; ++i) {
      velocity_.block(i).reinit(scalar_partitioner);
      velocity_interp_.block(i).reinit(scalar_partitioner);
      if constexpr (dim == 3)
        vorticity_.block(i).reinit(scalar_partitioner);
      boundary_stress_.block(i).reinit(scalar_partitioner);
      boundary_stress_interp_.block(i).reinit(scalar_partitioner);
    }
    if constexpr (dim == 2)
      vorticity_.block(0).reinit(scalar_partitioner);

    lumped_boundary_mass_.reinit(scalar_partitioner);
    pressure_.reinit(scalar_partitioner);

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
        phi.integrate_scatter(true, false, lumped_boundary_mass_
#endif
      }

      lumped_boundary_mass_.update_ghost_values();
    }

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
          const auto &[normal, id, position] = entry.second;
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
  void PointQuantities<dim, Number>::compute(
      const vector_type &U,
      const Number t,
      const block_vector_type &velocity_interp,
      const Number t_interp,
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
     * Step 0: Copy velocity and compute pressure:
     */

    {
      for (unsigned int d = 0; d < dim; ++d) {
        velocity_interp_.block(d).copy_locally_owned_data_from(
            velocity_interp.block(d));
        velocity_interp_.block(d).update_ghost_values();
      }

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

      velocity_.update_ghost_values();
      velocity_interp_.update_ghost_values();
      pressure_.update_ghost_values();
    }

#if 0
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

      /* Divide by lumped mass matrix: */

      const auto &lumped_mass_matrix_inverse =
          offline_data_->lumped_mass_matrix_inverse();

      RYUJIN_PARALLEL_REGION_BEGIN
      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto m_i_inv = simd_load(lumped_mass_matrix_inverse, i);
        for (unsigned int d = 0; d < dim; ++d) {
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
          if constexpr (dim == 3)
            vorticity_.block(d).local_element(i) *= m_i_inv;
        }
        if constexpr (dim == 2)
          vorticity_.block(0).local_element(i) *= m_i_inv;
      }

      vorticity_.update_ghost_values();
    }
#endif

    /*
     * Step 2: Boundary stress:
     */

    {
      constexpr auto order_fe = Discretization<dim>::order_finite_element;
      constexpr auto order_quad = Discretization<dim>::order_quadrature;

      FEFaceEvaluation<dim, order_fe, order_quad, dim, Number> velocity(
          matrix_free_);
      FEFaceEvaluation<dim, order_fe, order_quad, dim, Number> velocity_interp(
          matrix_free_);

      boundary_stress_ = 0.;
      boundary_stress_interp_ = 0.;

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
        velocity_interp.reinit(face);
#if DEAL_II_VERSION_GTE(9, 3, 0)
        velocity.gather_evaluate(velocity_, EvaluationFlags::gradients);
        velocity_interp.gather_evaluate(velocity_interp_,
                                        EvaluationFlags::gradients);
#else
        velocity.gather_evaluate(velocity_, false, true);
        velocity_interp.gather_evaluate(velocity_interp_,
                                        EvaluationFlags::gradients);
#endif
        for (unsigned int q = 0; q < velocity.n_q_points; ++q) {
          const auto normal = velocity.get_normal_vector(q);
          {
            const auto symmetric_gradient = velocity.get_symmetric_gradient(q);
            const auto divergence = trace(symmetric_gradient);
            auto S = 2. * mu * symmetric_gradient;
            for (unsigned int d = 0; d < dim; ++d)
              S[d][d] += (lambda - 2. / 3. * mu) * divergence;
            velocity.submit_value(S * (-normal), q);
          }
          {
            const auto symmetric_gradient =
                velocity_interp.get_symmetric_gradient(q);
            const auto divergence = trace(symmetric_gradient);
            auto S = 2. * mu * symmetric_gradient;
            for (unsigned int d = 0; d < dim; ++d)
              S[d][d] += (lambda - 2. / 3. * mu) * divergence;
            velocity_interp.submit_value(S * (-normal), q);
          }
        }

#if DEAL_II_VERSION_GTE(9, 3, 0)
        velocity.integrate_scatter(EvaluationFlags::values, boundary_stress_);
        velocity_interp.integrate_scatter(EvaluationFlags::values,
                                          boundary_stress_interp_);
#else
        velocity.integrate_scatter(true, false, boundary_stress_);
        velocity_interp.integrate_scatter(true, false, boundary_stress_interp_);
#endif
      }
    }

    /*
     * Collect all boundary points of interest and output to log file:
     */

    {
      const auto &boundary_map = offline_data_->boundary_map();

      using entry = std::tuple<Point<dim>,              // position
                               Number,                  // lumped boundary mass
                               Tensor<1, dim, Number>,  // normal
                               rank1_type,              // state
                               Number,                  // pressure
                               Tensor<1, dim, Number>,  // stress
                               Tensor<1, dim, Number>>; // stress interp

      std::vector<entry> entries;
      for (const auto &it : boundary_map) {

        /* Only record locally owned degrees of freedom */
        const auto i = it.first;
        if (i >= n_owned)
          continue;

        const auto &[normal, id, position] = it.second;

        const auto U_i = U.get_tensor(i);
        const auto m_i = lumped_boundary_mass_.local_element(i);
        const auto P_i = pressure_.local_element(i);

        Tensor<1, dim, Number> Sn_i;
        Tensor<1, dim, Number> Sn_i_interp;
        for (unsigned int d = 0; d < dim; ++d) {
          Sn_i[d] = boundary_stress_.block(d).local_element(i);
          Sn_i_interp[d] = boundary_stress_interp_.block(d).local_element(i);
        }

        entries.push_back(
            {position, m_i, normal, U_i, P_i, Sn_i / m_i, Sn_i_interp / m_i});
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
        std::ofstream output(name + "-boundary_values-" +
                             Utilities::to_string(cycle, 6) + ".log");

        output << std::scientific << std::setprecision(14);
        output << "# stress_interp at time t = " << t_interp << std::endl;
        output << "# state and pressure at time t = " << t << std::endl;
        output << "# position\tlumped boundary mass\tnormal\t"
               << "state (rho,M,E)\tpressure\tstress\tstress_interp"
               << std::endl;

        for (const auto &entry : all_entries) {
          const auto &[position, m_i, normal, U_i, P_i, Sn_i, Sn_i_interp] =
              entry;
          output << position << "\t" << m_i << "\t" << normal << "\t"         //
                 << U_i << "\t" << P_i << "\t" << Sn_i << "\t" << Sn_i_interp //
                 << std::endl;
        }
      }
    }

    // TODO
  }

} /* namespace ryujin */

#endif /* POINT_QUANTITIES_TEMPLATE_H */
