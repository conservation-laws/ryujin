//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef POINT_QUANTITIES_TEMPLATE_H
#define POINT_QUANTITIES_TEMPLATE_H

#include "point_quantities.h"
#include "openmp.h"
#include "scope.h"
#include "simd.h"

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
  }


  template <int dim, typename Number>
  void PointQuantities<dim, Number>::compute(const vector_type &U, Number t)
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

      vorticity_.update_ghost_values();
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

          velocity.submit_value(S * normal, q);
        }

#if DEAL_II_VERSION_GTE(9, 3, 0)
        velocity.integrate_scatter(EvaluationFlags::values, boundary_stress_);
#else
        velocity.integrate_scatter(true, false, boundary_stress_);
#endif
      }

      boundary_stress_.update_ghost_values();
    }

    // DEBUG
    {
      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(offline_data_->dof_handler());
      data_out.add_data_vector(boundary_stress_.block(0), "stress_0");
      data_out.add_data_vector(boundary_stress_.block(1), "stress_1");
      const auto &discretization = offline_data_->discretization();
      const auto &mapping = discretization.mapping();
      const auto patch_order = discretization.finite_element().degree - 1;
      data_out.build_patches(mapping, patch_order);
      data_out.write_vtu_in_parallel("stress-" + std::to_string(t) + ".vtu",
                                     mpi_communicator_);
    }
  }


} /* namespace ryujin */

#endif /* POINT_QUANTITIES_TEMPLATE_H */
