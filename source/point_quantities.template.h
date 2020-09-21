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

    matrix_free_.reinit(offline_data_->discretization().mapping(),
                        offline_data_->dof_handler(),
                        offline_data_->affine_constraints(),
                        offline_data_->discretization().quadrature_1d());

    const auto &scalar_partitioner =
        matrix_free_.get_dof_info(0).vector_partitioner;

    velocity_.reinit(dim);
    boundary_stress_.reinit(dim);
    if constexpr (dim >= 2)
      vorticity_.reinit(dim == 2 ? 1 : dim);
    for (unsigned int i = 0; i < dim; ++i) {
      velocity_.block(i).reinit(scalar_partitioner);
      boundary_stress_.block(i).reinit(scalar_partitioner);
      if constexpr (dim == 3)
        vorticity_.block(i).reinit(scalar_partitioner);
    }
    if constexpr (dim == 2)
      vorticity_.block(0).reinit(scalar_partitioner);
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
          [this](const auto &data,
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
    }

    /*
     * Step 2: Compute vorticity:
     */
    {
    }
  }


} /* namespace ryujin */

#endif /* POINT_QUANTITIES_TEMPLATE_H */
