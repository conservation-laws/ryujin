//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "postprocessor.h"

#include <simd.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <atomic>
#include <chrono>
#include <fstream>

namespace ryujin
{
#ifndef DOXYGEN
  template <>
  const std::array<std::string, 1> Postprocessor<1, NUMBER>::component_names{
      {"schlieren"}};

  template <>
  const std::array<std::string, 2> Postprocessor<2, NUMBER>::component_names{
      {"schlieren", "vorticity"}};

  template <>
  const std::array<std::string, 2> Postprocessor<3, NUMBER>::component_names{
      {"schlieren", "vorticity"}};
#endif


  template <int dim, typename Number>
  Postprocessor<dim, Number>::Postprocessor(
      const MPI_Comm &mpi_communicator,
      const ryujin::HyperbolicSystem &hyperbolic_system,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Postprocessor"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , hyperbolic_system_(&hyperbolic_system)
      , offline_data_(&offline_data)
  {
    schlieren_beta_ = 10.;
    add_parameter(
        "schlieren beta",
        schlieren_beta_,
        "Beta factor used in the exponential scale for the schlieren plot");

    vorticity_beta_ = 10.;
    add_parameter(
        "vorticity beta",
        vorticity_beta_,
        "Beta factor used in the exponential scale for the vorticity");
  }


  template <int dim, typename Number>
  void Postprocessor<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Postprocessor<dim, Number>::prepare()" << std::endl;
#endif

    const auto &partitioner = offline_data_->scalar_partitioner();

    for (auto &it : quantities_)
      it.reinit(partitioner);
  }


  template <int dim, typename Number>
  void Postprocessor<dim, Number>::compute(const vector_type &U) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Postprocessor<dim, Number>::schedule_output()" << std::endl;
#endif

    const auto &affine_constraints = offline_data_->affine_constraints();

    constexpr auto simd_length = dealii::VectorizedArray<Number>::size();

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_map = offline_data_->boundary_map();

    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_locally_owned = offline_data_->n_locally_owned();


    /*
     * Step 2: Compute r_i and r_i_max, r_i_min:
     */

    std::atomic<Number> r_i_max{0.};
    std::atomic<Number> r_i_min{std::numeric_limits<Number>::infinity()};
    std::atomic<Number> v_i_max{0.};
    std::atomic<Number> v_i_min{std::numeric_limits<Number>::infinity()};

    {
      RYUJIN_PARALLEL_REGION_BEGIN

      Number r_i_max_on_subrange = 0.;
      Number r_i_min_on_subrange = std::numeric_limits<Number>::infinity();
      Number v_i_max_on_subrange = 0.;
      Number v_i_min_on_subrange = std::numeric_limits<Number>::infinity();

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_locally_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom */
        if (row_length == 1)
          continue;

        dealii::Tensor<1, dim, Number> grad_rho_i;
        curl_type curl_v_i;

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const auto j =
              *(i < n_internal ? js + col_idx * simd_length : js + col_idx);

          const auto U_j = U.get_tensor(j);
          const auto M_j = HyperbolicSystem::momentum(U_j);

          const auto c_ij = cij_matrix.get_tensor(i, col_idx);

          const auto rho_j = HyperbolicSystem::density(U_j);

          grad_rho_i += c_ij * rho_j;

          if constexpr (dim == 2) {
            curl_v_i[0] += cross_product_2d(c_ij) * M_j / rho_j;
          } else if constexpr (dim == 3) {
            curl_v_i += cross_product_3d(c_ij, M_j / rho_j);
          }
        }

        /* Fix up boundaries: */

        const auto range = boundary_map.equal_range(i);
        for (auto it = range.first; it != range.second; ++it) {
          const auto normal = std::get<0>(it->second);
          const auto id = std::get<3>(it->second);
          /* Remove normal components of the gradient on the boundary: */
          if (id == Boundary::slip || id == Boundary::no_slip) {
            grad_rho_i -= 1. * (grad_rho_i * normal) * normal;
          } else {
            grad_rho_i = 0.;
          }
          /* Only retain the normal component of the curl on the boundary: */
          if constexpr (dim == 2) {
            curl_v_i = 0.;
          } else if constexpr (dim == 3) {
            curl_v_i = (curl_v_i * normal) * normal;
          }
        }

        /* Populate quantities: */

        const Number m_i = lumped_mass_matrix.local_element(i);

        dealii::Tensor<1, n_quantities, Number> quantities;

        quantities[0] = grad_rho_i.norm() / m_i;
        if constexpr (dim == 2) {
          quantities[1] = curl_v_i[0] / m_i;
        } else if constexpr (dim == 3) {
          quantities[1] = curl_v_i.norm() / m_i;
        }

        r_i_max_on_subrange = std::max(r_i_max_on_subrange, quantities[0]);
        r_i_min_on_subrange = std::min(r_i_min_on_subrange, quantities[0]);
        if constexpr (dim > 1) {
          v_i_max_on_subrange =
              std::max(v_i_max_on_subrange, std::abs(quantities[1]));
          v_i_min_on_subrange =
              std::min(v_i_min_on_subrange, std::abs(quantities[1]));
        }

        for (unsigned int j = 0; j < n_quantities; ++j)
          quantities_[j].local_element(i) = quantities[j];
      }

      /* Synchronize over all threads: */

      Number temp = r_i_max.load();
      while (temp < r_i_max_on_subrange &&
             !r_i_max.compare_exchange_weak(temp, r_i_max_on_subrange))
        ;
      temp = r_i_min.load();
      while (temp > r_i_min_on_subrange &&
             !r_i_min.compare_exchange_weak(temp, r_i_min_on_subrange))
        ;

      temp = v_i_max.load();
      while (temp < v_i_max_on_subrange &&
             !v_i_max.compare_exchange_weak(temp, v_i_max_on_subrange))
        ;
      temp = v_i_min.load();
      while (temp > v_i_min_on_subrange &&
             !v_i_min.compare_exchange_weak(temp, v_i_min_on_subrange))
        ;

      RYUJIN_PARALLEL_REGION_END
    }

    /*
     * And synchronize over all processors: Add +-eps to avoid division by
     * zero in the exponentiation further down below.
     */

    constexpr auto eps = std::numeric_limits<Number>::epsilon();
    {
      namespace MPI = dealii::Utilities::MPI;
      r_i_max.store(MPI::max(r_i_max.load() + eps, mpi_communicator_));
      r_i_min.store(MPI::min(r_i_min.load() - eps, mpi_communicator_));
      v_i_max.store(MPI::max(v_i_max.load() + eps, mpi_communicator_));
      v_i_min.store(MPI::min(v_i_min.load() - eps, mpi_communicator_));
    }

    /*
     * Step 3: Normalize schlieren and vorticity:
     */

    {
      RYUJIN_PARALLEL_REGION_BEGIN

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_locally_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom */
        if (row_length == 1)
          continue;

        auto &r_i = quantities_[0].local_element(i);
        r_i = Number(1.) - std::exp(-schlieren_beta_ * (r_i - r_i_min) /
                                    (r_i_max - r_i_min));

        if constexpr (dim > 1) {
          auto &v_i = quantities_[1].local_element(i);
          const auto magnitude =
              Number(1.) -
              std::exp(-vorticity_beta_ * (std::abs(v_i) - v_i_min) /
                       (v_i_max - v_i_min));
          v_i = std::copysign(magnitude, v_i);
        }
      }

      RYUJIN_PARALLEL_REGION_END
    }

    /*
     * Step 4: Fix up constraints and distribute:
     */

    for (auto &it : quantities_) {
      affine_constraints.distribute(it);
      it.update_ghost_values();
    }
  }

} // namespace ryujin
