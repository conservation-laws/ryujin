//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "simd.h"
#include "vtu_output.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <atomic>
#include <chrono>
#include <fstream>

namespace ryujin
{
  using namespace dealii;

#ifndef DOXYGEN
  template <>
  const std::array<std::string, 2> VTUOutput<1, double>::component_names{
      {"schlieren", "residual_mu"}};

  template <>
  const std::array<std::string, 3> VTUOutput<2, double>::component_names{
      {"schlieren", "vorticity", "residual_mu"}};

  template <>
  const std::array<std::string, 3> VTUOutput<3, double>::component_names{
      {"schlieren", "vorticity", "residual_mu"}};

  template <>
  const std::array<std::string, 2> VTUOutput<1, float>::component_names{
      {"schlieren", "residual_mu"}};

  template <>
  const std::array<std::string, 3> VTUOutput<2, float>::component_names{
      {"schlieren", "vorticity", "residual_mu"}};

  template <>
  const std::array<std::string, 3> VTUOutput<3, float>::component_names{
      {"schlieren", "vorticity", "residual_mu"}};
#endif


  template <int dim, typename Number>
  VTUOutput<dim, Number>::VTUOutput(
      const MPI_Comm &mpi_communicator,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "VTUOutput"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , offline_data_(&offline_data)
  {
    use_mpi_io_ = false;
    add_parameter("use mpi io",
                  use_mpi_io_,
                  "If enabled write out one vtu file via MPI IO using "
                  "write_vtu_in_parallel() instead of independent output files "
                  "via write_vtu_with_pvtu_record()");


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

    add_parameter("manifolds",
                  manifolds_,
                  "List of level set functions. The description is used to "
                  "only output cells that intersect the given level set.");
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::prepare()" << std::endl;
#endif

    const auto &partitioner = offline_data_->scalar_partitioner();

    for (auto &it : quantities_)
      it.reinit(partitioner);
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::schedule_output(const vector_type &U,
                                               const scalar_type &residual_mu,
                                               std::string name,
                                               Number t,
                                               unsigned int cycle,
                                               bool output_full,
                                               bool output_levelsets)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::schedule_output()" << std::endl;
#endif

    constexpr auto simd_length = VectorizedArray<Number>::size();

    const auto &affine_constraints = offline_data_->affine_constraints();
    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_map = offline_data_->boundary_map();

    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_locally_owned = offline_data_->n_locally_owned();

    /*
     * Step 1: Copy state vector:
     */
    {
      unsigned int d = 0;
      for (auto &it : state_vector_) {
        it.reinit(offline_data_->scalar_partitioner());
        U.extract_component(it, d++);
        affine_constraints.distribute(it);
        it.update_ghost_values();
      }
    }

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

        Tensor<1, dim, Number> grad_rho_i;
        curl_type curl_v_i;

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const auto j =
              *(i < n_internal ? js + col_idx * simd_length : js + col_idx);

          state_type U_j;
          for (unsigned int d = 0; d < problem_dimension; ++d)
            U_j[d] = state_vector_[d].local_element(j);
          const auto M_j = ProblemDescription::momentum(U_j);

          const auto c_ij = cij_matrix.get_tensor(i, col_idx);

          const auto rho_j = ProblemDescription::density(U_j);

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
          const auto [normal, id, _] = it->second;
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

        Tensor<1, n_quantities, Number> quantities;

        quantities[0] = grad_rho_i.norm() / m_i;
        if constexpr (dim == 2) {
          quantities[1] = curl_v_i[0] / m_i;
        } else if constexpr (dim == 3) {
          quantities[1] = curl_v_i.norm() / m_i;
        }
        quantities[n_quantities - 1] = residual_mu.local_element(i);

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
    r_i_max.store(Utilities::MPI::max(r_i_max.load() + eps, mpi_communicator_));
    r_i_min.store(Utilities::MPI::min(r_i_min.load() - eps, mpi_communicator_));
    v_i_max.store(Utilities::MPI::max(v_i_max.load() + eps, mpi_communicator_));
    v_i_min.store(Utilities::MPI::min(v_i_min.load() - eps, mpi_communicator_));

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

    /*
     * Step 5: DataOut:
     */

    auto data_out = std::make_unique<dealii::DataOut<dim>>();
    auto data_out_levelsets = std::make_unique<dealii::DataOut<dim>>();

    const auto &discretization = offline_data_->discretization();
    const auto &mapping = discretization.mapping();
    const auto patch_order = discretization.finite_element().degree - 1;

    if (output_full) {
      data_out->attach_dof_handler(offline_data_->dof_handler());

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out->add_data_vector(state_vector_[i],
                                  ProblemDescription::component_names<dim>[i]);
      for (unsigned int i = 0; i < n_quantities; ++i)
        data_out->add_data_vector(quantities_[i], component_names[i]);

      data_out->build_patches(mapping, patch_order);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out->set_flags(flags);
    }

    if (output_levelsets && manifolds_.size() != 0) {
      data_out_levelsets->attach_dof_handler(offline_data_->dof_handler());

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out_levelsets->add_data_vector(
            state_vector_[i], ProblemDescription::component_names<dim>[i]);
      for (unsigned int i = 0; i < n_quantities; ++i)
        data_out_levelsets->add_data_vector(quantities_[i], component_names[i]);

      /*
       * Specify an output filter that selects only cells for output that are
       * in the viscinity of a specified set of output planes:
       */

      std::vector<std::shared_ptr<FunctionParser<dim>>> level_set_functions;
      for (const auto &expression : manifolds_)
        level_set_functions.emplace_back(
            std::make_shared<FunctionParser<dim>>(expression));

      data_out_levelsets->set_cell_selection(
          [level_set_functions](const auto &cell) {
            if (!cell->is_active() || cell->is_artificial())
              return false;

            for (const auto &function : level_set_functions) {

              unsigned int above = 0;
              unsigned int below = 0;

              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   ++v) {
                const auto vertex = cell->vertex(v);
                constexpr auto eps = std::numeric_limits<Number>::epsilon();
                if (function->value(vertex) >= 0. - 100. * eps)
                  above++;
                if (function->value(vertex) <= 0. + 100. * eps)
                  below++;
                if (above > 0 && below > 0)
                  return true;
              }
            }
            return false;
          });

      data_out_levelsets->build_patches(mapping, patch_order);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out_levelsets->set_flags(flags);
    }

    if (use_mpi_io_) {
      /* synchronous IO */
      if (output_full) {
        data_out->write_vtu_in_parallel(
            name + "_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_communicator_);
      }
      if (output_levelsets && manifolds_.size() != 0) {
        data_out_levelsets->write_vtu_in_parallel(
            name + "-levelsets_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_communicator_);
      }

    } else {

      /* schedule asynchronous writeback: */
      background_thread_status = std::async(
          std::launch::async,
          [=,
           data_out = std::move(data_out),
           data_out_levelsets = std::move(data_out_levelsets)]() mutable {
            if (output_full) {
              data_out->write_vtu_with_pvtu_record(
                  "", name, cycle, mpi_communicator_, 6);
            }
            if (output_levelsets && manifolds_.size() != 0) {
              data_out_levelsets->write_vtu_with_pvtu_record(
                  "", name + "-levelsets", cycle, mpi_communicator_, 6);
            }
            /* Explicitly delete pointer to free up memory early: */
            data_out.reset();
            data_out_levelsets.reset();
          });
    }
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::wait()
  {
    if (background_thread_status.valid())
      background_thread_status.wait();
  }


  template <int dim, typename Number>
  bool VTUOutput<dim, Number>::is_active()
  {
    if (!background_thread_status.valid())
      return false;

    return (std::future_status::ready !=
            background_thread_status.wait_for(std::chrono::nanoseconds(0)));
  }

} /* namespace ryujin */
