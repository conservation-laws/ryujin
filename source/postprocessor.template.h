//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "postprocessor.h"

#include <simd.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

namespace ryujin
{
  template <typename Description, int dim, typename Number>
  Postprocessor<Description, dim, Number>::Postprocessor(
      const MPI_Comm &mpi_communicator,
      const HyperbolicSystem &hyperbolic_system,
      const OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Postprocessor"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , hyperbolic_system_(&hyperbolic_system)
      , offline_data_(&offline_data)
  {
    beta_ = 10.;
    add_parameter("schlieren beta",
                  beta_,
                  "Beta factor used in the exponential scale for the schlieren "
                  "and vorticity plots");

    recompute_bounds_ = true;
    add_parameter(
        "schlieren recompute bounds",
        recompute_bounds_,
        "Recompute bounds for every output cycle. If set to false, bounds once "
        "at the beginning and reused thereafter.");

    static_assert(View::component_names.size() > 0,
                  "Need at least one scalar quantitity");
    schlieren_quantities_.push_back(View::component_names[0]);

    add_parameter(
        "schlieren quantities",
        schlieren_quantities_,
        "List of conserved quantities used for the schlieren postprocessor.");

    if constexpr (dim > 1) {
      add_parameter(
          "vorticity quantities",
          vorticity_quantities_,
          "List of conserved quantities used for the vorticity postprocessor.");
    }
  }


  template <typename Description, int dim, typename Number>
  void Postprocessor<Description, dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Postprocessor<dim, Number>::prepare()" << std::endl;
#endif

    bounds_.clear();
    component_names_.clear();
    schlieren_indices_.clear();
    vorticity_indices_.clear();

    const auto populate = [&](const auto &strings,
                              auto &indices,
                              const auto &pre) {
      const auto &cons = View::component_names;
      const auto &prim = View::primitive_component_names;
      for (const auto &entry : strings) {
        bool found = false;
        for (const auto &[is_primitive, names] :
             {std::make_pair(false, cons), std::make_pair(true, prim)}) {
          const auto pos = std::find(std::begin(names), std::end(names), entry);
          if (!found && pos != std::end(names)) {
            const auto index = std::distance(std::begin(names), pos);
            indices.push_back(std::make_pair(is_primitive, index));
            component_names_.push_back(pre + entry);
            found = true;
          }
        }
        AssertThrow(
            found,
            dealii::ExcMessage("Invalid component name »" + entry + "«"));
      }
    };
    populate(schlieren_quantities_, schlieren_indices_, "schlieren_");
    populate(vorticity_quantities_, vorticity_indices_, "vorticity_");

    const auto &partitioner = offline_data_->scalar_partitioner();

    quantities_.resize(component_names_.size());
    for (auto &it : quantities_)
      it.reinit(partitioner);
  }


  template <typename Description, int dim, typename Number>
  void
  Postprocessor<Description, dim, Number>::compute(const vector_type &U) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Postprocessor<dim, Number>::compute()" << std::endl;
#endif

    using VA = dealii::VectorizedArray<Number>;

    const auto &affine_constraints = offline_data_->affine_constraints();

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_map = offline_data_->boundary_map();

    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    const unsigned int n_schlieren = schlieren_indices_.size();
    Assert(n_schlieren == schlieren_quantities_.size(),
           dealii::ExcInternalError());
    const unsigned int n_vorticities = vorticity_indices_.size();
    Assert(n_vorticities == vorticity_quantities_.size(),
           dealii::ExcInternalError());
    const unsigned int n_quantities = n_schlieren + n_vorticities;
    Assert(n_quantities == quantities_.size(), dealii::ExcInternalError());
    Assert(n_quantities == component_names_.size(), dealii::ExcInternalError());

    /*
     * Step 1: Compute quantities:
     */

    {
      RYUJIN_PARALLEL_REGION_BEGIN

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        std::vector<grad_type<T>> local_schlieren_values(n_schlieren);
        std::vector<curl_type<T>> local_vorticity_values(n_vorticities);

        RYUJIN_OMP_FOR
        for (unsigned int i = left; i < right; i += stride_size) {

          for (auto &it : local_schlieren_values)
            it = grad_type<T>();
          for (auto &it : local_vorticity_values)
            it = curl_type<T>();

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          const unsigned int *js = sparsity_simd.columns(i);
          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = U.template get_tensor<T>(js);
            const auto view = hyperbolic_system_->template view<dim, T>();
            const auto prim_j = view.to_primitive_state(U_j);

            const auto c_ij = cij_matrix.template get_tensor<T>(i, col_idx);

            unsigned int k = 0;
            for (const auto &[is_primitive, index] : schlieren_indices_) {
              local_schlieren_values[k++] -=
                  c_ij * (is_primitive ? prim_j[index] : U_j[index]);
            }

            k = 0;
            for (const auto &[is_primitive, index] : vorticity_indices_) {
              grad_type<T> q_j;
              for (unsigned int d = 0; d < dim; ++d)
                q_j[d] = (is_primitive ? prim_j[index + d] : U_j[index + d]);

              if constexpr (dim == 2) {
                local_vorticity_values[k++][0] -= cross_product_2d(c_ij) * q_j;
              } else if constexpr (dim == 3) {
                local_vorticity_values[k++] -= cross_product_3d(c_ij, q_j);
              }
            }
          }

          /* Fix up boundaries: */

          /* Serialize over the stride: */
          for (unsigned int k = 0; k < stride_size; ++k) {
            const auto range = boundary_map.equal_range(i + k);
            for (auto it = range.first; it != range.second; ++it) {
              const auto normal = std::get<0>(it->second);
              const auto id = std::get<3>(it->second);

              if (id == Boundary::slip || id == Boundary::no_slip) {
                /* Remove normal component of schlieren values at slip
                 * and no-slip boundaries: */
                for (auto &it : local_schlieren_values) {
                  auto grad_q_i = serialize_tensor(it, k);
                  grad_q_i -= 1. * (grad_q_i * normal) * normal;
                  assign_serial_tensor(it, grad_q_i, k);
                }

              } else {
                /* Set schlieren values to zero everywhere else: */
                for (auto &it : local_schlieren_values) {
                  assign_serial_tensor(it, grad_type<Number>(), k);
                }
              }

              for (auto &it : local_vorticity_values) {
                /* Retain only the normal component of the curl on the
                 * boundary: */
                if constexpr (dim == 2) {
                  assign_serial_tensor(it, curl_type<Number>(), k);
                } else if constexpr (dim == 3) {
                  auto curl_q_i = serialize_tensor(it, k);
                  curl_q_i = (curl_q_i * normal) * normal;
                  assign_serial_tensor(it, curl_q_i, k);
                }
              }
            }
          }

          /* Populate quantities: */

          const auto m_i = get_entry<T>(lumped_mass_matrix, i);

          unsigned int k = 0;

          for (const auto &schlieren : local_schlieren_values) {
            const auto value_i = schlieren.norm() / m_i;
            write_entry<T>(quantities_[k++], value_i, i);
          }

          for (const auto &vorticity : local_vorticity_values) {
            auto value_i =
                (dim == 2 ? vorticity[0] / m_i : vorticity.norm() / m_i);
            write_entry<T>(quantities_[k++], value_i, i);
          }
        } /* i */
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      RYUJIN_PARALLEL_REGION_END
    }

    /*
     * Step 2: Compute bounds and synchronize over MPI ranks:
     */

    /* Force recomputation of bounds: */
    if (recompute_bounds_)
      bounds_.clear();

    if (bounds_.size() != n_quantities) {
      bounds_.clear();
      bounds_.resize(
          n_quantities,
          std::make_pair(Number(0.), std::numeric_limits<Number>::max()));

      for (unsigned int d = 0; d < n_quantities; ++d) {
        auto &[q_max, q_min] = bounds_[d];
        for (unsigned int i = 0; i < n_owned; ++i) {
          const auto q = quantities_[d].local_element(i);
          q_max = std::max(q_max, std::abs(q));
          q_min = std::min(q_min, std::abs(q));
        }
        q_max = dealii::Utilities::MPI::max(q_max, mpi_communicator_);
        q_min = dealii::Utilities::MPI::min(q_min, mpi_communicator_);
        Assert(q_max >= q_min, dealii::ExcInternalError());
      }
    }

    /*
     * Step 3: Normalize quantities on exponential scale:
     */

    {
      for (unsigned int d = 0; d < n_quantities; ++d) {
        auto &[q_max, q_min] = bounds_[d];
        for (unsigned int i = 0; i < n_owned; ++i) {
          auto &q = quantities_[d].local_element(i);
          constexpr auto eps = std::numeric_limits<Number>::epsilon();
          const auto magnitude =
              Number(1.) -
              std::exp(-beta_ * (std::abs(q) - q_min) / (q_max - q_min + eps));
          q = std::copysign(magnitude, q);
        }
      }
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
