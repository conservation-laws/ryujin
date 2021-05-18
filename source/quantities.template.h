//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "openmp.h"
#include "quantities.h"
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
  Quantities<dim, Number>::Quantities(
      const MPI_Comm &mpi_communicator,
      const ryujin::ProblemDescription &problem_description,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Quantities"*/)
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
  void Quantities<dim, Number>::prepare(std::string name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::prepare()" << std::endl;
#endif

    base_name_ = name;

    // FIXME
    AssertThrow(interior_manifolds_.empty(),
                dealii::ExcMessage("Not implemented. Output for interior "
                                   "manifolds has not been written yet."));

    const unsigned int n_owned = offline_data_->n_locally_owned();

    /*
     * Create boundary maps and allocate statistics vector:
     */

    boundary_maps_.clear();
    std::transform(
        boundary_manifolds_.begin(),
        boundary_manifolds_.end(),
        std::back_inserter(boundary_maps_),
        [this, n_owned](auto it) {
          const auto &[name, expression] = it;
          FunctionParser<dim> level_set_function(expression);

          std::vector<boundary_point> map;

          for (const auto &entry : offline_data_->boundary_map()) {
            /* skip nonlocal */
            if (entry.first >= n_owned)
              continue;

            /* skip constrained */
            if (offline_data_->affine_constraints().is_constrained(
                    offline_data_->scalar_partitioner()->local_to_global(
                        entry.first)))
              continue;

            const auto &[normal, normal_mass, boundary_mass, id, position] =
                entry.second;
            if (std::abs(level_set_function.value(position)) < 1.e-12)
              map.push_back({entry.first,
                             normal,
                             normal_mass,
                             boundary_mass,
                             id,
                             position});
          }
          return std::make_tuple(name, map);
        });

    boundary_statistics_.clear();
    boundary_statistics_.resize(boundary_manifolds_.size());

    for (std::size_t i = 0; i < boundary_maps_.size(); ++i) {
      const auto n_entries = std::get<1>(boundary_maps_[i]).size();
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          boundary_statistics_[i];
      val_old.resize(n_entries);
      val_new.resize(n_entries);
      val_sum.resize(n_entries);
      t_old = t_new = t_sum = 0.;
    }

    /*
     * Output boundary maps:
     */

    for (const auto &it : boundary_maps_) {
      const auto &[description, boundary_map] = it;

      /*
       * FIXME: This currently distributes boundary maps to all MPI ranks.
       * This is unnecessarily wasteful. Ideally, we should do MPI IO with
       * only MPI ranks participating who actually have boundary values.
       */

      const auto received =
          Utilities::MPI::gather(mpi_communicator_, boundary_map);

      if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

        std::ofstream output(base_name_ + "-" + description + "-points.log");
        output << std::scientific << std::setprecision(14);

        output << "# position\tnormal\tnormal mass\tboundary mass" << std::endl;

        unsigned int rank = 0;
        for (const auto &entries : received) {
          output << "# rank " << rank++ << std::endl;
          for (const auto &entry : entries) {
            const auto &[index, n_i, nm_i, bm_i, id, x_i] = entry;
            output << x_i << "\t" << n_i << "\t" << nm_i << "\t" << bm_i
                   << std::endl;
          } /*entry*/
        }   /*entries*/
      }
    }
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::accumulate(const vector_type &U, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::accumulate()" << std::endl;
#endif

    for (std::size_t i = 0; i < boundary_maps_.size(); ++i) {
      const auto boundary_map = std::get<1>(boundary_maps_[i]);
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          boundary_statistics_[i];

      /* accumulate new values */

      std::transform(boundary_map.begin(),
                     boundary_map.end(),
                     val_new.begin(),
                     [&](auto point) -> boundary_value {
                       const auto &[i, n_i, nm_i, bm_i, id, x_i] = point;

                       const auto U_i = U.get_tensor(i);
                       const auto rho_i = problem_description_->density(U_i);
                       const auto m_i = problem_description_->momentum(U_i);
                       const auto v_i = m_i / rho_i;
                       const auto p_i = problem_description_->pressure(U_i);

                       // FIXME: acquire symmetric diffusion tensor s(U)
                       // from dissipation module
                       const auto S_i = dealii::Tensor<2, dim, Number>();
                       const auto tau_n_i = S_i * n_i;

                       if constexpr (dim == 1)
                         return {
                             state_type{{rho_i, v_i[0], p_i}},
                             state_type{
                                 {rho_i * rho_i, v_i[0] * v_i[0], p_i * p_i}},
                             tau_n_i,
                             p_i * n_i};
                       else if constexpr (dim == 2)
                         return {state_type{{rho_i, v_i[0], v_i[1], p_i}},
                                 state_type{{rho_i * rho_i,
                                             v_i[0] * v_i[0],
                                             v_i[1] * v_i[1],
                                             p_i * p_i}},
                                 tau_n_i,
                                 p_i * n_i};
                       else
                         return {
                             state_type{{rho_i, v_i[0], v_i[1], v_i[2], p_i}},
                             state_type{{rho_i * rho_i,
                                         v_i[0] * v_i[0],
                                         v_i[1] * v_i[1],
                                         v_i[2] * v_i[2],
                                         p_i * p_i}},
                             tau_n_i,
                             p_i * n_i};
                     });

      if (RYUJIN_UNLIKELY(t_old == Number(0.) && t_new == Number(0.))) {
        /* We have not accumulated any statistics yet: */
        t_old = t - 1.;
        t_new = t;

      } else {

        t_new = t;
        const Number tau = t_new - t_old;

        for (std::size_t i = 0; i < val_sum.size(); ++i) {
          /* sometimes I miss haskell's type classes... */
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_old[i]);
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_new[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_old[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_new[i]);
          std::get<2>(val_sum[i]) += 0.5 * tau * std::get<2>(val_old[i]);
          std::get<2>(val_sum[i]) += 0.5 * tau * std::get<2>(val_new[i]);
          std::get<3>(val_sum[i]) += 0.5 * tau * std::get<3>(val_old[i]);
          std::get<3>(val_sum[i]) += 0.5 * tau * std::get<3>(val_new[i]);
        }
        t_sum += tau;
      }

      std::swap(t_old, t_new);
      std::swap(val_old, val_new);
    }

  }

} /* namespace ryujin */
