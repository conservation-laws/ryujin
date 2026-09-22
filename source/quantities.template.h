//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "computing_timer.h"
#include "quantities.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.templates.h>

#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

namespace ryujin
{
  using namespace dealii;

  template <typename Description, int dim, typename Number>
  Quantities<Description, dim, Number>::Quantities(
      const MPIEnsemble &mpi_ensemble,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const std::string &subsection /*= "Quantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , base_name_("")
      , mesh_files_have_been_written_(false)
  {
    add_parameter("interior manifolds",
                  interior_manifolds_,
                  "List of level set functions describing interior manifolds. "
                  "The description is used to only output point values for "
                  "vertices belonging to a certain level set. "
                  "Format: '<name> : <level set formula> : <options> , [...] "
                  "(options: time_averaged, space_averaged, instantaneous)");

    add_parameter("boundary manifolds",
                  boundary_manifolds_,
                  "List of level set functions describing boundary. The "
                  "description is used to only output point values for "
                  "boundary vertices belonging to a certain level set. "
                  "Format: '<name> : <level set formula> : <options> , [...] "
                  "(options: time_averaged, space_averaged, instantaneous)");

    clear_temporal_statistics_on_writeout_ = true;
    add_parameter("clear statistics on writeout",
                  clear_temporal_statistics_on_writeout_,
                  "If set to true then all temporal statistics (for "
                  "\"time_averaged\" quantities) accumulated so far are reset "
                  "each time a writeout of quantities is performed");
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::prepare(const std::string &name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::prepare()" << std::endl;
#endif

    base_name_ = name;

    const unsigned int n_owned = offline_data_->n_locally_owned();
    const auto sparsity_simd_view =
        offline_data_->sparsity_pattern_simd().view();
    const auto lumped_mass_matrix_view =
        offline_data_->lumped_mass_matrix().view();

    /*
     * Create a manifold record with the name and parsed options for every
     * entry of the parameter lists:
     */

    const auto create_manifold = [](const auto &entry, const bool boundary) {
      const auto &[name, expression, options] = entry;

      Manifold manifold;
      manifold.name = name;
      manifold.boundary = boundary;
      manifold.instantaneous =
          options.find("instantaneous") != std::string::npos;
      manifold.time_averaged =
          options.find("time_averaged") != std::string::npos;
      manifold.space_averaged =
          options.find("space_averaged") != std::string::npos;

      AssertThrow(manifold.instantaneous || manifold.time_averaged ||
                      manifold.space_averaged,
                  dealii::ExcMessage(
                      "Invalid options \"" + options + "\" for manifold \"" +
                      name +
                      "\": at least one of instantaneous, time_averaged, or "
                      "space_averaged has to be selected."));

      return manifold;
    };

    manifolds_.clear();

    /*
     * Create interior manifolds: We have to loop over all cells and
     * collect all degrees of freedom satisfying the level set condition.
     */

    for (const auto &entry : interior_manifolds_) {
      auto manifold = create_manifold(entry, /*boundary*/ false);
      FunctionParser<dim> level_set_function(std::get<1>(entry));

      const auto &discretization = offline_data_->discretization();
      const auto &dof_handler = offline_data_->dof_handler();

      const auto support_points =
          dof_handler.get_fe().get_unit_support_points();

      std::vector<dealii::types::global_dof_index> local_dof_indices;

      /* We use a map to sort and deduplicate the collected points: */
      std::map<unsigned int, ManifoldPoint> preliminary_map;

      for (auto cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
        local_dof_indices.resize(dofs_per_cell);
        cell->get_active_or_mg_dof_indices(local_dof_indices);

        const auto &mapping = discretization.mapping()[cell->active_fe_index()];

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          const Point<dim> position =
              mapping.transform_unit_to_real_cell(cell, support_points[j]);

          if (std::abs(level_set_function.value(position)) > 1.e-12)
            continue;

          const auto global_index = local_dof_indices[j];
          const auto index =
              offline_data_->scalar_partitioner()->global_to_local(
                  global_index);

          /* Skip constrained degrees of freedom: */
          if (sparsity_simd_view.row_length(index) == 1)
            continue;

          if (index >= n_owned)
            continue;

          const Number mass = lumped_mass_matrix_view.read_entry(index);
          preliminary_map[index] = {index,
                                    dealii::Tensor<1, dim, Number>(),
                                    Number(0.),
                                    mass,
                                    dealii::numbers::internal_face_boundary_id,
                                    position};
        }
      }

      for (const auto &[index, point] : preliminary_map)
        manifold.points.push_back(point);

      manifolds_.push_back(std::move(manifold));
    }

    /*
     * Create boundary manifolds: We loop over the boundary map and collect
     * all degrees of freedom satisfying the level set condition.
     */

    for (const auto &entry : boundary_manifolds_) {
      auto manifold = create_manifold(entry, /*boundary*/ true);
      FunctionParser<dim> level_set_function(std::get<1>(entry));

      for (const auto &point : offline_data_->boundary_map()) {
        const auto &i = std::get<0>(point);

        /* skip nonlocal */
        if (i >= n_owned)
          continue;

        /* skip constrained */
        if (offline_data_->affine_constraints().is_constrained(
                offline_data_->scalar_partitioner()->local_to_global(i)))
          continue;

        const auto &position = std::get<5>(point);
        if (std::abs(level_set_function.value(position)) < 1.e-12)
          manifold.points.push_back(point);
      }

      manifolds_.push_back(std::move(manifold));
    }

    /* Clear statistics: */
    clear_statistics();

    /* Make sure we output new mesh files: */
    mesh_files_have_been_written_ = false;

    /* Prepare header string: */
    const auto &names = View::primitive_component_names;
    header_ = std::accumulate(
                  std::begin(names),
                  std::end(names),
                  std::string(),
                  [](const std::string &description, const std::string &name) {
                    return description.empty()
                               ? (std::string("primitive state (") + name)
                               : (description + ", " + name);
                  }) +
              ")\t and 2nd moments\n";
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::accumulate(
      const StateVector &state_vector, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::accumulate()" << std::endl;
#endif

    for (auto &manifold : manifolds_) {
      /* skip if we don't average in space or time: */
      if (!manifold.time_averaged && !manifold.space_averaged)
        continue;

      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          manifold.statistics;

      std::swap(t_old, t_new);
      std::swap(val_old, val_new);

      /* accumulate new values */

      const auto spatial_average = internal_accumulate(state_vector, manifold);

      /* Average in time with trapezoidal rule: */

      if (RYUJIN_UNLIKELY(t_old == Number(0.) && t_new == Number(0.))) {
        /* We have not accumulated any statistics yet: */
        t_old = t - 1.;
        t_new = t;

      } else {

        t_new = t;
        const Number tau = t_new - t_old;

        for (std::size_t i = 0; i < val_sum.size(); ++i) {
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_old[i]);
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_new[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_old[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_new[i]);
        }
        t_sum += tau;
      }

      /* Record average in space: */
      manifold.time_series.push_back({t, spatial_average});
    }
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::write_out(
      const StateVector &state_vector, const Number t, unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::write_out()" << std::endl;
#endif

    /*
     * First, write out mesh files if this hasn't happened yet.
     */
    if (!mesh_files_have_been_written_) {
      write_mesh_files(cycle);
      mesh_files_have_been_written_ = true;
    }

    /*
     * Next write out instantaneous and time_averaged maps, and flush the
     * space_averaged values to the corresponding log files:
     */

    for (auto &manifold : manifolds_) {
      const auto prefix = base_name_ + "-" + manifold.name + "-R" +
                          Utilities::to_string(cycle, 4);

      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          manifold.statistics;

      /*
       * Compute and output instantaneous field:
       */

      if (manifold.instantaneous) {
        const std::string file_name = prefix + "-instantaneous.dat";

        std::stringstream time_stamp;
        time_stamp << std::scientific << std::setprecision(14);
        time_stamp << "# at t = " << t << std::endl;

        /* We have not computed any updated statistics yet: */

        if (!manifold.time_averaged && !manifold.space_averaged)
          internal_accumulate(state_vector, manifold);
        else
          AssertThrow(t_new == t, dealii::ExcInternalError());

        internal_write_out(file_name, time_stamp.str(), val_new, Number(1.));
      }

      /*
       * Output time averaged field:
       */

      if (manifold.time_averaged) {
        const std::string file_name = prefix + "-time_averaged.dat";

        /* Check whether we have accumulated any statistics yet: */
        if (t_sum != Number(0.)) {
          std::stringstream time_stamp;
          time_stamp << std::scientific << std::setprecision(14);
          time_stamp << "# averaged from t = " << t_new - t_sum
                     << " to t = " << t_new << std::endl;

          internal_write_out(
              file_name, time_stamp.str(), val_sum, Number(1.) / t_sum);
        }
      }

      /*
       * Output space averaged field:
       */

      if (manifold.space_averaged) {
        /* Write to a new time series file after every call to prepare(): */
        bool append = true;
        if (!manifold.time_series_cycle.has_value()) {
          manifold.time_series_cycle = cycle;
          append = false;
        }

        const auto file_name =
            base_name_ + "-" + manifold.name + "-R" +
            Utilities::to_string(manifold.time_series_cycle.value(), 4) +
            "-space_averaged_time_series.dat";

        internal_write_out_time_series(
            file_name, manifold.time_series, /*append*/ append);
        manifold.time_series.clear();
      }
    }

    if (clear_temporal_statistics_on_writeout_)
      clear_statistics();
  }


  template <typename Description, int dim, typename Number>
  void
  Quantities<Description, dim, Number>::write_mesh_files(unsigned int cycle)
  {
    for (const auto &manifold : manifolds_) {
      /* Skip outputting the point map for spatial averages. */
      if (!manifold.instantaneous && !manifold.time_averaged)
        continue;

      /*
       * FIXME: This currently distributes point maps to all MPI ranks.
       * This is unnecessarily wasteful. Ideally, we should do MPI IO with
       * only MPI ranks participating who actually have values.
       */

      const auto received = Utilities::MPI::gather(
          mpi_ensemble_.ensemble_communicator(), manifold.points);

      if (Utilities::MPI::this_mpi_process(
              mpi_ensemble_.ensemble_communicator()) != 0)
        continue;

      std::ofstream output(base_name_ + "-" + manifold.name + "-R" +
                           Utilities::to_string(cycle, 4) + "-points.dat");

      output << std::scientific << std::setprecision(14);

      if (manifold.boundary)
        output << "#\n# position\tnormal\tnormal mass\tboundary mass\n";
      else
        output << "#\n# position\tinterior mass\n";

      unsigned int rank = 0;
      for (const auto &entries : received) {
        output << "# rank " << rank++ << "\n";
        for (const auto &entry : entries) {
          const auto &[index, n_i, nm_i, m_i, id, x_i] = entry;
          if (manifold.boundary)
            output << x_i << "\t" << n_i << "\t" << nm_i << "\t" << m_i << "\n";
          else
            output << x_i << "\t" << m_i << "\n";
        } /*entry*/
      } /*entries*/

      output << std::flush;
    }
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::clear_statistics()
  {
    for (auto &manifold : manifolds_) {
      const auto n_entries = manifold.points.size();
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          manifold.statistics;
      val_old.assign(n_entries, value_type());
      val_new.assign(n_entries, value_type());
      val_sum.assign(n_entries, value_type());
      t_old = t_new = t_sum = 0.;
      manifold.time_series.clear();
    }
  }


  template <typename Description, int dim, typename Number>
  auto Quantities<Description, dim, Number>::internal_accumulate(
      const StateVector &state_vector, Manifold &manifold) -> value_type
  {
    /* Ensure that the state vector is resident on the host memory space. */
    if constexpr (have_separate_memory_spaces) {
      ComputingTimer::Scope scope("time step [X] _ - memory space transfers");
      const auto &[U, precomputed, parabolic] = state_vector;
      U.template copy_to_memory_space<dealii::MemorySpace::Host>();
      precomputed.template copy_to_memory_space<dealii::MemorySpace::Host>();
    }

    const auto U_view = std::get<0>(state_vector).view();
    const auto view = hyperbolic_system_->template view<dim, Number>();

    auto &val_new = manifold.statistics.current;

    value_type spatial_average;
    Number mass_sum = Number(0.);

    std::transform(manifold.points.begin(),
                   manifold.points.end(),
                   val_new.begin(),
                   [&](const auto &point) -> value_type {
                     const auto i = std::get<0>(point);
                     const auto mass_i = std::get<3>(point);

                     const auto U_i = U_view.read_tensor(i);
                     const auto primitive_state = view.to_primitive_state(U_i);

                     value_type result;
                     std::get<0>(result) = primitive_state;
                     /* Compute second moments of the primitive state: */
                     std::get<1>(result) =
                         schur_product(primitive_state, primitive_state);

                     mass_sum += mass_i;
                     std::get<0>(spatial_average) +=
                         mass_i * std::get<0>(result);
                     std::get<1>(spatial_average) +=
                         mass_i * std::get<1>(result);

                     return result;
                   });

    /* synchronize MPI ranks (MPI Barrier): */

    mass_sum =
        Utilities::MPI::sum(mass_sum, mpi_ensemble_.ensemble_communicator());

    std::get<0>(spatial_average) = Utilities::MPI::sum(
        std::get<0>(spatial_average), mpi_ensemble_.ensemble_communicator());
    std::get<1>(spatial_average) = Utilities::MPI::sum(
        std::get<1>(spatial_average), mpi_ensemble_.ensemble_communicator());

    /* take average: */

    std::get<0>(spatial_average) /= mass_sum;
    std::get<1>(spatial_average) /= mass_sum;

    return spatial_average;
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::internal_write_out(
      const std::string &file_name,
      const std::string &time_stamp,
      const std::vector<value_type> &values,
      const Number scale)
  {
    /*
     * FIXME: This currently distributes all values to all MPI ranks. This
     * is unnecessarily wasteful. Ideally, we should do MPI IO with only
     * MPI ranks participating who actually have values.
     */

    const auto received =
        Utilities::MPI::gather(mpi_ensemble_.ensemble_communicator(), values);

    if (Utilities::MPI::this_mpi_process(
            mpi_ensemble_.ensemble_communicator()) != 0)
      return;

    std::ofstream output(file_name);
    output << std::scientific << std::setprecision(14);
    output << time_stamp << "# " << header_;

    unsigned int rank = 0;
    for (const auto &entries : received) {
      output << "# rank " << rank++ << "\n";
      for (const auto &entry : entries) {
        const auto &[state, state_square] = entry;
        output << scale * state << "\t" << scale * state_square << "\n";
      } /*entry*/
    }   /*entries*/

    output << std::flush;
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::internal_write_out_time_series(
      const std::string &file_name,
      const std::vector<std::pair<Number, value_type>> &values,
      bool append)
  {
    if (Utilities::MPI::this_mpi_process(
            mpi_ensemble_.ensemble_communicator()) != 0)
      return;

    std::ofstream output;
    output << std::scientific << std::setprecision(14);

    if (append) {
      output.open(file_name, std::ofstream::out | std::ofstream::app);
    } else {
      output.open(file_name, std::ofstream::out | std::ofstream::trunc);
      output << "# time t\t" << header_;
    }

    for (const auto &[t, value] : values) {
      const auto &[state, state_square] = value;
      output << t << "\t" << state << "\t" << state_square << "\n";
    }

    output << std::flush;
  }

} /* namespace ryujin */
