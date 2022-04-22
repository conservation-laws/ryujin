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
#include <deal.II/base/mpi.templates.h>
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
      , output_cycle_averages_(0)
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


  template <int dim, typename Number>
  void Quantities<dim, Number>::prepare(std::string name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::prepare()" << std::endl;
#endif

    base_name_ = name;
    first_cycle_ = true;

    const unsigned int n_owned = offline_data_->n_locally_owned();

    /*
     * Create interior maps and allocate statistics.
     *
     * We have to loop over the cells and populate the std::map interior_maps_.
     */

    interior_maps_.clear();
    std::transform(
        interior_manifolds_.begin(),
        interior_manifolds_.end(),
        std::inserter(interior_maps_, interior_maps_.end()),
        [this, n_owned](auto it) {
          const auto &[name, expression, option] = it;
          FunctionParser<dim> level_set_function(expression);

          std::vector<interior_point> map;
          std::map<int, interior_point> preliminary_map;

          const auto &discretization = offline_data_->discretization();
          const auto &dof_handler = offline_data_->dof_handler();

          const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

          const auto support_points =
              dof_handler.get_fe().get_unit_support_points();

          std::vector<dealii::types::global_dof_index> local_dof_indices(
              dofs_per_cell);

          /* Loop over cells */
          for (auto cell : dof_handler.active_cell_iterators()) {

            /* skip if not locally owned */
            if (!cell->is_locally_owned())
              continue;

            cell->get_active_or_mg_dof_indices(local_dof_indices);

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {

              Point<dim> position =
                  discretization.mapping().transform_unit_to_real_cell(
                      cell, support_points[j]);

              /*
               * Insert index, interior mass value and position into
               * a preliminary map if we satisfy level set condition.
               */

              if (std::abs(level_set_function.value(position)) > 1.e-12)
                continue;

              const auto global_index = local_dof_indices[j];
              const auto index =
                  offline_data_->scalar_partitioner()->global_to_local(
                      global_index);

              if (index >= n_owned)
                continue;

              const Number interior_mass =
                  offline_data_->lumped_mass_matrix().local_element(index);
              // FIXME: change to std::set
              preliminary_map[index] = {index, interior_mass, position};
            }
          }

          /*
           * Now we populate the std::vector(interior_point) object called map.
           */
          // FIXME: use std::copy
          for (const auto &[index, tuple] : preliminary_map) {
            map.push_back(tuple);
          }

          return std::make_pair(name, map);
        });

    /* Output interior maps: */

    for (const auto &[name, interior_map] : interior_maps_) {

      /*
       * FIXME: This currently distributes boundary maps to all MPI ranks.
       * This is unnecessarily wasteful. Ideally, we should do MPI IO with
       * only MPI ranks participating who actually have boundary values.
       */

      const auto received =
          Utilities::MPI::gather(mpi_communicator_, interior_map);

      if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

        std::ofstream output(base_name_ + "-" + name + "-R" +
                             std::to_string(output_cycle_averages_) +
                             "-points.dat");
        output << std::scientific << std::setprecision(14);

        output << "#\n# position\tinterior mass\n";

        unsigned int rank = 0;
        for (const auto &entries : received) {
          output << "# rank " << rank++ << "\n";
          for (const auto &entry : entries) {
            const auto &[index, mass_i, x_i] = entry;
            output << x_i << "\t" << mass_i << "\n";
          } /*entry*/
        }   /*entries*/

        output << std::flush;
      }
    }

    /*
     * Create boundary maps and allocate statistics vector:
     *
     * We want to loop over the boundary_map() once and populate the map
     * object boundary_maps_. We have to create a vector of
     * boundary_manifolds.size() that holds a std::vector<boundary_point>
     * for each map entry.
     */

    boundary_maps_.clear();
    std::transform(
        boundary_manifolds_.begin(),
        boundary_manifolds_.end(),
        std::inserter(boundary_maps_, boundary_maps_.end()),
        [this, n_owned](auto it) {
          const auto &[name, expression, option] = it;
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
          return std::make_pair(name, map);
        });

    /*
     * Output boundary maps:
     */

    for (const auto &[name, boundary_map] : boundary_maps_) {

      /*
       * FIXME: This currently distributes boundary maps to all MPI ranks.
       * This is unnecessarily wasteful. Ideally, we should do MPI IO with
       * only MPI ranks participating who actually have boundary values.
       */

      const auto received =
          Utilities::MPI::gather(mpi_communicator_, boundary_map);

      if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

        std::ofstream output(base_name_ + "-" + name + "-R" +
                             std::to_string(output_cycle_averages_) +
                             "-points.dat");

        output << std::scientific << std::setprecision(14);

        output << "#\n# position\tnormal\tnormal mass\tboundary mass\n";

        unsigned int rank = 0;
        for (const auto &entries : received) {
          output << "# rank " << rank++ << "\n";
          for (const auto &entry : entries) {
            const auto &[index, n_i, nm_i, bm_i, id, x_i] = entry;
            output << x_i << "\t" << n_i << "\t" << nm_i << "\t" << bm_i
                   << "\n";
          } /*entry*/
        }   /*entries*/

        output << std::flush;
      }
    }

    /* Clear statistics: */
    clear_statistics();

    /* Prepare header string: */
    const auto &names =
        problem_description_->template primitive_component_names<dim>;
    header_ =
        std::accumulate(std::begin(names),
                        std::end(names),
                        std::string(),
                        [](std::string &description, auto name) {
                          return description.empty()
                                     ? std::string("primitive state (") + name
                                     : description + ", " + name;
                        }) +
        ")\t and 2nd moments\n";
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::clear_statistics()
  {
    /* Clear statistics and time series for interior maps: */

    interior_statistics_.clear();

    for (const auto &[name, interior_map] : interior_maps_) {
      const auto n_entries = interior_map.size();
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          interior_statistics_[name];
      val_old.resize(n_entries);
      val_new.resize(n_entries);
      val_sum.resize(n_entries);
      t_old = t_new = t_sum = 0.;
    }

    /* Clear statistics and time series for boundary maps: */

    boundary_statistics_.clear();

    for (const auto &[name, boundary_map] : boundary_maps_) {
      const auto n_entries = boundary_map.size();
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          boundary_statistics_[name];
      val_old.resize(n_entries);
      val_new.resize(n_entries);
      val_sum.resize(n_entries);
      t_old = t_new = t_sum = 0.;
    }

    boundary_time_series_.clear();
  }



  template <int dim, typename Number>
  auto Quantities<dim, Number>::accumulate_interior(
      const vector_type &U,
      const std::vector<interior_point> &interior_map,
      std::vector<interior_value> &val_new) -> interior_value
  {
    interior_value spatial_average;
    Number mass_sum = Number(0.);

    std::transform(
        interior_map.begin(),
        interior_map.end(),
        val_new.begin(),
        [&](auto point) -> interior_value {
          const auto &[i, mass_i, x_i] = point;

          const auto U_i = U.get_tensor(i);
          const auto primitive_state =
              problem_description_->to_primitive_state(U_i);

          interior_value result;
          std::get<0>(result) = primitive_state;
          /* Compute second moments of the primitive state: */
          std::get<1>(result) = schur_product(primitive_state, primitive_state);

          mass_sum += mass_i;
          std::get<0>(spatial_average) += mass_i * std::get<0>(result);
          std::get<1>(spatial_average) += mass_i * std::get<1>(result);

          return result;
        });

    /* synchronize MPI ranks (MPI Barrier): */

    mass_sum = Utilities::MPI::sum(mass_sum, mpi_communicator_);

    std::get<0>(spatial_average) =
        Utilities::MPI::sum(std::get<0>(spatial_average), mpi_communicator_);
    std::get<1>(spatial_average) =
        Utilities::MPI::sum(std::get<1>(spatial_average), mpi_communicator_);

    /* take average: */

    std::get<0>(spatial_average) /= mass_sum;
    std::get<1>(spatial_average) /= mass_sum;

    return spatial_average;
  }



  template <int dim, typename Number>
  void Quantities<dim, Number>::write_out_interior(
      std::ostream &output,
      const std::vector<interior_value> &values,
      const Number scale)
  {
    /*
     * FIXME: This currently distributes interior maps to all MPI ranks.
     * This is unnecessarily wasteful. Ideally, we should do MPI IO with
     * only MPI ranks participating who actually have interior values.
     */

    const auto received = Utilities::MPI::gather(mpi_communicator_, values);

    if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

      output << "# " << header_;

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
  }


  template <int dim, typename Number>
  auto Quantities<dim, Number>::accumulate_boundary(
      const vector_type &U,
      const std::vector<boundary_point> &boundary_map,
      std::vector<boundary_value> &val_new) -> boundary_value
  {
    boundary_value spatial_average;
    Number boundary_mass_sum = Number(0.);

    std::transform(
        boundary_map.begin(),
        boundary_map.end(),
        val_new.begin(),
        [&](auto point) -> boundary_value {
          const auto &[i, n_i, nm_i, bm_i, id, x_i] = point;

          const auto U_i = U.get_tensor(i);
          const auto primitive_state =
              problem_description_->to_primitive_state(U_i);

          boundary_value result;
          std::get<0>(result) = primitive_state;
          /* Compute second moments of the primitive state: */
          std::get<1>(result) = schur_product(primitive_state, primitive_state);

          boundary_mass_sum += bm_i;
          std::get<0>(spatial_average) += bm_i * std::get<0>(result);
          std::get<1>(spatial_average) += bm_i * std::get<1>(result);

          return result;
        });

    /* synchronize MPI ranks (MPI Barrier): */

    boundary_mass_sum =
        Utilities::MPI::sum(boundary_mass_sum, mpi_communicator_);
    std::get<0>(spatial_average) =
        Utilities::MPI::sum(std::get<0>(spatial_average), mpi_communicator_);
    std::get<1>(spatial_average) =
        Utilities::MPI::sum(std::get<1>(spatial_average), mpi_communicator_);

    /* take average: */

    std::get<0>(spatial_average) /= boundary_mass_sum;
    std::get<1>(spatial_average) /= boundary_mass_sum;

    return spatial_average;
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::write_out_boundary(
      std::ostream &output,
      const std::vector<boundary_value> &values,
      const Number scale)
  {
    /*
     * FIXME: This currently distributes boundary maps to all MPI ranks.
     * This is unnecessarily wasteful. Ideally, we should do MPI IO with
     * only MPI ranks participating who actually have boundary values.
     */

    const auto received = Utilities::MPI::gather(mpi_communicator_, values);

    if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

      output << "# " << header_;

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
  }

  template <int dim, typename Number>
  void Quantities<dim, Number>::interior_write_out_time_series(
      std::ostream &output,
      const std::vector<std::tuple<Number, interior_value>> &values,
      bool append)
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

      if (!append)
        output << "# time t\t" << header_;

      for (const auto &entry : values) {
        const auto t = std::get<0>(entry);
        const auto &[state, state_square] = std::get<1>(entry);

        output << t << "\t" << state << "\t" << state_square << "\n";
      }

      output << std::flush;
    }
  }

  template <int dim, typename Number>
  void Quantities<dim, Number>::write_out_time_series(
      std::ostream &output,
      const std::vector<std::tuple<Number, boundary_value>> &values,
      bool append)
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

      if (!append)
        output << "# time t\t" << header_;

      for (const auto &entry : values) {
        const auto t = std::get<0>(entry);
        const auto &[state, state_square] = std::get<1>(entry);

        output << t << "\t" << state << "\t" << state_square << "\n";
      }

      output << std::flush;
    }
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::accumulate(const vector_type &U, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::accumulate()" << std::endl;
#endif

    /* For interior_maps_ */
    for (const auto &[name, interior_map] : interior_maps_) {

      /* Find the correct option string in interior_manifolds_ (a vector) */
      const auto it =
          std::find_if(interior_manifolds_.begin(),
                       interior_manifolds_.end(),
                       [&, name = std::cref(name)](const auto &element) {
                         return std::get<0>(element) == name.get();
                       });
      Assert(it != interior_manifolds_.end(), dealii::ExcInternalError());
      const auto options = std::get<2>(*it);

      /* skip if we don't average in space or time: */
      if (options.find("time_averaged") == std::string::npos &&
          options.find("space_averaged") == std::string::npos)
        continue;

      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          interior_statistics_[name];

      std::swap(t_old, t_new);
      std::swap(val_old, val_new);

      /* accumulate new values */

      const auto spatial_average =
          accumulate_interior(U, interior_map, val_new);

      /* Average in time with trapezoidal rule: */

      if (RYUJIN_UNLIKELY(t_old == Number(0.) && t_new == Number(0.))) {
        /* We have not accumulated any statistics yet: */
        t_old = t - 1.;
        t_new = t;

      } else {

        t_new = t;
        const Number tau = t_new - t_old;

        for (std::size_t i = 0; i < val_sum.size(); ++i) {
          /* sometimes I miss Haskell's type classes... */
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_old[i]);
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_new[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_old[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_new[i]);
        }
        t_sum += tau;
      }

      /* Record average in space: */
      interior_time_series_[name].push_back({t, spatial_average});
    }

    /* For Boundary maps */
    for (const auto &[name, boundary_map] : boundary_maps_) {

      /* Find the correct option string in boundary_manifolds_ (a vector) */
      const auto it =
          std::find_if(boundary_manifolds_.begin(),
                       boundary_manifolds_.end(),
                       [&, name = std::cref(name)](const auto &element) {
                         return std::get<0>(element) == name.get();
                       });
      Assert(it != boundary_manifolds_.end(), dealii::ExcInternalError());
      const auto options = std::get<2>(*it);

      /* skip if we don't average in space or time: */
      if (options.find("time_averaged") == std::string::npos &&
          options.find("space_averaged") == std::string::npos)
        continue;

      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          boundary_statistics_[name];

      std::swap(t_old, t_new);
      std::swap(val_old, val_new);

      /* accumulate new values */

      const auto spatial_average =
          accumulate_boundary(U, boundary_map, val_new);

      /* Average in time with trapezoidal rule: */

      if (RYUJIN_UNLIKELY(t_old == Number(0.) && t_new == Number(0.))) {
        /* We have not accumulated any statistics yet: */
        t_old = t - 1.;
        t_new = t;

      } else {

        t_new = t;
        const Number tau = t_new - t_old;

        for (std::size_t i = 0; i < val_sum.size(); ++i) {
          /* sometimes I miss Haskell's type classes... */
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_old[i]);
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_new[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_old[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_new[i]);
        }
        t_sum += tau;
      }

      /* Record average in space: */
      boundary_time_series_[name].push_back({t, spatial_average});
    }
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::write_out(const vector_type &U,
                                          const Number t,
                                          unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::write_out()" << std::endl;
#endif

    for (const auto &[name, interior_map] : interior_maps_) {

      /* Find the correct option string in interior_manifolds_ (a vector) */
      const auto it =
          std::find_if(interior_manifolds_.begin(),
                       interior_manifolds_.end(),
                       [&, name = std::cref(name)](const auto &element) {
                         return std::get<0>(element) == name.get();
                       });
      Assert(it != interior_manifolds_.end(), dealii::ExcInternalError());
      const auto options = std::get<2>(*it);

      /* Compute and output instantaneous field: */

      if (options.find("instantaneous") != std::string::npos) {

        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            interior_statistics_[name];

        /* We have not computed any updated statistics yet: */

        if (options.find("time_averaged") == std::string::npos &&
            options.find("space_averaged") == std::string::npos)
          accumulate_interior(U, interior_map, val_new);
        else
          AssertThrow(t_new == t, dealii::ExcInternalError());

        std::string file_name = base_name_ + "-" + name + "-instantaneous-" +
                                Utilities::to_string(cycle, 6) + ".dat";
        std::ofstream output(file_name);

        output << std::scientific << std::setprecision(14);
        output << "# at t = " << t << std::endl;

        write_out_interior(output, val_new, Number(1.));
      }

      /* Output time averaged field: */

      if (options.find("time_averaged") != std::string::npos) {

        std::string file_name = base_name_ + "-" + name + "-R" +
                                std::to_string(output_cycle_averages_) +
                                "-time_averaged.dat";
        std::ofstream output(file_name);

        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            interior_statistics_[name];

        output << std::scientific << std::setprecision(14);
        output << "# averaged from t = " << t_new - t_sum << " to t = " << t_new
               << std::endl;

        write_out_interior(output, val_sum, Number(1.) / t_sum);

      }

      /* Output space averaged field: */

      if (options.find("space_averaged") != std::string::npos) {

        auto &time_series = interior_time_series_[name];

        std::string file_name =
            base_name_ + "-" + name + "-space_averaged_time_series.dat";

        std::ofstream output;
        output << std::scientific << std::setprecision(14);

        if (first_cycle_) {
          first_cycle_ = false;
          output.open(file_name, std::ofstream::out | std::ofstream::trunc);
          interior_write_out_time_series(output, time_series, /*append*/ false);

        } else {

          output.open(file_name, std::ofstream::out | std::ofstream::app);
          interior_write_out_time_series(output, time_series, /*append*/ true);
        }

        time_series.clear();
      }

    } /* i */

    for (const auto &[name, boundary_map] : boundary_maps_) {

      /* Find the correct option string in boundary_manifolds_ (a vector) */
      const auto it =
          std::find_if(boundary_manifolds_.begin(),
                       boundary_manifolds_.end(),
                       [&, name = std::cref(name)](const auto &element) {
                         return std::get<0>(element) == name.get();
                       });
      Assert(it != boundary_manifolds_.end(), dealii::ExcInternalError());
      const auto options = std::get<2>(*it);

      /* Compute and output instantaneous field: */

      if (options.find("instantaneous") != std::string::npos) {

        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            boundary_statistics_[name];

        /* We have not computed any updated statistics yet: */

        if (options.find("time_averaged") == std::string::npos &&
            options.find("space_averaged") == std::string::npos)
          accumulate_boundary(U, boundary_map, val_new);
        else
          AssertThrow(t_new == t, dealii::ExcInternalError());

        std::string file_name = base_name_ + "-" + name + "-instantaneous-" +
                                Utilities::to_string(cycle, 6) + ".dat";
        std::ofstream output(file_name);

        output << std::scientific << std::setprecision(14);
        output << "# at t = " << t << std::endl;

        write_out_boundary(output, val_new, Number(1.));
      }

      /* Output time averaged field: */

      if (options.find("time_averaged") != std::string::npos) {

        std::string file_name = base_name_ + "-" + name + "-R" +
                                std::to_string(output_cycle_averages_) +
                                "-time_averaged.dat";
        std::ofstream output(file_name);

        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            boundary_statistics_[name];

        output << std::scientific << std::setprecision(14);
        output << "# averaged from t = " << t_new - t_sum << " to t = " << t_new
               << std::endl;

        write_out_boundary(output, val_sum, Number(1.) / t_sum);
      }

      /* Output space averaged field: */

      if (options.find("space_averaged") != std::string::npos) {

        auto &time_series = boundary_time_series_[name];

        std::string file_name =
            base_name_ + "-" + name + "-space_averaged_time_series.dat";

        std::ofstream output;
        output << std::scientific << std::setprecision(14);

        if (first_cycle_) {
          first_cycle_ = false;
          output.open(file_name, std::ofstream::out | std::ofstream::trunc);
          write_out_time_series(output, time_series, /*append*/ false);

        } else {

          output.open(file_name, std::ofstream::out | std::ofstream::app);
          write_out_time_series(output, time_series, /*append*/ true);
        }

        time_series.clear();
      }

    } /* i */

    output_cycle_averages_++;

    if(clear_temporal_statistics_on_writeout_)
      clear_statistics();
  }

} /* namespace ryujin */
