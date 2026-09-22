//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "computing_timer.h"
#include "quantities.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.templates.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace ryujin
{
  using namespace dealii;

  namespace
  {
    /**
     * Convert the raw moments of a single point (stored with n_selected
     * consecutive values per moment) in place into the mean and the
     * central moments of order 2, ..., n_moments.
     */
    template <typename Number>
    void to_central_moments(Number *moments,
                            const unsigned int n_moments,
                            const unsigned int n_selected)
    {
      for (unsigned int c = 0; c < n_selected; ++c) {
        const auto m = [&](const unsigned int k) -> Number & {
          return moments[(k - 1) * n_selected + c];
        };
        const Number m1 = m(1);

        /* Higher central moments are computed from raw moments first: */
        if (n_moments >= 4)
          m(4) +=
              -4. * m1 * m(3) + 6. * m1 * m1 * m(2) - 3. * m1 * m1 * m1 * m1;
        if (n_moments >= 3)
          m(3) += -3. * m1 * m(2) + 2. * m1 * m1 * m1;
        if (n_moments >= 2)
          m(2) -= m1 * m1;
      }
    }
  } // namespace


  template <typename Description, int dim, typename Number>
  Quantities<Description, dim, Number>::Quantities(
      const MPIEnsemble &mpi_ensemble,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const InitialPrecomputedVector &initial_precomputed,
      const std::string &subsection /*= "Quantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , extractor_(offline_data,
                   hyperbolic_system,
                   parabolic_system,
                   initial_precomputed)
      , base_name_("")
      , mesh_files_have_been_written_(false)
  {
    std::copy(std::begin(View::primitive_component_names),
              std::end(View::primitive_component_names),
              std::back_inserter(quantities_));

    add_parameter("quantities",
                  quantities_,
                  "List of conserved, primitive, precomputed, initial, or "
                  "parabolic quantities for which statistics are accumulated "
                  "on all manifolds.");

    n_moments_ = 2;
    add_parameter("number of moments",
                  n_moments_,
                  "Number of moments computed for every selected quantity in "
                  "the time averaged and space averaged statistics: 1 (mean), "
                  "2 (mean and variance), 3 (additionally the third central "
                  "moment), 4 (additionally the fourth central moment).");

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

    extractor_.prepare(quantities_);

    AssertThrow(1 <= n_moments_ && n_moments_ <= 4,
                dealii::ExcMessage("Invalid number of moments: \"" +
                                   std::to_string(n_moments_) +
                                   "\" is not in the range 1 to 4."));

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

    /*
     * Sort the points of a manifold lexicographically by position (and
     * normal) so that the output does not depend on the local numbering
     * of degrees of freedom:
     */

    const auto sort_points = [](std::vector<ManifoldPoint> &points) {
      const auto less = [](const auto &left, const auto &right) {
        for (unsigned int d = 0; d < dim; ++d)
          if (left[d] != right[d])
            return left[d] < right[d];
        return false;
      };
      std::sort(
          points.begin(), points.end(), [&](const auto &a, const auto &b) {
            const auto &[i_a, n_a, nm_a, m_a, id_a, x_a] = a;
            const auto &[i_b, n_b, nm_b, m_b, id_b, x_b] = b;
            return less(x_a, x_b) || (!less(x_b, x_a) && less(n_a, n_b));
          });
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

      /* We use a map to deduplicate the collected points: */
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

      sort_points(manifold.points);
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

      sort_points(manifold.points);
      manifolds_.push_back(std::move(manifold));
    }

    /* Clear statistics: */
    clear_statistics();

    /* Make sure we output new mesh files: */
    mesh_files_have_been_written_ = false;
  }


  template <typename Description, int dim, typename Number>
  unsigned int Quantities<Description, dim, Number>::stride() const
  {
    return n_moments_ * extractor_.n_selected();
  }


  template <typename Description, int dim, typename Number>
  std::string
  Quantities<Description, dim, Number>::header(const bool averaged) const
  {
    static const std::array<std::string, 4> labels{
        "mean", "var", "mu_3", "mu_4"};

    std::string result;
    for (unsigned int k = 0; k < (averaged ? n_moments_ : 1); ++k)
      for (const auto &name : quantities_)
        result += (result.empty() ? "" : "\t") +
                  (averaged ? labels[k] + "(" + name + ")" : name);
    return result + "\n";
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::accumulate(
      const StateVector &state_vector, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::accumulate()" << std::endl;
#endif

    prepare_extraction(state_vector);

    for (auto &manifold : manifolds_) {
      /* skip if we don't average in space or time: */
      if (!manifold.time_averaged && !manifold.space_averaged)
        continue;

      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          manifold.statistics;

      std::swap(t_old, t_new);
      std::swap(val_old, val_new);

      /* accumulate new values */

      auto spatial_average = internal_accumulate(manifold);

      /* Average in time with trapezoidal rule: */

      if (RYUJIN_UNLIKELY(t_old == Number(0.) && t_new == Number(0.))) {
        /* We have not accumulated any statistics yet: */
        t_old = t - 1.;
        t_new = t;

      } else {

        t_new = t;
        const Number tau = t_new - t_old;

        for (std::size_t i = 0; i < val_sum.size(); ++i)
          val_sum[i] += 0.5 * tau * (val_old[i] + val_new[i]);
        t_sum += tau;
      }

      /* Record average in space: */
      to_central_moments(
          spatial_average.data(), n_moments_, extractor_.n_selected());
      manifold.time_series.emplace_back(t, std::move(spatial_average));
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
     * Manifolds that are only output instantaneously have not been
     * evaluated in accumulate(). Prepare the extractor if we have to
     * evaluate any of them:
     */

    if (std::any_of(manifolds_.begin(), manifolds_.end(), [](const auto &m) {
          return m.instantaneous && !m.time_averaged && !m.space_averaged;
        }))
      prepare_extraction(state_vector);

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
          internal_accumulate(manifold);
        else
          AssertThrow(t_new == t, dealii::ExcInternalError());

        internal_write_out(file_name,
                           time_stamp.str(),
                           val_new,
                           Number(1.),
                           /*averaged*/ false);
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

          internal_write_out(file_name,
                             time_stamp.str(),
                             val_sum,
                             Number(1.) / t_sum,
                             /*averaged*/ true);
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
      }   /*entries*/

      output << std::flush;
    }
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::clear_statistics()
  {
    for (auto &manifold : manifolds_) {
      const auto n_entries = manifold.points.size() * stride();
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          manifold.statistics;
      val_old.assign(n_entries, Number(0.));
      val_new.assign(n_entries, Number(0.));
      val_sum.assign(n_entries, Number(0.));
      t_old = t_new = t_sum = 0.;
      manifold.time_series.clear();
    }
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::prepare_extraction(
      const StateVector &state_vector)
  {
    /* Ensure that the state vector is resident on the host memory space. */
    if constexpr (have_separate_memory_spaces) {
      ComputingTimer::Scope scope("time step [X] _ - memory space transfers");
      const auto &[U, precomputed, parabolic] = state_vector;
      U.template copy_to_memory_space<dealii::MemorySpace::Host>();
      precomputed.template copy_to_memory_space<dealii::MemorySpace::Host>();
    }

    extractor_.prepare_extraction(state_vector);
  }


  template <typename Description, int dim, typename Number>
  std::vector<Number>
  Quantities<Description, dim, Number>::internal_accumulate(Manifold &manifold)
  {
    const auto extractor_view = extractor_.view();
    const unsigned int n_selected = extractor_view.n_selected();
    const unsigned int stride = this->stride();

    std::vector<Number> values(n_selected);
    std::vector<Number> spatial_average(stride, Number(0.));
    Number mass_sum = Number(0.);

    auto *current = manifold.statistics.current.data();

    for (const auto &point : manifold.points) {
      const auto i = std::get<0>(point);
      const auto mass_i = std::get<3>(point);

      extractor_view.extract_element(values.data(), i);

      /* Store the raw moments, i.e., powers of the values: */
      for (unsigned int c = 0; c < n_selected; ++c) {
        Number power = Number(1.);
        for (unsigned int k = 0; k < n_moments_; ++k) {
          power *= values[c];
          current[k * n_selected + c] = power;
        }
      }

      for (unsigned int j = 0; j < stride; ++j)
        spatial_average[j] += mass_i * current[j];
      mass_sum += mass_i;

      current += stride;
    }

    /* synchronize MPI ranks (MPI Barrier): */

    mass_sum =
        Utilities::MPI::sum(mass_sum, mpi_ensemble_.ensemble_communicator());
    Utilities::MPI::sum(spatial_average,
                        mpi_ensemble_.ensemble_communicator(),
                        spatial_average);

    /* take average: */

    for (auto &it : spatial_average)
      it /= mass_sum;

    return spatial_average;
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::internal_write_out(
      const std::string &file_name,
      const std::string &time_stamp,
      const std::vector<Number> &values,
      const Number scale,
      const bool averaged)
  {
    /*
     * FIXME: This currently distributes all values to all MPI ranks. This
     * is unnecessarily wasteful. Ideally, we should do MPI IO with only
     * MPI ranks participating who actually have values.
     */

    auto received =
        Utilities::MPI::gather(mpi_ensemble_.ensemble_communicator(), values);

    if (Utilities::MPI::this_mpi_process(
            mpi_ensemble_.ensemble_communicator()) != 0)
      return;

    const unsigned int n_selected = extractor_.n_selected();
    const unsigned int stride = this->stride();

    /* For instantaneous values we only output the first moment: */
    const unsigned int n_columns = averaged ? stride : n_selected;

    std::ofstream output(file_name);
    output << std::scientific << std::setprecision(14);
    output << time_stamp << "# " << header(averaged);

    unsigned int rank = 0;
    for (auto &entries : received) {
      output << "# rank " << rank++ << "\n";
      for (std::size_t j = 0; j < entries.size(); j += stride) {
        auto *point = entries.data() + j;

        if (averaged) {
          for (unsigned int m = 0; m < stride; ++m)
            point[m] *= scale;
          to_central_moments(point, n_moments_, n_selected);
        }

        for (unsigned int m = 0; m < n_columns; ++m)
          output << (m == 0 ? "" : "\t") << point[m];
        output << "\n";
      }
    }

    output << std::flush;
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::internal_write_out_time_series(
      const std::string &file_name,
      const std::vector<std::pair<Number, std::vector<Number>>> &values,
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
      output << "# time t\t" << header(/*averaged*/ true);
    }

    for (const auto &[t, entry] : values) {
      output << t;
      for (const auto &value : entry)
        output << "\t" << value;
      output << "\n";
    }

    output << std::flush;
  }

} /* namespace ryujin */
