//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "mesh_adaptor.h"
#include "mpi_ensemble.h"
#include "openmp.h"
#include "selected_components_extractor.h"
#include "simd.h"

#include <deal.II/base/array_view.h>
#include <deal.II/distributed/grid_refinement.h>

#include <boost/container/small_vector.hpp>

namespace ryujin
{
  template <typename Description, int dim, typename Number>
  MeshAdaptor<Description, dim, Number>::MeshAdaptor(
      const MPIEnsemble &mpi_ensemble,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const InitialPrecomputedVector &initial_precomputed,
      const ScalarVector &alpha,
      const std::string &subsection /*= "MeshAdaptor"*/)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , initial_precomputed_(initial_precomputed)
      , alpha_(alpha)
      , need_mesh_adaptation_(false)
  {
    adaptation_strategy_ = AdaptationStrategy::smoothness_indicators;
    add_parameter("adaptation strategy",
                  adaptation_strategy_,
                  "The chosen adaptation strategy. Possible values are: global "
                  "refinement, random adaptation, smoothness indicators");

    marking_strategy_ = MarkingStrategy::fixed_threshold;
    add_parameter(
        "marking strategy",
        marking_strategy_,
        "The chosen marking strategy. Possible values are: fixed threshold.");

    time_point_selection_strategy_ =
        TimePointSelectionStrategy::simulation_cycle;
    add_parameter("time point selection strategy",
                  time_point_selection_strategy_,
                  "The chosen time point selection strategy. Possible values "
                  "are: fixed time points, simulation cycle");

    /* Options for various adaptation strategies: */
    enter_subsection("adaptation strategies");
    random_adaptation_mersenne_twister_seed_ = 42u;
    add_parameter("random adaptation: mersenne_twister_seed",
                  random_adaptation_mersenne_twister_seed_,
                  "Seed for 64bit Mersenne Twister used for random refinement");

    add_parameter(
        "smoothness indicators: quantities",
        smoothness_selected_quantities_,
        "List of conserved, primitive or precomputed quantities that will be "
        "used for constructing the smoothness indicator.");

    smoothness_local_global_ratio_ = 0.5;
    add_parameter(
        "smoothness indicators: local global ratio",
        smoothness_local_global_ratio_,
        "Ratio between local and global denominator value used for normalizing "
        "the smoothness indicator. A value of 1 indicates pure global "
        "normalization, a value of 0 indicates pure local normalization.");

    smoothness_min_cutoff_ = 0.0;
    add_parameter("smoothness indicators: min cutoff",
                  smoothness_min_cutoff_,
                  "minimal cutoff for the smoothness indicator: values below "
                  "this threshold will be set to the cutoff value.");

    smoothness_max_cutoff_ = 1.0e16;
    add_parameter("smoothness indicators: max cutoff",
                  smoothness_max_cutoff_,
                  "minimal cutoff for the smoothness indicator: values above "
                  "this threshold will be set to the cutoff value.");

    smoothness_widen_stencil_ = 15;
    add_parameter(
        "smoothness indicators: stencil size",
        smoothness_widen_stencil_,
        "Number of layers to widen the smoothness indicator stencil.");
    leave_subsection();

    /* Options for various marking strategies: */

    enter_subsection("marking strategies");

    coarsening_threshold_ = 0.25;
    add_parameter(
        "coarsening threshold",
        coarsening_threshold_,
        "Marking: normalized or absolute threshold for selecting cells for "
        "coarsening (used in \"fixed threshhold\" marking strategy).");

    refinement_threshold_ = 0.75;
    add_parameter(
        "refinement threshold",
        refinement_threshold_,
        "Marking: normalized or absolute threshold for selecting cells for "
        "refinement (used in \"fixed threshold\" marking strategy).");

    absolute_threshold_ = false;
    add_parameter("absolute threshold",
                  absolute_threshold_,
                  "Marking: if set to true use an absolute threshold for the "
                  "\"refinement threshold\" and \"coarsening threshold\" "
                  "values instead of a relative one. If this parameter is set "
                  "to false then the smoothness indicator is normalized into "
                  "the number range of [0., 1] and the threshold parameters "
                  "are also expected to be a value in this interval.");

    min_refinement_level_ = 0;
    add_parameter("minimal refinement level",
                  min_refinement_level_,
                  "Marking: minimal refinement level of cells that will be "
                  "maintained while coarsening cells.");

    max_refinement_level_ = 1000;
    add_parameter("maximal refinement level",
                  max_refinement_level_,
                  "Marking: maximal refinement level of cells that will be "
                  "maintained while refininig cells.");
    leave_subsection();

    /* Options for various time point selection strategies: */

    enter_subsection("time point selection strategies");
    adaptation_time_points_ = {};
    add_parameter("fixed time points",
                  adaptation_time_points_,
                  "List of time points in (simulation) time at which we will "
                  "perform a mesh adaptation cycle.");

    adaptation_cycle_interval_ = 10;
    add_parameter("simulation cycle: interval",
                  adaptation_cycle_interval_,
                  "The nth simulation cycle at which we will "
                  "perform mesh adapation.");
    leave_subsection();

    const auto call_back = [this] {
      /* Initialize Mersenne Twister with configured seed: */
      mersenne_twister_.seed(random_adaptation_mersenne_twister_seed_);
    };

    call_back();
    ParameterAcceptor::parse_parameters_call_back.connect(call_back);
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::prepare(const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MeshAdaptor<dim, Number>::prepare()" << std::endl;
#endif

    if (time_point_selection_strategy_ ==
        TimePointSelectionStrategy::fixed_time_points) {
      /* Remove outdated refinement timestamps: */
      const auto new_end = std::remove_if(
          adaptation_time_points_.begin(),
          adaptation_time_points_.end(),
          [&](const Number &t_refinement) { return (t > t_refinement); });
      adaptation_time_points_.erase(new_end, adaptation_time_points_.end());
    }

    SelectedComponentsExtractor<Description, dim, Number>::check(
        parabolic_system_->parabolic_component_names(),
        {"alpha"},
        smoothness_selected_quantities_);

    /* toggle mesh adaptation flag to off. */
    need_mesh_adaptation_ = false;
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::compute_smoothness_indicators(
      const StateVector &state_vector) const
  {
    const auto &affine_constraints = offline_data_->affine_constraints();
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();
    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    using VA = dealii::VectorizedArray<Number>;

    /*
     * Extract selected quantities:
     */

    auto quantities =
        SelectedComponentsExtractor<Description, dim, Number>::extract(
            *offline_data_,
            *hyperbolic_system_,
            *parabolic_system_,
            state_vector,
            initial_precomputed_,
            {"alpha"},
            {alpha_},
            smoothness_selected_quantities_);

    for (auto &it : quantities) {
      it.update_ghost_values();
      affine_constraints.distribute(it);
      it.update_ghost_values();
    }

    /*
     * Set up temporary vectors:
     */

    const unsigned int n_entries = quantities.size();
    const auto &scalar_partitioner = offline_data_->scalar_partitioner();

    std::vector<ScalarVector> numerator(n_entries);
    std::vector<ScalarVector> denominator(std::max(1u, n_entries));
    for (auto &it : numerator)
      it.reinit(scalar_partitioner);
    for (auto &it : denominator)
      it.reinit(scalar_partitioner);

    /*
     * Commpute numerators and denominators for the smoothness indicators:
     */

    RYUJIN_PARALLEL_REGION_BEGIN

    auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
      using T = decltype(sentinel);
      unsigned int stride_size = get_stride_size<T>;

      RYUJIN_OMP_FOR
      for (unsigned int i = left; i < right; i += stride_size) {

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        boost::container::small_vector<T, 10> value_i(n_entries, T(0.));
        for (unsigned int k = 0; k < n_entries; ++k) {
          value_i[k] = get_entry<T>(quantities[k], i);
        }

        boost::container::small_vector<T, 10> numerator_i(n_entries, T(0.));
        boost::container::small_vector<T, 10> denominator_i(n_entries, T(0.));

        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += stride_size) {

          /* Skip diagonal. */
          if (col_idx == 0)
            continue;

          const auto beta_ij = betaij_matrix.template get_entry<T>(i, col_idx);

          for (unsigned int k = 0; k < n_entries; ++k) {
            const auto value_j_k = get_entry<T>(quantities[k], js);
            numerator_i[k] += beta_ij * (value_j_k - value_i[k]);
            denominator_i[k] +=
                std::abs(beta_ij) * //
                std::max(std::abs(value_j_k), std::abs(value_i[k]));
          }

          for (unsigned int k = 0; k < n_entries; ++k) {
            write_entry<T>(numerator[k], numerator_i[k], i);
            write_entry<T>(denominator[k], denominator_i[k], i);
          }
        }
      }
    };

    /* Parallel non-vectorized loop: */
    loop(Number(), n_internal, n_owned);
    /* Parallel vectorized SIMD loop: */
    loop(VA(), 0, n_internal);

    RYUJIN_PARALLEL_REGION_END

    /*
     * Normalize and populate smoothness_indicators_ vector:
     */

    smoothness_indicators_.reinit(scalar_partitioner);

    constexpr Number eps = std::numeric_limits<Number>::epsilon();

    std::vector<Number> denominator_global_maximum(n_entries);
    for (unsigned int k = 0; k < n_entries; ++k) {
      denominator_global_maximum[k] = dealii::Utilities::MPI::max(
          denominator[k].linfty_norm(), mpi_ensemble_.ensemble_communicator());

      denominator_global_maximum[k] =
          std::max(denominator_global_maximum[k], eps);
    }

    RYUJIN_PARALLEL_REGION_BEGIN

    auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
      using T = decltype(sentinel);
      unsigned int stride_size = get_stride_size<T>;

      RYUJIN_OMP_FOR
      for (unsigned int i = left; i < right; i += stride_size) {

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        auto alpha_i = T(0.);
        for (unsigned int k = 0; k < n_entries; ++k) {
          const auto numerator_i = get_entry<T>(numerator[k], i);
          const auto denominator_i = get_entry<T>(denominator[k], i);

          auto denominator =
              (Number(1.) - smoothness_local_global_ratio_) * denominator_i +
              smoothness_local_global_ratio_ * denominator_global_maximum[k];
          denominator = std::max(T(eps), denominator);

          alpha_i += std::abs(numerator_i) / denominator;
        }

        alpha_i = std::min(alpha_i, T(smoothness_max_cutoff_));
        alpha_i = std::max(alpha_i, T(smoothness_min_cutoff_));
        write_entry<T>(smoothness_indicators_, alpha_i, i);
      }
    };

    /* Parallel non-vectorized loop: */
    loop(Number(), n_internal, n_owned);
    /* Parallel vectorized SIMD loop: */
    loop(VA(), 0, n_internal);

    RYUJIN_PARALLEL_REGION_END

    /*
     * Widen indicators over stencil via max() operator:
     */

    for (unsigned int cycle = 0; cycle < smoothness_widen_stencil_; ++cycle) {
      smoothness_indicators_.update_ghost_values();

      RYUJIN_PARALLEL_REGION_BEGIN

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using ScalarNumber = typename get_value_type<Number>::type;

        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        /* Stored thread locally: */

        RYUJIN_OMP_FOR
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          auto alpha_i = get_entry<T>(smoothness_indicators_, i);

          const unsigned int *js = sparsity_simd.columns(i);
          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += stride_size) {

            /* Skip diagonal. */
            if (col_idx == 0)
              continue;

            const auto alpha_j = get_entry<T>(smoothness_indicators_, js);

            alpha_i = std::max(alpha_i, alpha_j);
          }

          write_entry<T>(/*SIC!*/ denominator[0], alpha_i, i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      RYUJIN_PARALLEL_REGION_END

      smoothness_indicators_ = /*SIC!*/ denominator[0];
    }

    smoothness_indicators_.update_ghost_values();
    affine_constraints.distribute(smoothness_indicators_);
    smoothness_indicators_.update_ghost_values();
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::analyze(
      const StateVector &state_vector, const Number t, unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MeshAdaptor<dim, Number>::analyze()" << std::endl;
#endif

    /*
     * Decide whether we perform an adaptation cycle with the chosen time
     * point selection strategy:
     */

    switch (time_point_selection_strategy_) {
    case TimePointSelectionStrategy::fixed_time_points: {
      /* Remove all refinement points from the vector that lie in the past: */
      const auto new_end = std::remove_if( //
          adaptation_time_points_.begin(),
          adaptation_time_points_.end(),
          [&](const Number &t_refinement) {
            if (t < t_refinement)
              return false;
            need_mesh_adaptation_ = true;
            return true;
          });
      adaptation_time_points_.erase(new_end, adaptation_time_points_.end());
    } break;

    case TimePointSelectionStrategy::simulation_cycle: {
      /* check whether we reached a cycle interval: */
      if (cycle % adaptation_cycle_interval_ == 0)
        need_mesh_adaptation_ = true;
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    if (!need_mesh_adaptation_)
      return;

    /*
     * Some adaptation strategies require us to prepare some internal
     * data fields:
     */

    switch (adaptation_strategy_) {
    case AdaptationStrategy::global_refinement:
      /* do nothing */
      break;

    case AdaptationStrategy::random_adaptation:
      /* do nothing */
      break;

    case AdaptationStrategy::smoothness_indicators: {
      compute_smoothness_indicators(state_vector);
      break;
    }

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::
      populate_cell_indicators_with_random_values() const
  {
    std::generate(std::begin(indicators_), std::end(indicators_), [&]() {
      static std::uniform_real_distribution<double> distribution(0.0, 10.0);
      return distribution(mersenne_twister_);
    });
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::
      populate_cell_indicators_from_smoothness_indicators() const
  {
    const auto &scalar_partitioner = offline_data_->scalar_partitioner();

    /*
     * Distribute to cells by taking a cell-wise average:
     */

    std::vector<dealii::types::global_dof_index> local_dof_indices;

    const auto &dof_handler = offline_data_->dof_handler();
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
      const auto scale = Number(1. / dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      auto alpha_cell = Number(0.);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const auto global_i = local_dof_indices[i];
        const auto local_i = scalar_partitioner->global_to_local(global_i);
        auto alpha_i = get_entry<Number>(smoothness_indicators_, local_i);
        alpha_cell += alpha_i;
      }
      alpha_cell *= scale;

      indicators_[cell->active_cell_index()] = static_cast<float>(alpha_cell);
    }
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::
      mark_cells_for_coarsening_and_refinement(
          dealii::Triangulation<dim> &triangulation [[maybe_unused]]) const
  {
    auto &discretization [[maybe_unused]] = offline_data_->discretization();
    Assert(&triangulation == &discretization.triangulation(),
           dealii::ExcInternalError());

#if !DEAL_II_VERSION_GTE(9, 6, 0)
    AssertThrow(
        false,
        dealii::ExcMessage(
            "The MeshAdaptor class needs deal.II version 9.6.0 or newer"));

#else

    /*
     * Compute cell indicators with the chosen adaptation strategy:
     */

    switch (adaptation_strategy_) {
    case AdaptationStrategy::global_refinement: {
      /* Simply mark all cells for refinement and return: */
      for (auto &cell : triangulation.active_cell_iterators())
        cell->set_refine_flag();
      return;
    } break;

    case AdaptationStrategy::random_adaptation: {
      indicators_.reinit(triangulation.n_active_cells());
      populate_cell_indicators_with_random_values();
    } break;

    case AdaptationStrategy::smoothness_indicators: {
      indicators_.reinit(triangulation.n_active_cells());
      populate_cell_indicators_from_smoothness_indicators();
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /*
     * Mark cells with chosen marking strategy:
     */

    switch (marking_strategy_) {
    case MarkingStrategy::fixed_threshold: {

      float inv_denominator = 1.f;
      float bias = 0.f;

      if (!absolute_threshold_) {
        /*
         * Normalize indicators to the interval [0., 1.]
         */

        float minimum = std::numeric_limits<float>::max();
        float maximum = 0.f;
        for (const auto &cell : triangulation.active_cell_iterators()) {
          if (!cell->is_locally_owned())
            continue;
          const auto indicator = indicators_[cell->active_cell_index()];
          minimum = std::min(minimum, indicator);
          maximum = std::max(maximum, indicator);
        }
        minimum = dealii::Utilities::MPI::min(
            minimum, mpi_ensemble_.ensemble_communicator());
        maximum = dealii::Utilities::MPI::max(
            maximum, mpi_ensemble_.ensemble_communicator());

        constexpr float eps = std::numeric_limits<float>::epsilon();
        // Ensure that if minimum == maximum we end up with 0.5 everywhere
        inv_denominator = 1.f / (maximum - minimum + 10.f * eps);
        bias = (minimum + 5.f * eps) * inv_denominator;
      }

      /*
       * And mark all cells according to threshold:
       */

      for (const auto &cell : triangulation.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;

        auto indicator = indicators_[cell->active_cell_index()];
        indicator = indicator * inv_denominator - bias;
        if (indicator < coarsening_threshold_)
          cell->set_coarsen_flag();
        else if (indicator > refinement_threshold_)
          cell->set_refine_flag();
      }
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /*
     * Constrain refinement and coarsening to maximum and minimum
     * refinement levels:
     */

    if (triangulation.n_levels() > max_refinement_level_)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_refinement_level_))
        cell->clear_refine_flag();

    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_refinement_level_))
      cell->clear_coarsen_flag();
#endif
  }
} // namespace ryujin
