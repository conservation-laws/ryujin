//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "instrumentation.h"
#include "mesh_adaptor.h"
#include "mpi_ensemble.h"
#include "openmp.h"
#include "selected_components_extractor.h"
#include "simd.h"

#include <deal.II/base/array_view.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

#include "grid_refinement.h"

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
      , need_mesh_adaptation_(false)
      , initial_precomputed_(initial_precomputed)
      , alpha_(alpha)
  {
    adaptation_strategy_ = AdaptationStrategy::global_refinement;
    add_parameter("adaptation strategy",
                  adaptation_strategy_,
                  "The chosen adaptation strategy. Possible values are: global "
                  "refinement, random adaptation, kelly estimator");

    marking_strategy_ = MarkingStrategy::fixed_number;
    add_parameter("marking strategy",
                  marking_strategy_,
                  "The chosen marking strategy. Possible values are: fixed "
                  "number, fixed fraction");

    time_point_selection_strategy_ =
        TimePointSelectionStrategy::fixed_time_points;
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
        "kelly estimator: quantities",
        kelly_quantities_,
        "List of conserved, primitive or precomputed quantities that will be "
        "used for the Kelly error estimator for refinement and coarsening.");

    smoothness_widen_stencil_ = 0;
    add_parameter(
        "smoothness estimator: stencil size",
        smoothness_widen_stencil_,
        "Number of layers to widen the smoothness indicator stencil.");

    smoothness_lower_threshold_ = 0.075;
    add_parameter("smoothness estimator: lower threshold",
                  smoothness_lower_threshold_,
                  "Lower threshold of the smoothness indicator mapped to 0.");

    smoothness_upper_threshold_ = 0.50;
    add_parameter("smoothness estimator: upper threshold",
                  smoothness_upper_threshold_,
                  "Upper threshold of the smoothness indicator mapped to 1.");
    leave_subsection();

    /* Options for various marking strategies: */

    enter_subsection("marking strategies");
    refinement_fraction_ = 0.3;
    add_parameter("refinement fraction",
                  refinement_fraction_,
                  "Marking: fraction of cells selected for refinement.");


    refinement_tolerance_ = 0.25;
    add_parameter(
        "refinement tolerance",
        refinement_tolerance_,
        "Marking: normalized tolerance for selecting cells for refinement.");

    coarsening_tolerance_ = 0.125;
    add_parameter(
        "coarsening tolerance",
        coarsening_tolerance_,
        "Marking normalized tolerance for selecting cells for coarsening.");

    coarsening_fraction_ = 0.3;
    add_parameter("coarsening fraction",
                  coarsening_fraction_,
                  "Marking: fraction of cells selected for coarsening.");

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

    max_num_cells_ = 100000;
    add_parameter(
        "maximal number of cells",
        max_num_cells_,
        "Marking: maximal number of cells used for the fixed number "
        "strategy. Note this is only an indicator and not strictly enforced.");

    leave_subsection();

    /* Options for various time point selection strategies: */

    enter_subsection("time point selection strategies");
    adaptation_time_points_ = {};
    add_parameter("fixed time points",
                  adaptation_time_points_,
                  "List of time points in (simulation) time at which we will "
                  "perform a mesh adaptation cycle.");

    adaptation_cycle_interval_ = 5;
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

    if (adaptation_strategy_ == AdaptationStrategy::kelly_estimator) {
      SelectedComponentsExtractor<Description, dim, Number>::check(
          kelly_quantities_);
    }

    /* toggle mesh adaptation flag to off. */
    need_mesh_adaptation_ = false;
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::compute_random_indicators() const
  {
    std::generate(std::begin(indicators_), std::end(indicators_), [&]() {
      static std::uniform_real_distribution<double> distribution(0.0, 10.0);
      return distribution(mersenne_twister_);
    });
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::populate_selected_quantities(
      const StateVector &state_vector) const
  {
    /* Populate Kelly quantities: */
    const auto &affine_constraints = offline_data_->affine_constraints();

    selected_components_ =
        SelectedComponentsExtractor<Description, dim, Number>::extract(
            *hyperbolic_system_,
            state_vector,
            initial_precomputed_,
            alpha_,
            kelly_quantities_);

    for (auto &it : selected_components_) {
      affine_constraints.distribute(it);
      it.update_ghost_values();
    }
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::compute_kelly_indicators() const
  {
#if !DEAL_II_VERSION_GTE(9, 6, 0)
    AssertThrow(
        false,
        dealii::ExcMessage("The MeshAdaptor::compute_kelly_indicators() method "
                           "needs deal.II version 9.6.0 or newer"));
#else

    /*
     * Calculate a Kelly error estimator for each configured quantitity:
     */

    std::vector<dealii::Vector<float>> kelly_errors;
    std::vector<dealii::Vector<float> *> ptr_kelly_errors;

    const auto size = indicators_vec_[0].size();
    kelly_errors.resize(selected_components_.size());

    for (auto &it : kelly_errors) {
      it.reinit(size);
      ptr_kelly_errors.push_back(&it);
    }

    auto array_view_kelly_errors = dealii::make_array_view(ptr_kelly_errors);
    std::vector<const dealii::ReadVector<Number> *> ptr_kelly_components;
    for (const auto &it : selected_components_)
      ptr_kelly_components.push_back(&it);

    const auto array_view_kelly_components =
        dealii::make_array_view(ptr_kelly_components);

    // Workaround: select the first mapping
    const auto index = 0; // FIXME: come up with a strategy to get an
                          // appropriate index.
    dealii::KellyErrorEstimator<dim>::estimate(
        offline_data_->discretization().mapping()[index],
        offline_data_->dof_handler(),
        offline_data_->discretization().face_quadrature(),
        {},
        array_view_kelly_components,
        array_view_kelly_errors);

    for (unsigned int entry_index = 0;
         entry_index < selected_components_.size();
         ++entry_index)
      indicators_vec_[entry_index] = kelly_errors[entry_index];
#endif
  }


  template <typename Description, int dim, typename Number>
  void
  MeshAdaptor<Description, dim, Number>::compute_smoothness_indicators() const
  {
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();
    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    using VA = dealii::VectorizedArray<Number>;

    RYUJIN_PARALLEL_REGION_BEGIN

    auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
      using T = decltype(sentinel);
      unsigned int stride_size = get_stride_size<T>;

      /* Stored thread locally: */

      const unsigned int n_entries = selected_components_.size();

      const T inv_denom =
          T(1.) / T(smoothness_upper_threshold_ - smoothness_lower_threshold_);
      const T scale = T(0.5) / T(n_entries) * inv_denom;
      const T bias = T(smoothness_lower_threshold_) * inv_denom;

      RYUJIN_OMP_FOR
      for (unsigned int i = left; i < right; i += stride_size) {

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        std::vector<T> value_i(n_entries, T(0.));
        std::vector<T> numerator(n_entries, T(0.));
        std::vector<T> denominator(n_entries, T(0.));

        for (unsigned int k = 0; k < n_entries; ++k) {
          value_i[k] = get_entry<T>(selected_components_[k], i);
        }

        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += stride_size) {

          /* Skip diagonal. */
          if (col_idx == 0)
            continue;

          const auto beta_ij = betaij_matrix.template get_entry<T>(i, col_idx);

          for (unsigned int k = 0; k < n_entries; ++k) {
            const auto value_j_k = get_entry<T>(selected_components_[k], js);
            numerator[k] += beta_ij * (value_j_k - value_i[k]);
            denominator[k] +=
                std::abs(beta_ij) *
                std::max(std::abs(value_j_k), std::abs(value_i[k]));
          }
        }

        auto alpha_i = T(0.);
        for (unsigned int k = 0; k < n_entries; ++k) {
          alpha_i += std::abs(numerator[k]) / (T(1.e-6) + denominator[k]);
        }

        // FIXME: more sophisticated activation function?
        alpha_i *= scale;
        alpha_i -= bias;
        alpha_i = std::max(alpha_i, T(0.));
        alpha_i = std::min(alpha_i, T(1.));

        write_entry<T>(smoothness_indicator_, alpha_i, i);
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
      smoothness_indicator_.update_ghost_values();

      auto new_smoothness_indicator = smoothness_indicator_;

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

          auto alpha_i = get_entry<T>(smoothness_indicator_, i);

          const unsigned int *js = sparsity_simd.columns(i);
          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += stride_size) {

            /* Skip diagonal. */
            if (col_idx == 0)
              continue;

            const auto alpha_j = get_entry<T>(smoothness_indicator_, js);

            alpha_i = std::max(alpha_i, alpha_j);
          }

          write_entry<T>(new_smoothness_indicator, alpha_i, i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      RYUJIN_PARALLEL_REGION_END

      smoothness_indicator_ = new_smoothness_indicator;
    }

    smoothness_indicator_.update_ghost_values();
    const auto &affine_constraints = offline_data_->affine_constraints();
    affine_constraints.distribute(smoothness_indicator_);

    /*
     * Distribute to cells by taking a cell-wise average:
     */

    std::vector<dealii::types::global_dof_index> local_dof_indices;
    const auto &scalar_partitioner = offline_data_->scalar_partitioner();

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
        auto alpha_i = get_entry<Number>(smoothness_indicator_, local_i);
        alpha_cell += alpha_i;
      }
      alpha_cell *= scale;

      indicators_[cell->active_cell_index()] = static_cast<float>(alpha_cell);
    }
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

    case AdaptationStrategy::kelly_estimator:
      [[fallthrough]];
    case AdaptationStrategy::smoothness_estimator:
      populate_selected_quantities(state_vector);
      break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
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
     * Compute an indicator with the chosen adaptation strategy:
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
      compute_random_indicators();
    } break;

    case AdaptationStrategy::kelly_estimator: {
      indicators_vec_.resize(selected_components_.size());
      for (auto &entry : indicators_vec_) {
        entry.reinit(triangulation.n_active_cells());
        entry = 0;
      }

      compute_kelly_indicators();
    } break;

    case AdaptationStrategy::smoothness_estimator: {
      const auto &scalar_partitioner = offline_data_->scalar_partitioner();
      smoothness_indicator_.reinit(scalar_partitioner);
      indicators_.reinit(triangulation.n_active_cells());
      compute_smoothness_indicators();
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /*
     * Mark cells with chosen marking strategy:
     */

    if (adaptation_strategy_ != AdaptationStrategy::kelly_estimator)
      switch (marking_strategy_) {
      case MarkingStrategy::fixed_number: {
        dealii::parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_number(triangulation,
                                            indicators_,
                                            refinement_fraction_,
                                            coarsening_fraction_,
                                            max_num_cells_);
      } break;
      case MarkingStrategy::fixed_fraction: {
        dealii::parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_fraction(triangulation,
                                              indicators_,
                                              refinement_fraction_,
                                              coarsening_fraction_);
      } break;
      case MarkingStrategy::fixed_tolerance: {
        ryujin::GridMarking::refine_and_coarsen_fixed_tolerance(
            triangulation,
            indicators_,
            refinement_tolerance_,
            coarsening_tolerance_);
      } break;
      default:
        AssertThrow(false, dealii::ExcInternalError());
        __builtin_trap();
      }
    else
      switch (marking_strategy_) {
      case MarkingStrategy::fixed_number: {
        // harmon: refinement and coarsening by consensus
        dealii::parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_number(triangulation,
                                            indicators_vec_[0],
                                            refinement_fraction_,
                                            coarsening_fraction_,
                                            max_num_cells_);

        for (unsigned int entry_index = 1; entry_index < indicators_vec_.size();
             ++entry_index)
          dealii::parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_number(triangulation,
                                              indicators_vec_[entry_index],
                                              refinement_fraction_,
                                              0, // harmon: coarsen by consensus
                                              max_num_cells_);

        for (auto &cell : triangulation.active_cell_iterators())
          if (cell->refine_flag_set())
            cell->clear_coarsen_flag();
      } break;
      case MarkingStrategy::fixed_fraction: {
        // harmon: refinement and coarsening by consensus
        dealii::parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_fraction(triangulation,
                                              indicators_vec_[0],
                                              refinement_fraction_,
                                              coarsening_fraction_);

        for (unsigned int entry_index = 1; entry_index < indicators_vec_.size();
             ++entry_index)
          dealii::parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_fraction(
                  triangulation,
                  indicators_vec_[entry_index],
                  refinement_fraction_,
                  0); // harmon: coarsen by consensus


        for (auto &cell : triangulation.active_cell_iterators())
          if (cell->refine_flag_set())
            cell->clear_coarsen_flag();

      } break;
      case MarkingStrategy::fixed_tolerance: {
        GridMarking::refine_and_coarsen_fixed_tolerance_by_consensus(
            triangulation,
            indicators_vec_,
            refinement_tolerance_,
            coarsening_tolerance_);
        break;
      }

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
