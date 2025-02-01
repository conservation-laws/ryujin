//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "mesh_adaptor.h"
#include "selected_components_extractor.h"

#include <deal.II/base/array_view.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

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
    leave_subsection();

    /* Options for various marking strategies: */

    enter_subsection("marking strategies");
    refinement_fraction_ = 0.3;
    add_parameter("refinement fraction",
                  refinement_fraction_,
                  "Marking: fraction of cells selected for refinement.");

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
  void MeshAdaptor<Description, dim, Number>::populate_kelly_quantities(
      const StateVector &state_vector) const
  {
    /* Populate Kelly quantities: */
    const auto &affine_constraints = offline_data_->affine_constraints();

    kelly_components_ =
        SelectedComponentsExtractor<Description, dim, Number>::extract(
            *hyperbolic_system_,
            state_vector,
            initial_precomputed_,
            alpha_,
            kelly_quantities_);

    for (auto &it : kelly_components_) {
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

    const auto size = indicators_.size();
    kelly_errors.resize(kelly_components_.size());

    for (auto &it : kelly_errors) {
      it.reinit(size);
      ptr_kelly_errors.push_back(&it);
    }

    auto array_view_kelly_errors = dealii::make_array_view(ptr_kelly_errors);
    std::vector<const dealii::ReadVector<Number> *> ptr_kelly_components;
    for (const auto &it : kelly_components_)
      ptr_kelly_components.push_back(&it);

    const auto array_view_kelly_components =
        dealii::make_array_view(ptr_kelly_components);

    dealii::KellyErrorEstimator<dim>::estimate(
        offline_data_->discretization().mapping(),
        offline_data_->dof_handler(),
        offline_data_->discretization().face_quadrature(),
        {},
        array_view_kelly_components,
        array_view_kelly_errors);

    indicators_ = 0.;
    for (const auto &it : kelly_errors)
      indicators_ += it;
#endif
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
      populate_kelly_quantities(state_vector);
      break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }
  }


  template <typename Description, int dim, typename Number>
  void MeshAdaptor<Description, dim, Number>::
      mark_cells_for_coarsening_and_refinement(Triangulation &triangulation
                                               [[maybe_unused]]) const
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
      indicators_.reinit(triangulation.n_active_cells());
      compute_kelly_indicators();
    } break;

    default:
      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /*
     * Mark cells with chosen marking strategy:
     */

    Assert(indicators_.size() == triangulation.n_active_cells(),
           dealii::ExcInternalError());

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
