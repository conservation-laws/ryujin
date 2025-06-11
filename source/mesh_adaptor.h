//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "mpi_ensemble.h"
#include "observer_pointer.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>

#include <random>

namespace ryujin
{
  /**
   * Controls the spatial mesh adaptation strategy.
   *
   * @ingroup Mesh
   */
  enum class AdaptationStrategy {
    /**
     * Perform a uniform global refinement.
     */
    global_refinement,

    /**
     * Perform random refinement and coarsening with a deterministic
     * Mersenne Twister and a chosen seed. This refinement strategy is only
     * useful for debugging and testing.
     */
    random_adaptation,

    /**
     * Perform local refinement and coarsening based on Kelly error estimator.
     */
    kelly_estimator,

    /**
     * Perform local refinement and coarsening based on a smoothness estimator.
     */
    smoothness_estimator,
  };

  /**
   * Controls the marking strategy used for mesh adaptation. This
   * configuration option is ignored for the uniform global refinement
   * strategy.
   *
   * @ingroup Mesh
   */
  enum class MarkingStrategy {
    /**
     * Refine and coarsen a configurable selected percentage of cells.
     */
    fixed_number,
    /**
     * Refine and coarsen such that the criteria of cells getting flagged for
     * refinement make up for a certain fraction of the total "error".
     */
    fixed_fraction,
    /**
     * Refine and coarsen according to a fixed tolerance normalized according
     * to the difference between the max and min attained refinement indicator
     */
    fixed_tolerance,
  };

  /**
   * Controls the time point selection strategy.
   *
   * @ingroup Mesh
   */
  enum class TimePointSelectionStrategy {
    /**
     * Perform a mesh adaptation cycle at preselected fixed time points.
     */
    fixed_time_points,

    /**
     * Perform a mesh adaptation cycle at every nth simulation cycle.
     */
    simulation_cycle,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(
    ryujin::AdaptationStrategy,
    LIST({ryujin::AdaptationStrategy::global_refinement, "global refinement"},
         {ryujin::AdaptationStrategy::random_adaptation, "random adaptation"},
         {ryujin::AdaptationStrategy::kelly_estimator, "kelly estimator"},
         {ryujin::AdaptationStrategy::smoothness_estimator,
          "smoothness estimator"}, ));

DECLARE_ENUM(ryujin::MarkingStrategy,
             LIST({ryujin::MarkingStrategy::fixed_number, "fixed number"},
                  {ryujin::MarkingStrategy::fixed_fraction, "fixed fraction"},
                  {ryujin::MarkingStrategy::fixed_tolerance,
                   "fixed tolerance"}));

DECLARE_ENUM(ryujin::TimePointSelectionStrategy,
             LIST({ryujin::TimePointSelectionStrategy::fixed_time_points,
                   "fixed time points"},
                  {ryujin::TimePointSelectionStrategy::simulation_cycle,
                   "simulation cycle"}, ));
#endif

namespace ryujin
{
  /**
   * The MeshAdaptor class is responsible for performing global or local
   * mesh adaptation.
   *
   * @ingroup Mesh
   */
  template <typename Description, int dim, typename Number = double>
  class MeshAdaptor final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using StateVector = typename View::StateVector;
    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;
    using ScalarVector = Vectors::ScalarVector<Number>;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    MeshAdaptor(const MPIEnsemble &mpi_ensemble,
                const OfflineData<dim, Number> &offline_data,
                const HyperbolicSystem &hyperbolic_system,
                const ParabolicSystem &parabolic_system,
                const InitialPrecomputedVector &initial_precomputed,
                const ScalarVector &alpha,
                const std::string &subsection = "/MeshAdaptor");

    /**
     * Prepare temporary storage and clean up internal data for the
     * analyze() facility.
     */
    void prepare(const Number t);

    /**
     * Analyze the given StateVector with the configured adaptation
     * strategy and time point selection strategy and decide whether a mesh
     * adaptation cycle should be performed.
     */
    void analyze(const StateVector &state_vector,
                 const Number t,
                 unsigned int cycle);

    /**
     * A boolean indicating whether we should perform a mesh adapation step
     * in the current cycle. The analyze() method will set this boolean to
     * true whenever the selected adaptation strategy advices to perform an
     * adaptation cycle.
     */
    ACCESSOR_READ_ONLY(need_mesh_adaptation)

    /**
     * Mark cells for coarsening and refinement with the configured mesh
     * adaptation and marking strategies.
     */
    void mark_cells_for_coarsening_and_refinement(
        dealii::Triangulation<dim> &triangulation) const;

    /**
     * The computed cell indicators.
     */
    ACCESSOR_READ_ONLY(indicators);

    /**
     * The smoothness indicators. The vector is only valid if the
     * "smoothness indicator" refinement strategy has been selected.
     */
    ACCESSOR_READ_ONLY(smoothness_indicators);

  private:
    /**
     * @name Run time options
     */
    //@{

    AdaptationStrategy adaptation_strategy_;
    std::uint_fast64_t random_adaptation_mersenne_twister_seed_;

    MarkingStrategy marking_strategy_;
    double refinement_fraction_;
    double coarsening_fraction_;
    double refinement_tolerance_;
    double coarsening_tolerance_;
    unsigned int min_refinement_level_;
    unsigned int max_refinement_level_;
    unsigned int max_num_cells_;

    TimePointSelectionStrategy time_point_selection_strategy_;
    std::vector<Number> adaptation_time_points_;
    unsigned int adaptation_cycle_interval_;

    double smoothness_local_global_ratio_;
    unsigned int smoothness_widen_stencil_;
    double smoothness_lower_threshold_;
    double smoothness_upper_threshold_;

    std::vector<std::string> kelly_quantities_;

    //@}
    /**
     * @name Internal fields and methods
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;

    bool need_mesh_adaptation_;

    mutable dealii::Vector<float> indicators_;

    mutable std::vector<dealii::Vector<float>> indicators_vec_;

    /* random adaptation: */

    void compute_random_indicators() const;

    mutable std::mt19937_64 mersenne_twister_;

    /* Kelly and smoothness estimator: */

    void populate_selected_quantities(const StateVector &state_vector) const;
    void compute_kelly_indicators() const;
    void compute_smoothness_indicators() const;

    const InitialPrecomputedVector &initial_precomputed_;
    const ScalarVector &alpha_;

    mutable std::vector<ScalarVector> quantities_;
    mutable ScalarVector smoothness_indicators_;
    //@}
  };

} // namespace ryujin
