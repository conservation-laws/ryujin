//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/postprocessor.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/interface/task.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <random>

namespace ryujin
{
  /**
   * Controls the spatial mesh adaptation strategy.
   *
   * @ingroup MeshAndOfflineData
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
     * Perform local refinement and coarsening based on smoothness indicators.
     */
    smoothness_indicators,
  };

  /**
   * Controls the marking strategy used for mesh adaptation. This
   * configuration option is ignored for the uniform global refinement
   * strategy.
   *
   * @ingroup MeshAndOfflineData
   */
  enum class MarkingStrategy {
    /**
     * Refine and coarsen according to a fixed tolerance normalized
     * according to the difference between the maximal and minimal attained
     * values for the chosen refinement indicators.
     */
    fixed_threshold,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(
    ryujin::AdaptationStrategy,
    LIST({ryujin::AdaptationStrategy::global_refinement, "global refinement"},
         {ryujin::AdaptationStrategy::random_adaptation, "random adaptation"},
         {ryujin::AdaptationStrategy::smoothness_indicators,
          "smoothness indicators"}, ));

DECLARE_ENUM(ryujin::MarkingStrategy,
             LIST({ryujin::MarkingStrategy::fixed_threshold,
                   "fixed threshold"}));
#endif

namespace ryujin
{
  /**
   * The MeshAdaptor class is responsible for performing global or local
   * mesh adaptation.
   *
   * The class is the task "mesh adaptation" that, whenever its trigger
   * fires (by default in every 10th cycle), analyzes the state vector,
   * marks cells for coarsening and refinement, and adapts the mesh. The
   * state vector of the Simulation is attached to the triangulation
   * and transferred to the new mesh when the Scheduler prepares all tasks
   * again.
   *
   * @ingroup MeshAndOfflineData
   */
  template <int dim, typename Number = double>
  class MeshAdaptor final : public dealii::ParameterAcceptor,
                            public Interface::Task<Number>
  {
  public:
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    MeshAdaptor(Interface::Simulation<dim, Number> &simulation,
                const std::string &subsection = "/MeshAdaptor");

    //@}
    /**
     * @name Task
     */
    //@{

    /**
     * Prepare temporary storage and clean up internal data for the
     * analyze() facility.
     */
    void prepare(TaskContext<Number> &context) override;

    /**
     * Adapt the mesh whenever the trigger fires.
     */
    void execute(TaskContext<Number> &context) override;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    AdaptationStrategy adaptation_strategy_;
    std::uint_fast64_t random_adaptation_mersenne_twister_seed_;

    MarkingStrategy marking_strategy_;
    double coarsening_threshold_;
    double refinement_threshold_;
    bool absolute_threshold_;
    unsigned int min_refinement_level_;
    unsigned int max_refinement_level_;

    std::vector<std::string> smoothness_selected_quantities_;
    Number smoothness_local_global_ratio_;
    Number smoothness_min_cutoff_;
    Number smoothness_max_cutoff_;
    unsigned int smoothness_widen_stencil_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    Trigger<Number> trigger_;

    Interface::Simulation<dim, Number> &simulation_;
    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;

    const Interface::Postprocessor<dim, Number> &postprocessor_;

    typename Interface::Postprocessor<dim, Number>::Selection selection_;

    /* The computed cell indicators. */
    dealii::Vector<float> indicators_;

    /* random adaptation: */
    std::mt19937_64 mersenne_twister_;

    /*
     * smoothness indicators: only valid if the "smoothness indicator"
     * refinement strategy has been selected.
     */
    ScalarVector<Number> smoothness_indicators_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Analyze the given StateVector with the configured adaptation
     * strategy.
     */
    void analyze(const StateVector<Number> &state_vector);

    /**
     * Mark the cells of the triangulation for coarsening and refinement
     * with the configured mesh adaptation and marking strategies.
     */
    void mark_cells_for_coarsening_and_refinement();

    /* random adaptation: */
    void populate_cell_indicators_with_random_values();

    /* smoothness indicators: */
    void populate_cell_indicators_from_smoothness_indicators();

    /*
     * Compute smoothness indicators. This function reinitializes and
     * populates the smoothness_indicators_ vector.
     */
    void compute_smoothness_indicators(const StateVector<Number> &state_vector);

    //@}
  };

} // namespace ryujin
