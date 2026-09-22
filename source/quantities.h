//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "mpi_ensemble.h"
#include "observer_pointer.h"
#include "offline_data.h"
#include "selected_components_extractor.h"

#include <deal.II/base/parameter_acceptor.h>

#include <optional>

namespace ryujin
{
  /**
   * A postprocessor class for quantities of interest.
   *
   * The class accumulates statistics of a user selected list of
   * (conserved, primitive, precomputed, initial, or parabolic) quantities
   * on level set defined interior and boundary manifolds. For every
   * degree of freedom of a manifold the raw moments of all selected
   * quantities are stored and averaged in time (with a trapezoidal rule),
   * and averaged in space (weighted by the lumped, or boundary mass).
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class Quantities final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using StateVector = typename View::StateVector;
    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    /**
     * The number of raw moments we store for every selected quantity.
     */
    static constexpr unsigned int n_moments = 2;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    Quantities(const MPIEnsemble &mpi_ensemble,
               const OfflineData<dim, Number> &offline_data,
               const HyperbolicSystem &hyperbolic_system,
               const ParabolicSystem &parabolic_system,
               const InitialPrecomputedVector &initial_precomputed,
               const std::string &subsection = "/Quantities");

    /**
     * Prepare evaluation. A call to @ref prepare() allocates temporary
     * storage and is necessary before accumulate() and write_out() can be
     * called.
     *
     * Calling prepare() allocates temporary storage for various boundary
     * and interior maps. The storage requirement varies according to the
     * supplied manifold descriptions.
     *
     * The string parameter @p name is used as base name for output files.
     */
    void prepare(const std::string &name);

    //@}
    /**
     * @name Functions for accumulating and writing out quantities
     */
    //@{

    /**
     * Takes a state vector @p U at time t (obtained at the end of a full
     * Strang step) and accumulates statistics for quantities of interests
     * for all defined manifolds.
     */
    void accumulate(const StateVector &state_vector, const Number t);

    /**
     * Write quantities of interest to designated output files.
     */
    void write_out(const StateVector &state_vector,
                   const Number t,
                   unsigned int cycle);

  private:
    //@}
    /**
     * @name Internal typedefs
     */
    //@{

    /**
     * A tuple describing (local) dof index, normal, normal mass, mass,
     * boundary id, and position of a degree of freedom belonging to a
     * manifold. We use the same description for interior and boundary
     * manifolds: For an interior degree of freedom the normal and normal
     * mass are zero, the mass is the lumped mass matrix entry, and the
     * boundary id is set to dealii::numbers::internal_face_boundary_id.
     */
    using ManifoldPoint =
        typename OfflineData<dim, Number>::BoundaryDescription;

    /**
     * Temporal statistics we store for each manifold: the values of the
     * previous and the current time step, and the trapezoidal sum over
     * time.
     *
     * All values are stored in a flat array with stride() entries per
     * point: for every point the raw moments are stored consecutively,
     * with all selected quantities of the first moment first, followed
     * by all selected quantities of the second moment, and so on.
     */
    struct Statistics {
      std::vector<Number> old;
      std::vector<Number> current;
      std::vector<Number> sum;
      Number t_old;
      Number t_new;
      Number t_sum;
    };

    /**
     * All data associated with a single interior or boundary manifold.
     */
    struct Manifold {
      std::string name;
      bool boundary;
      bool instantaneous;
      bool time_averaged;
      bool space_averaged;
      std::vector<ManifoldPoint> points;
      Statistics statistics;
      std::vector<std::pair<Number, std::vector<Number>>> time_series;
      std::optional<unsigned int> time_series_cycle;
    };

    //@}
    /**
     * @name Run time options
     */
    //@{

    std::vector<std::string> quantities_;

    std::vector<std::tuple<std::string, std::string, std::string>>
        interior_manifolds_;

    std::vector<std::tuple<std::string, std::string, std::string>>
        boundary_manifolds_;

    bool clear_temporal_statistics_on_writeout_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;

    SelectedComponentsExtractor<Description, dim, Number> extractor_;

    /**
     * All interior and boundary manifolds with associated point maps and
     * statistics.
     */
    std::vector<Manifold> manifolds_;

    std::string base_name_;
    std::string header_;
    bool mesh_files_have_been_written_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * The number of values stored per point: the number of moments times
     * the number of selected quantities.
     */
    unsigned int stride() const;

    void write_mesh_files(unsigned int cycle);

    void clear_statistics();

    /**
     * Ensure that the state vector is resident on the host and prepare
     * the extractor for reading from it.
     */
    void prepare_extraction(const StateVector &state_vector);

    /**
     * Read the current values of all points of the manifold into the
     * current statistics and return the mass weighted spatial average.
     * The extraction has to be prepared with prepare_extraction().
     */
    std::vector<Number> internal_accumulate(Manifold &manifold);

    void internal_write_out(const std::string &file_name,
                            const std::string &time_stamp,
                            const std::vector<Number> &values,
                            const Number scale);

    void internal_write_out_time_series(
        const std::string &file_name,
        const std::vector<std::pair<Number, std::vector<Number>>> &values,
        bool append);

    //@}
  };

} /* namespace ryujin */
