//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "gpu.h"
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
   * degree of freedom of a manifold the raw moments (up to a selectable
   * order) of all selected quantities are stored and averaged in time
   * (with a trapezoidal rule), and averaged in space (weighted by the
   * lumped, or boundary mass). Raw moments are linear in the samples and
   * can thus be accumulated exactly. They are converted to the mean and
   * the central moments (variance, third and fourth central moment) on
   * output.
   *
   * All statistics are accumulated on the selected memory space (i.e., on
   * the device if the state vector resides there) and are only
   * transferred to the host for writing out.
   *
   * @note The conversion from raw to central moments is subject to
   * cancellation if the fluctuations of a quantity are small compared to
   * its mean.
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
     * All data associated with a single interior or boundary manifold.
     *
     * The raw moments are stored in flat, mirrored arrays with stride()
     * entries per point: for every point the raw moments are stored
     * consecutively, with all selected quantities of the first moment
     * first, followed by all selected quantities of the second moment,
     * and so on.
     */
    struct Manifold {
      std::string name;
      bool boundary;
      bool instantaneous;
      bool time_averaged;
      bool space_averaged;

      /**
       * The point map. It is kept on the host for writing out mesh files.
       */
      std::vector<ManifoldPoint> points;

      /**
       * The local dof index and mass of every point, and the total mass
       * summed over all MPI ranks.
       */
      Mirrored<unsigned int *> indices{"quantities_indices"};
      Mirrored<Number *> masses{"quantities_masses"};
      Number mass_sum;

      /**
       * Temporal statistics: the raw moments of the previous and the
       * current time step, and the trapezoidal sum over time.
       */
      Mirrored<Number *> old{"quantities_old"};
      Mirrored<Number *> current{"quantities_current"};
      Mirrored<Number *> sum{"quantities_sum"};
      Number t_old;
      Number t_new;
      Number t_sum;

      /**
       * The time series of mass weighted spatial averages: the mean and
       * central moments of all selected quantities per time step.
       */
      std::vector<std::pair<Number, std::vector<Number>>> time_series;
      std::optional<unsigned int> time_series_cycle;
    };

    //@}
    /**
     * @name Run time options
     */
    //@{

    std::vector<std::string> quantities_;

    unsigned int n_moments_;

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

    /**
     * Return a tab separated list of column names: the plain names of
     * all selected quantities for instantaneous values, or the names of
     * the mean and all central moments for averaged values.
     */
    std::string header(bool averaged) const;

    void write_mesh_files(unsigned int cycle);

    void clear_statistics();

    /**
     * Read the current values of all points of the manifold into the
     * current statistics and return the mass weighted spatial average of
     * the raw moments. The extraction has to be prepared on the selected
     * memory space beforehand.
     */
    std::vector<Number> internal_accumulate(Manifold &manifold);

    /**
     * Write out instantaneous values, or (if @p averaged is set) the mean
     * and central moments computed from the raw moments scaled by
     * @p scale.
     */
    void internal_write_out(const std::string &file_name,
                            const std::string &time_stamp,
                            const Mirrored<Number *> &values,
                            const Number scale,
                            bool averaged);

    void internal_write_out_time_series(
        const std::string &file_name,
        const std::vector<std::pair<Number, std::vector<Number>>> &values,
        bool append);

    //@}
  };

} /* namespace ryujin */
