//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "mpi_ensemble.h"
#include "observer_pointer.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>

#include <optional>

namespace ryujin
{
  /**
   * A postprocessor class for quantities of interest.
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

    using state_type = typename View::state_type;

    using StateVector = typename View::StateVector;

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
     * A tuple describing the values we are interested in: the primitive
     * state and its second moment.
     */
    using value_type =
        std::tuple<state_type /* primitive state */,
                   state_type /* primitive state second moment */>;

    /**
     * Temporal statistics we store for each manifold: the values of the
     * previous and the current time step, and the trapezoidal sum over
     * time.
     */
    struct Statistics {
      std::vector<value_type> old;
      std::vector<value_type> current;
      std::vector<value_type> sum;
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
      std::vector<std::pair<Number, value_type>> time_series;
      std::optional<unsigned int> time_series_cycle;
    };

    //@}
    /**
     * @name Run time options
     */
    //@{

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
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;

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

    void write_mesh_files(unsigned int cycle);

    void clear_statistics();

    value_type internal_accumulate(const StateVector &state_vector,
                                   Manifold &manifold);

    void internal_write_out(const std::string &file_name,
                            const std::string &time_stamp,
                            const std::vector<value_type> &values,
                            const Number scale);

    void internal_write_out_time_series(
        const std::string &file_name,
        const std::vector<std::pair<Number, value_type>> &values,
        bool append);

    //@}
  };

} /* namespace ryujin */
