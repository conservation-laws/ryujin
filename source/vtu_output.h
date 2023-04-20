//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "postprocessor.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <future>

namespace ryujin
{

  /**
   * the VTUOutput class implements output of the conserved state vector
   * and a number of postprocessed quantities computed by the Postprocessor
   * class.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class VTUOutput final : public dealii::ParameterAcceptor
  {
    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

  public:
    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension =
        HyperbolicSystem::template problem_dimension<dim>;

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type =
        typename HyperbolicSystem::template state_type<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::n_precomputed_values
     */
    static constexpr unsigned int n_precomputed_values =
        HyperbolicSystem::template n_precomputed_values<dim>;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * Typedef for a MultiComponentVector storing the state U.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * Typedef for a MultiComponentVector storing precomputed values.
     */
    using precomputed_type = MultiComponentVector<Number, n_precomputed_values>;

    /**
     * Constructor.
     */
    VTUOutput(
        const MPI_Comm &mpi_communicator,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicModule<Description, dim, Number> &hyperbolic_module,
        const Postprocessor<Description, dim, Number> &postprocessor,
        const std::string &subsection = "VTUOutput");

    /**
     * Prepare VTU output. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     *
     * Calling prepare() allocates temporary storage for additional (dim +
     * 5) scalar vectors of type OfflineData::scalar_type.
     */
    void prepare();

    /**
     * Given a state vector @p U and a file name prefix @p name, the
     * current time @p t, and the current output cycle @p cycle) schedule a
     * solution output.
     *
     * The function post-processes quantities synchronously, and (depending
     * on configuration options) schedules the write-out asynchronously
     * onto a background worker thread. This implies that @p U can again be
     * modified once schedule_output() returned.
     *
     * The booleans @p output_full controls whether the full vector field
     * is written out. Correspondingly, @p output_cutplanes controls
     * whether cells in the vicinity of predefined cutplanes are written
     * out.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void schedule_output(const vector_type &U,
                         const precomputed_type &precomputed_values,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

    /**
     * Return a boolean indicating whether Steps 0 to 2 of an explicit
     * Euler step have to be prepared prior to scheduling output. This is
     * the case if a precomputed value or alpha is used in the output
     * process.
     */
    ACCESSOR_READ_ONLY(need_to_prepare_step)

  private:
    /**
     * @name Run time options
     */
    //@{

    bool use_mpi_io_;

    std::vector<std::string> manifolds_;

    std::vector<std::string> vtu_output_quantities_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicModule<Description, dim, Number>>
        hyperbolic_module_;
    dealii::SmartPointer<const Postprocessor<Description, dim, Number>>
        postprocessor_;

    bool need_to_prepare_step_;

    std::vector<scalar_type> quantities_;

    std::vector<std::tuple<std::string /*name*/,
                           std::function<void(scalar_type & /*result*/,
                                              const vector_type & /*U*/,
                                              const precomputed_type &)>>>
        quantities_mapping_;

    //@}
  };

} /* namespace ryujin */
