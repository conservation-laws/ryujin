//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "offline_data.h"
#include "postprocessor.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

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

    using state_type = typename View::state_type;

    static constexpr auto n_precomputed_values = View::n_precomputed_values;

    using precomputed_type = typename View::precomputed_type;

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
    VTUOutput(const MPI_Comm &mpi_communicator,
              const OfflineData<dim, Number> &offline_data,
              const HyperbolicSystem &hyperbolic_system,
              const ParabolicSystem &parabolic_system,
              const Postprocessor<Description, dim, Number> &postprocessor,
              const InitialPrecomputedVector &initial_precomputed,
              const ScalarVector &alpha,
              const std::string &subsection = "/VTUOutput");

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
    void schedule_output(const StateVector &state_vector,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

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
    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const ParabolicSystem> parabolic_system_;
    dealii::SmartPointer<const Postprocessor<Description, dim, Number>>
        postprocessor_;

    const InitialPrecomputedVector &initial_precomputed_;
    const ScalarVector &alpha_;
    //@}
  };

} /* namespace ryujin */
