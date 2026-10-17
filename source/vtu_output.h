//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/interface/postprocessor.h>

#include "mpi_ensemble.h"
#include "observer_pointer.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

namespace ryujin
{

  /**
   * The VTUOutput class implements output of a selection of components of
   * the state vector and of postprocessed quantities.
   *
   * The postprocessed quantities, "schlieren_<name>" and
   * "vorticity_<name>" (see Postprocessor), are scaled and
   * normalized for visualization as follows:
   * \f[
   *   \text{quantity}[i] = 1 - \exp\left(-\beta \frac{ |q_i| - \min_k
   *   |q_k|}{\max_k |q_k| - \min_k |q_k|}\right).
   * \f]
   *
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class VTUOutput final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    VTUOutput(const MPIEnsemble &mpi_ensemble,
              const OfflineData<dim, Number> &offline_data,
              const Interface::HyperbolicSystem &hyperbolic_system,
              const Interface::Postprocessor<dim, Number> &postprocessor,
              const std::string &subsection = "/VTUOutput");

    /**
     * Prepare VTU output. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     *
     * Calling prepare() allocates temporary storage for additional (dim +
     * 5) scalar vectors of type OfflineData::scalar_type.
     */
    void prepare();

    //@}
    /**
     * @name Functions for scheduling output
     */
    //@{

    /**
     * Given a state vector @p state_vector, a file name prefix @p name, the
     * current time @p t, and the current output cycle @p cycle schedule a
     * solution output.
     *
     * The function post-processes quantities synchronously, and (depending
     * on configuration options) schedules the write-out asynchronously
     * onto a background worker thread. This implies that @p state_vector
     * can again be modified once schedule_output() returned.
     *
     * The boolean @p output_full controls whether the full vector field
     * is written out. Correspondingly, @p output_cutplanes controls
     * whether cells in the vicinity of predefined cutplanes are written
     * out.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void schedule_output(const StateVector<Number> &state_vector,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

    /**
     * Reset computed normalization bounds. Calling this function will
     * force a recomputation of the normalization bounds during the next
     * call to schedule_output().
     */
    void reset_bounds()
    {
      bounds_.clear();
    }

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    bool use_mpi_io_;

    Number beta_;
    bool recompute_bounds_;

    std::vector<std::string> manifolds_;

    std::vector<std::string> vtu_output_quantities_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;

    const Interface::Postprocessor<dim, Number> &postprocessor_;

    typename Interface::Postprocessor<dim, Number>::Selection selection_;

    /**
     * The normalization bounds (maximum, minimum) of every selected
     * component; only used for postprocessed quantities.
     */
    std::vector<std::pair<Number, Number>> bounds_;

    //@}
  };

} /* namespace ryujin */
