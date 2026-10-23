//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/interface/postprocessor.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/interface/task.h>

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
   * The class is the task "vtu output" that writes out full vtu records
   * of the solution, or, if constructed with @p levelsets set to true,
   * the task "vtu levelsets" that writes out the cells intersecting the
   * given "manifolds". In both cases the analytic solution can be
   * written out in addition.
   *
   * @ingroup Output
   */
  template <int dim, typename Number = double>
  class VTUOutput final : public dealii::ParameterAcceptor,
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
    VTUOutput(const Interface::Simulation<dim, Number> &simulation,
              bool levelsets = false,
              const std::string &subsection = "/VTUOutput");

    //@}
    /**
     * @name Task
     */
    //@{

    /**
     * Prepare VTU output for the current mesh: allocate temporary
     * storage for additional (dim + 5) scalar vectors of type
     * OfflineData::scalar_type.
     */
    void prepare(TaskContext<Number> &context) override;

    /**
     * Write out the solution (and the analytic solution) whenever the
     * trigger fires.
     */
    void execute(TaskContext<Number> &context) override;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    bool use_mpi_io_;
    bool output_analytic_solution_;

    Number beta_;
    bool recompute_bounds_;

    std::vector<std::string> manifolds_;

    std::vector<std::string> vtu_output_quantities_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const bool levelsets_;
    const Interface::Simulation<dim, Number> &simulation_;
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
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Given a state vector @p state_vector, a file name prefix @p name, the
     * current time @p t, and the current output cycle @p cycle schedule a
     * solution output: the full vector field, or (for the levelsets task)
     * the cells in the vicinity of the predefined levelsets.
     *
     * The function post-processes quantities synchronously, and (depending
     * on configuration options) schedules the write-out asynchronously
     * onto a background worker thread. This implies that @p state_vector
     * can again be modified once schedule_output() returned.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void schedule_output(const StateVector<Number> &state_vector,
                         std::string name,
                         Number t,
                         unsigned int cycle);

    //@}
  };

} // namespace ryujin
