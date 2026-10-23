//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
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

#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The ErrorEvaluation class computes error norms of the numerical
   * solution with respect to an analytic solution.
   *
   * The analytic solution is provided by the InitialValues class, i.e.,
   * the selected initial state configuration is evaluated at the current
   * time. Consequently, error evaluation is only meaningful for initial
   * state configurations that describe an exact (time-dependent) solution
   * of the underlying equations.
   *
   * The quantities for which the error is computed can be freely chosen
   * via a configuration file, as well as the error norms (L1, L2, Linf).
   * The individual errors are then consolidated by summing over all
   * selected quantities (and, if applicable, over all MPI ensembles). The
   * result is one number per selected norm.
   *
   * The class is the task "error evaluation" that writes out the
   * computed error into a log file `<base name>-error.log` whenever its
   * trigger fires, and prints a summary of the error at the final time.
   *
   * @ingroup Output
   */
  template <int dim, typename Number = double>
  class ErrorEvaluation final : public dealii::ParameterAcceptor,
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
    ErrorEvaluation(const Interface::Simulation<dim, Number> &simulation,
                    const std::string &subsection = "/ErrorEvaluation");

    //@}
    /**
     * @name Task
     */
    //@{

    /**
     * Initialize error evaluation: validate the selected quantities and
     * norms.
     *
     * The base name is used for the error log file
     * `<basename>-error.log`. Calling initialize() resets the log file
     * and writes out a header describing its contents.
     */
    void initialize(TaskContext<Number> &context) override;

    /**
     * Append the error at the current time to the error log file whenever
     * the trigger fires.
     */
    void execute(TaskContext<Number> &context) override;

    /**
     * Print a summary of the error at the final time.
     */
    void finalize(TaskContext<Number> &context) override;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    std::vector<std::string> error_quantities_;

    bool error_normalize_;

    std::vector<std::string> error_norms_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const Interface::Simulation<dim, Number> &simulation_;
    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;

    const Interface::Postprocessor<dim, Number> &postprocessor_;

    typename Interface::Postprocessor<dim, Number>::Selection selection_;

    std::string base_name_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Compute the consolidated error norms of the difference between the
     * state vector @p state_vector and the analytic solution @p analytic
     * (interpolated at the same time). The returned vector contains one
     * value per selected norm in the order in which the norms have been
     * selected.
     *
     * @note The function requires MPI communication and is not reentrant.
     * The returned values are only meaningful on world rank 0.
     */
    std::vector<Number> compute(const StateVector<Number> &state_vector,
                                const StateVector<Number> &analytic) const;

    std::string description() const;

    //@}
  };

} // namespace ryujin
