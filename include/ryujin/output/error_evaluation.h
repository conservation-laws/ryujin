//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/interface/offline_data.h>
#include <ryujin/interface/postprocessor.h>

#include <deal.II/base/parameter_acceptor.h>

#include <ostream>
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
   * The class also writes out the computed error into a log file `<base
   * name>-error.log`.
   *
   * @ingroup Output
   */
  template <int dim, typename Number = double>
  class ErrorEvaluation final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    ErrorEvaluation(const MPIEnsemble &mpi_ensemble,
                    const Interface::OfflineData<dim, Number> &offline_data,
                    const Interface::HyperbolicSystem &hyperbolic_system,
                    const Interface::Postprocessor<dim, Number> &postprocessor,
                    const std::string &subsection = "/ErrorEvaluation");

    /**
     * Prepare error evaluation. A call to @ref prepare() validates the
     * selected quantities and norms and is necessary before compute(),
     * write_out(), or print_summary() can be called.
     *
     * The string parameter @p name is used as base name for the error log
     * file `<name>-error.log`. Calling prepare() resets the log file and
     * writes out a header describing its contents.
     */
    void prepare(const std::string &name);

    //@}
    /**
     * @name Functions for computing and reporting errors
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

    /**
     * Compute the consolidated error norms at time @p t and append a line
     * to the error log file.
     */
    void write_out(const StateVector<Number> &state_vector,
                   const StateVector<Number> &analytic,
                   Number t) const;

    /**
     * Print a human readable summary of the error norms @p norms (as
     * returned by compute()) at time @p t and for a discretization with
     * @p n_global_dofs degrees of freedom to @p stream.
     */
    void print_summary(std::ostream &stream,
                       Number t,
                       dealii::types::global_dof_index n_global_dofs,
                       const std::vector<Number> &norms) const;

  private:
    //@}
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

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const Interface::OfflineData<dim, Number>>
        offline_data_;

    const Interface::Postprocessor<dim, Number> &postprocessor_;

    typename Interface::Postprocessor<dim, Number>::Selection selection_;

    std::string base_name_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    std::string description() const;

    //@}
  };

} // namespace ryujin
