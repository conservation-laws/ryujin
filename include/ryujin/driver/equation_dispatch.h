//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/equation_library.h>
#include <ryujin/interface/equation.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <memory>
#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The Dave error message
   */
  inline const std::string dave =
      "\nDave, this conversation can serve no purpose anymore. Goodbye.\n\n";

  /**
   * Dispatcher class that runs the equation selected in the parameter
   * file.
   *
   * When starting up the ryujin executable we are faced with the following
   * difficulties:
   *  - The Simulation class is templated in the HyperbolicDescription, the
   *    ParabolicDescription, and the dimension that have to be read from the
   *    `ryujin.prm` parameter file.
   *  - The final set of valid parameters that can be configured in the
   *    `ryujin.prm` depend on the runtime parameters "dimension" and
   *    "equation" themselves.
   *
   * We thus first read in two parameters from the parameter file:
   * ```
   * subsection B - Equation
   *   set dimension           = ...
   *   set equation            = ...
   * end
   * ```
   * and then run the equation of the EquationLibrary with the given name
   * for the given dimension.
   *
   * @ingroup TimeLoop
   */
  class EquationDispatch final : dealii::ParameterAcceptor
  {
  public:
    EquationDispatch()
        : ParameterAcceptor("B - Equation")
        , equations_(EquationLibrary::create())
    {
      AssertThrow(!equations_.empty(),
                  dealii::ExcMessage(
                      dave + "No equation has been registered. Consequently, "
                             "there is nothing for us to do.\n"));

      dimension_ = 0;
      add_parameter("dimension", dimension_, "The spatial dimension");

      equation_ = equations_.front()->name();
      add_parameter("equation",
                    equation_,
                    "The PDE system",
                    this->prm,
                    EquationLibrary::selection(equations_));
    }


    /**
     * Write the default parameter files of all registered equations.
     */
    static void create_parameter_files()
    {
      const auto equations = EquationLibrary::create();

      AssertThrow(!equations.empty(),
                  dealii::ExcMessage(
                      dave + "No equation has been registered. Consequently, "
                             "there is nothing for us to do.\n"));

      for (const auto &equation : equations)
        equation->create_parameter_files();
    }


    /**
     * Read the parameters "dimension" and "equation" from @p
     * parameter_file and run the selected equation.
     */
    void dispatch(const std::string &parameter_file, const MPI_Comm &mpi_comm)
    {
      ParameterAcceptor::prm.parse_input(parameter_file,
                                         "",
                                         /* skip undefined */ true,
                                         /* assert entries present */ false);

      AssertThrow(dimension_ >= 1 && dimension_ <= 3,
                  dealii::ExcMessage(dave +
                                     "The dimension parameter needs to be "
                                     "either 1, 2, or 3, but we encountered »" +
                                     std::to_string(dimension_) + "«\n"));

      const auto &equation = EquationLibrary::select(equations_, equation_);
      equation.run(dimension_, parameter_file, mpi_comm);
    }

  private:
    /**
     * @name Run time options
     */
    //@{

    int dimension_;
    std::string equation_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const std::vector<std::unique_ptr<Interface::Equation>> equations_;

    //@}
  };
} // namespace ryujin
