//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/driver/simulation.h>
#include <ryujin/driver/task_composition.h>
#include <ryujin/typed/equation.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <iostream>
#include <string>

namespace ryujin
{
  template <typename HD,
            typename PD,
            typename Number,
            template <int, typename>
            class TaskComposition>
  void Equation<HD, PD, Number, TaskComposition>::run(
      int dim,
      const std::string &parameter_file,
      const MPI_Comm &mpi_comm) const
  {
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
      std::cout << "[INFO] dispatching to driver »" << this->name()
                << "« with dim=" << dim << std::endl;
    }

    switch (dim) {
    case 1:
      run_simulation<1>(parameter_file, mpi_comm);
      break;
    case 2:
      run_simulation<2>(parameter_file, mpi_comm);
      break;
    case 3:
      run_simulation<3>(parameter_file, mpi_comm);
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage("The dimension needs to be either 1, 2, "
                                     "or 3, but we encountered »" +
                                     std::to_string(dim) + "«"));
    }
  }


  template <typename HD,
            typename PD,
            typename Number,
            template <int, typename>
            class TaskComposition>
  void Equation<HD, PD, Number, TaskComposition>::create_parameter_files() const
  {
    create_parameter_file<1>(false);
    create_parameter_file<2>(true);
    create_parameter_file<3>(false);
  }


  template <typename HD,
            typename PD,
            typename Number,
            template <int, typename>
            class TaskComposition>
  template <int dim>
  void Equation<HD, PD, Number, TaskComposition>::run_simulation(
      const std::string &parameter_file, const MPI_Comm &mpi_comm)
  {
    Simulation<HD, PD, dim, Number> simulation(mpi_comm);
    TaskComposition<dim, Number> task_composition(simulation);
    dealii::ParameterAcceptor::initialize(parameter_file);
    task_composition.run();
  }


  template <typename HD,
            typename PD,
            typename Number,
            template <int, typename>
            class TaskComposition>
  template <int dim>
  void Equation<HD, PD, Number, TaskComposition>::create_parameter_file(
      bool write_detailed_description) const
  {
    {
      auto &prm = dealii::ParameterAcceptor::prm;

      /*
       * Workaround: The Scheduler is constructed last and declares the
       * "A - Scheduler" section last. Add an entry to the section so that it
       * shows up first.
       */
      prm.enter_subsection("A - Scheduler");
      prm.declare_entry("basename", "test");
      prm.leave_subsection();

      /*
       * The parameters "dimension" and "equation" of the EquationDispatch,
       * set to this equation:
       */
      prm.enter_subsection("B - Equation");
      prm.declare_entry("dimension",
                        std::to_string(dim),
                        dealii::Patterns::Integer(),
                        "The spatial dimension");
      prm.declare_entry("equation",
                        this->name(),
                        dealii::Patterns::Anything(),
                        "The PDE system");
      prm.leave_subsection();

      /*
       * Create temporary objects for the sole purpose of populating the
       * ParameterAcceptor::prm object.
       */
      Simulation<HD, PD, dim, Number> simulation(MPI_COMM_SELF);
      TaskComposition<dim, Number> task_composition(simulation);

      std::string base_name = this->name();
      std::ranges::replace(base_name, ' ', '_');
      base_name += "-" + std::to_string(dim) + "d";

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_SELF) == 0) {
        const auto full_name =
            "default_parameters-" + base_name + "-description.prm";
        if (write_detailed_description)
          prm.print_parameters(
              full_name,
              dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder);

        const auto short_name = "default_parameters-" + base_name + ".prm";
        prm.print_parameters(
            short_name,
            dealii::ParameterHandler::OutputStyle::Short |
                dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder);
      }
      // all objects have to go out of scope, see
      // https://github.com/dealii/dealii/issues/15111
    }

    dealii::ParameterAcceptor::clear();
  }
} // namespace ryujin
