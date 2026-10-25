//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/equation.h>
#include <ryujin/driver/parameter_editor.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace ryujin
{
  namespace
  {
    /* The parameters "dimension" and "equation", see source/main.cc: */
    std::string &get(ParameterValues &values, const std::string &name)
    {
      return values[{{"B - Equation"}, name}];
    }


    /*
     * Read the values of @p parameter_file, if it exists. Choose the
     * first equation in 2D if there is no valid choice:
     */
    ParameterValues read(const std::string &parameter_file)
    {
      ParameterValues values;
      if (std::filesystem::exists(parameter_file)) {
        std::stringstream stream;
        stream << std::ifstream(parameter_file).rdbuf();
        values = read_parameter_values(stream.str());
      }

      if (!equations().contains(get(values, "equation")))
        get(values, "equation") = equations().begin()->first;
      const std::string dimension = get(values, "dimension");
      if (dimension != "1" && dimension != "2" && dimension != "3")
        get(values, "dimension") = "2";

      return values;
    }
  } // namespace


  void edit_parameter_file(const std::string &parameter_file)
  {
    EditorSession session;
    session.parameter_file = parameter_file;
    ParameterValues values = read(parameter_file);

    /*
     * Show invalid values read from the parameter file, but replace
     * invalid values carried over from a different equation or
     * dimension by the default:
     */
    bool keep_invalid = true;

    for (;;) {
      EditorOutcome outcome = EditorOutcome::quit;
      const auto &equation = equations().at(get(values, "equation"));
      equation.with_parameters(std::stoi(get(values, "dimension")), [&] {
        ParameterModel model(dealii::ParameterAcceptor::prm);
        model.assign(values, keep_invalid);

        outcome = run_parameter_editor(model, session);

        /* Keep the values of parameters the equation does not have: */
        auto new_values = model.values();
        new_values.merge(values);
        values = std::move(new_values);
      });

      if (outcome == EditorOutcome::quit)
        return;

      keep_invalid = outcome == EditorOutcome::load;
      if (outcome == EditorOutcome::load)
        values = read(session.parameter_file);
    }
  }


  bool parameter_editor_available(const MPI_Comm &mpi_communicator)
  {
#if defined(WITH_FTXUI)
    return dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) == 1 &&
           ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO);
#else
    (void)mpi_communicator;
    return false;
#endif
  }


#if !defined(WITH_FTXUI)
  EditorOutcome run_parameter_editor(ParameterModel & /*model*/,
                                     EditorSession & /*session*/)
  {
    AssertThrow(false,
                dealii::ExcMessage("The parameter editor requires FTXUI."));
    __builtin_unreachable();
  }
#endif
} // namespace ryujin
