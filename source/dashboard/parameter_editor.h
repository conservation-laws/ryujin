//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/driver/parameter_model.h>

#include <deal.II/base/mpi.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ryujin
{
  /**
   * How the parameter editor has been left.
   *
   * @ingroup Driver
   */
  enum class EditorOutcome : std::uint8_t {
    /**
     * Quit the editor.
     */
    quit,

    /**
     * Load the parameter file EditorSession::parameter_file: open the
     * editor again with its values.
     */
    load,

    /**
     * The equation or the dimension has been changed: open the editor
     * again with the parameters of the new equation.
     */
    rebuild,
  };


  /**
   * The state of the parameter editor that is kept when the editor is
   * opened again for a different equation or dimension.
   *
   * @ingroup Driver
   */
  struct EditorSession {
    /**
     * The parameter file to write, changed by "load" and "save as".
     */
    std::string parameter_file;

    /**
     * The path of the selected section.
     */
    std::vector<std::string> section;

    /**
     * Whether there are unsaved changes from an earlier editor.
     */
    bool modified = false;

    /**
     * Whether to show all sections, also the ones not selected by a
     * selection such as "geometry" or "equation of state".
     */
    bool show_all = false;
  };


  /**
   * Return whether the parameter editor can be shown: ryujin is compiled
   * with FTXUI, runs on a single rank of @p mpi_communicator, and the
   * input and the output are an interactive terminal.
   *
   * @ingroup Driver
   */
  bool parameter_editor_available(const MPI_Comm &mpi_communicator);


  /**
   * Open the parameter editor for @p parameter_file. If the file does
   * not exist, start from the defaults of the first equation in 2D.
   *
   * @ingroup Driver
   */
  void edit_parameter_file(const std::string &parameter_file);


  /**
   * Show the editor for all parameters of @p model, and change the
   * values of the model. Return EditorOutcome::rebuild as soon as the
   * equation or the dimension in the section "B - Equation" changes,
   * and EditorOutcome::load to load a different parameter file. The
   * editor writes the parameter file on "save" and "save as".
   *
   * @ingroup Driver
   */
  EditorOutcome run_parameter_editor(ParameterModel &model,
                                     EditorSession &session);
} // namespace ryujin
