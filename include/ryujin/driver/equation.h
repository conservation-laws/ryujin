//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/driver/task_composition.h>
#include <ryujin/stub/parabolic_module.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

namespace ryujin
{
  /**
   * Populate dealii::ParameterAcceptor::prm with all parameters of the
   * objects created by @p setup, and return the objects. The objects have
   * to go out of scope before calling dealii::ParameterAcceptor::clear().
   *
   * @ingroup Driver
   */
  template <typename Setup>
  auto populate_parameters(const Setup &setup)
  {
    auto &prm = dealii::ParameterAcceptor::prm;

    /*
     * Workaround: The Scheduler is constructed last and declares the "A -
     * Scheduler" section last. Add an entry to the section so that it
     * shows up first.
     */
    prm.enter_subsection("A - Scheduler");
    prm.declare_entry("basename", "test");
    prm.leave_subsection();

    return setup();
  }


  /**
   * Write the default parameter file "default_parameters-<name>.prm", and,
   * if @p write_detailed_description is set, the parameter file
   * "default_parameters-<name>-description.prm" with the documentation of
   * all parameters.
   *
   * @code
   * ryujin::write_parameter_files("euler-2d", true, [] {
   *   struct Objects {
   *     Simulation<HyperbolicDescription, ParabolicDescription, 2> simulation{
   *         MPI_COMM_SELF};
   *     TaskComposition<2> task_composition{simulation};
   *   };
   *   return std::make_unique<Objects>();
   * });
   * @endcode
   *
   * @ingroup Driver
   */
  template <typename Setup>
  void write_parameter_files(const std::string &name,
                             bool write_detailed_description,
                             const Setup &setup)
  {
    {
      auto &prm = dealii::ParameterAcceptor::prm;

      [[maybe_unused]] const auto objects = populate_parameters(setup);

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_SELF) == 0) {
        if (write_detailed_description)
          prm.print_parameters(
              "default_parameters-" + name + "-description.prm",
              dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder);

        prm.print_parameters(
            "default_parameters-" + name + ".prm",
            dealii::ParameterHandler::OutputStyle::Short |
                dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder);
      }

      // all objects have to go out of scope, see
      // https://github.com/dealii/dealii/issues/15111
    }

    dealii::ParameterAcceptor::clear();
  }


  /**
   * An equation: a pairing of a hyperbolic and a parabolic description
   * that the ryujin executable selects by name with the run time
   * parameter "equation".
   *
   * @ingroup Driver
   */
  struct Equation {
    /**
     * Construct and run the simulation for dimension @p dim with the
     * parameters of @p parameter_file.
     */
    std::function<void(
        int dim, const std::string &parameter_file, const MPI_Comm &mpi_comm)>
        run;

    /**
     * Write the default parameter files of the equation for all
     * dimensions.
     */
    std::function<void()> create_parameter_files;

    /**
     * Populate dealii::ParameterAcceptor::prm with all parameters of the
     * equation for dimension @p dim, call @p f, and clear the
     * ParameterAcceptor afterwards.
     */
    std::function<void(int dim, const std::function<void()> &f)>
        with_parameters;
  };


  /**
   * All registered equations, sorted by name. The equations of ryujin
   * are registered by default.
   *
   * @ingroup Driver
   */
  std::map<std::string, Equation> &equations();


  /**
   * Register the equation that pairs the hyperbolic description @p HD
   * with the parabolic description @p PD and the time integrator @p TI
   * under the given name. An equation is registered with a constant at
   * namespace scope:
   * @code
   * const EquationRegistration<HD, PD> registration("equation");
   * @endcode
   *
   * @ingroup Driver
   */
  template <typename HD,
            typename PD = StubParabolicDescription,
            template <int, typename> class TI = TimeIntegrator>
  class EquationRegistration final
  {
  public:
    /**
     * Constructor taking the name of the equation.
     */
    EquationRegistration(const std::string &name)
    {
      const auto run = [](int dim,
                          const std::string &parameter_file,
                          const MPI_Comm &mpi_communicator) {
        dispatch(dim, [&](auto d) {
          run_simulation<d.value>(parameter_file, mpi_communicator);
        });
      };

      const auto create_parameter_files = [name] {
        create_parameter_file<1>(name, false);
        create_parameter_file<2>(name, true);
        create_parameter_file<3>(name, false);
      };

      const auto with_parameters = [name](int dim,
                                          const std::function<void()> &f) {
        dispatch(dim, [&](auto d) { call_with_parameters<d.value>(name, f); });
      };

      const Equation equation{run, create_parameter_files, with_parameters};

      const bool inserted = equations().emplace(name, equation).second;
      AssertThrow(inserted,
                  dealii::ExcMessage("The equation »" + name +
                                     "« is registered more than once."));
    }

  private:
    /**
     * Call @p f with std::integral_constant<int, dim> for the run time
     * dimension @p dim.
     */
    template <typename F>
    static auto dispatch(int dim, const F &f)
    {
      switch (dim) {
      case 1:
        return f(std::integral_constant<int, 1>{});
      case 2:
        return f(std::integral_constant<int, 2>{});
      case 3:
        return f(std::integral_constant<int, 3>{});
      default:
        AssertThrow(false, dealii::ExcInternalError());
        __builtin_unreachable();
      }
    }

    /**
     * Construct the Simulation and the TaskComposition for dimension
     * @p dim, initialize all parameters from @p parameter_file, and run.
     */
    template <int dim>
    static void run_simulation(const std::string &parameter_file,
                               const MPI_Comm &mpi_comm)
    {
      Simulation<HD, PD, dim, NUMBER, TI> simulation(mpi_comm);
      TaskComposition<dim, NUMBER> task_composition(simulation);

      dealii::ParameterAcceptor::initialize(parameter_file);

      task_composition.run();
    }

    /**
     * Declare the parameters "dimension" and "equation" (see
     * source/main.cc), set to this equation, and create the Simulation
     * and the TaskComposition for the sole purpose of populating the
     * ParameterAcceptor::prm object.
     */
    template <int dim>
    static auto create_objects(const std::string &name)
    {
      std::string names;
      for (const auto &[key, equation] : equations())
        names += (names.empty() ? "" : "|") + key;

      auto &prm = dealii::ParameterAcceptor::prm;
      prm.enter_subsection("B - Equation");
      prm.declare_entry("dimension",
                        std::to_string(dim),
                        dealii::Patterns::Integer(1, 3),
                        "The spatial dimension");
      prm.declare_entry("equation",
                        name,
                        dealii::Patterns::Selection(names),
                        "The PDE system");
      prm.leave_subsection();

      struct Objects {
        Simulation<HD, PD, dim, NUMBER, TI> simulation{MPI_COMM_SELF};
        TaskComposition<dim, NUMBER> task_composition{simulation};
      };
      return std::make_unique<Objects>();
    }

    /**
     * Write the default parameter file(s) for dimension @p dim.
     */
    template <int dim>
    static void create_parameter_file(const std::string &name,
                                      bool write_detailed_description)
    {
      std::string base_name = name;
      std::ranges::replace(base_name, ' ', '_');
      base_name += "-" + std::to_string(dim) + "d";

      write_parameter_files(base_name, write_detailed_description, [&name] {
        return create_objects<dim>(name);
      });
    }

    /**
     * Call @p f with all parameters for dimension @p dim, see
     * Equation::with_parameters.
     */
    template <int dim>
    static void call_with_parameters(const std::string &name,
                                     const std::function<void()> &f)
    {
      {
        [[maybe_unused]] const auto objects =
            populate_parameters([&name] { return create_objects<dim>(name); });

        f();

        // all objects have to go out of scope, see
        // https://github.com/dealii/dealii/issues/15111
      }

      dealii::ParameterAcceptor::clear();
    }
  };
} // namespace ryujin
