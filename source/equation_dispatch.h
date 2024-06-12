//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "time_loop.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <boost/signals2.hpp>

#include <string>

namespace ryujin
{
  /**
   * The Dave error message
   */
  inline const std::string dave =
      "\nDave, this conversation can serve no purpose anymore. Goodbye.\n\n";

  /**
   * Dispatcher class that calls into the right TimeLoop for a configured
   * equation depending on what has been set in the parameter file.
   *
   * @ingroup TimeLoop
   */
  class EquationDispatch : dealii::ParameterAcceptor
  {
  public:
    EquationDispatch()
        : ParameterAcceptor("B - Equation")
    {
      dimension_ = 0;
      add_parameter("dimension", dimension_, "The spatial dimension");
      add_parameter("equation", equation_, "The PDE system");

      time_loop_executed_ = false;
    }


    /**
     * Call create_parameter_files() for all registered equations.
     */
    static void create_parameter_files()
    {
      AssertThrow(signals != nullptr,
                  dealii::ExcMessage(
                      dave + "No equation has been registered. Consequently, "
                             "there is nothing for us to do.\n"));

      signals->create_parameter_files();
    }


    /**
     * Register a create_parameter_files() callback.
     */
    template <typename Callable>
    static void register_create_parameter_files(const Callable &callable)
    {
      if (signals == nullptr)
        signals = new Signals;

      signals->create_parameter_files.connect(callable);
    }


    /**
     * Call dispatch() for all registered equations.
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

      AssertThrow(signals != nullptr,
                  dealii::ExcMessage(
                      dave + "No equation has been registered. Consequently, "
                             "there is nothing for us to do.\n"));

      signals->dispatch(
          dimension_, equation_, parameter_file, mpi_comm, time_loop_executed_);

      AssertThrow(time_loop_executed_ == true,
                  dealii::ExcMessage(dave +
                                     "No equation was dispatched "
                                     "with the chosen equation parameter »" +
                                     equation_ + "«.\n"));
    }


    /**
     * Register a create_parameter_files() callback.
     */
    template <typename Callable>
    static void register_dispatch(const Callable &callable)
    {
      if (signals == nullptr)
        signals = new Signals;

      signals->dispatch.connect(callable);
    }

  protected:
    /**
     * @name Internal data structures:
     */
    //@{

    /**
     * A structure that holds two Signals for equations:
     *  - one for creating and running the appropriate timeloop
     *  - the other signal is used for creating default parameter files.
     */
    struct Signals {
      boost::signals2::signal<void()> create_parameter_files;

      boost::signals2::signal<void(int /*dimension*/,
                                   const std::string & /*equation*/,
                                   const std::string & /*parameter file*/,
                                   const MPI_Comm & /*MPI communicator*/,
                                   bool & /*time loop executed*/)>
          dispatch;
    };

    /*
     * Note: as a static field the pointer is zero initialized before any
     * static/global constructor is run.
     */
    static Signals *signals;

  private:
    //@}
    /**
     * @name Runtime parameters:
     */
    //@{

    int dimension_;
    std::string equation_;

    //@}

    bool time_loop_executed_;
  };


  /**
   * Create default parameter files for the specified equation Description,
   * dimension and number type. This function is called from the respective
   * equation driver.
   */
  template <typename Description, int dim, typename Number>
  void create_prm_files(const std::string &name,
                        bool write_detailed_description)
  {
    {
      /*
       * Create temporary objects for the sole purpose of populating the
       * ParameterAcceptor::prm object.
       */
      ryujin::EquationDispatch equation_dispatch;
      ryujin::TimeLoop<Description, dim, Number> time_loop(MPI_COMM_SELF);

      /* Fix up "equation" entry: */
      auto &prm = dealii::ParameterAcceptor::prm;
      prm.enter_subsection("B - Equation");
      prm.set("dimension", std::to_string(dim));
      prm.set("equation", name);
      prm.leave_subsection();

      std::string base_name = name;
      std::replace(base_name.begin(), base_name.end(), ' ', '_');
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
                dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder

        );
      }
      // all objects have to go out of scope, see
      // https://github.com/dealii/dealii/issues/15111
    }

    dealii::ParameterAcceptor::clear();
  }


  /**
   * A small Dispatch struct templated in Description that registers the
   * call backs.
   */
  template <typename Description, typename Number>
  struct Dispatch {
    Dispatch(const std::string &name)
    {
#ifdef DEBUG_OUTPUT
      std::cout << "Dispatch<Description, Number>::Dispatch() for »" << name
                << "«" << std::endl;
#endif

      EquationDispatch::register_create_parameter_files([name]() {
        create_prm_files<Description, 1, Number>(name, false);
        create_prm_files<Description, 2, Number>(name, true);
        create_prm_files<Description, 3, Number>(name, false);
      });

      EquationDispatch::register_dispatch(
          [name](const int dimension,
                 const std::string &equation,
                 const std::string &parameter_file,
                 const MPI_Comm &mpi_comm,
                 bool &time_loop_executed) {
            if (equation != name)
              return;

            if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
              std::cout << "[INFO] dispatching to driver »" << equation
                        << "« with dim=" << dimension << std::endl;
            }

            AssertThrow(time_loop_executed == false,
                        dealii::ExcMessage(
                            dave +
                            "Trying to execute more than one TimeLoop object "
                            "with the given equation parameter »" +
                            equation + "«"));

            if (dimension == 1) {
              TimeLoop<Description, 1, Number> time_loop(mpi_comm);
              dealii::ParameterAcceptor::initialize(parameter_file);
              time_loop.run();
              time_loop_executed = true;
            } else if (dimension == 2) {
              TimeLoop<Description, 2, Number> time_loop(mpi_comm);
              dealii::ParameterAcceptor::initialize(parameter_file);
              time_loop.run();
              time_loop_executed = true;
            } else if (dimension == 3) {
              TimeLoop<Description, 3, Number> time_loop(mpi_comm);
              dealii::ParameterAcceptor::initialize(parameter_file);
              time_loop.run();
              time_loop_executed = true;
            }
          });
    }
  };

} // namespace ryujin
