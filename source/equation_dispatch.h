//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "convenience_macros.h"
#include "patterns_conversion.h"
#include "time_loop.h"

#include "euler/description.h"
#include "euler_aeos/description.h"
#include "navier_stokes/description.h"
// #include "shallow_water/description.h"

#include <deal.II/base/mpi.h>

#include <filesystem>

namespace ryujin
{
  /**
   * An enum class that controls what problem description to use.
   */
  enum class Equation {
    /**
     * The compressible Euler equations of gas dynamics with a polytropic
     * gas equation of state
     */
    euler,

    /**
     * The compressible Euler equations of gas dynamics with arbitrary
     * equation of state
     */
    euler_aeos,

    /**
     * The compressible Navier-Stokes equations of gas dynamics with a
     * polytropic gas equation of state, Newtonian fluid viscosity model,
     * and a heat flux governed by  Fourier's law.
     */
    navier_stokes,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Equation,
             LIST({ryujin::Equation::euler, "euler"},
                  {ryujin::Equation::euler_aeos, "euler aeos"},
                  {ryujin::Equation::navier_stokes, "navier stokes"},
                  // {ryujin::Equation::shallow_water, "shallow water"},
                  ));
#endif

namespace ryujin
{
  /**
   * Dispatcher class that calls into the right TimeLoop depending on
   * what equation has been selected.
   */
  class EquationDispatch : dealii::ParameterAcceptor
  {
  public:
    EquationDispatch()
        : ParameterAcceptor("B - Equation")
    {
      equation_ = Equation::euler;
      add_parameter("equation", equation_, "The PDE system");
    }

    void run(const std::string &parameter_file,
             const MPI_Comm &mpi_communicator)
    {
      std::string name =
          dealii::Patterns::Tools::Convert<Equation>::to_string(equation_);

      ParameterAcceptor::prm.parse_input(parameter_file,
                                         "",
                                         /* skip undefined */ true,
                                         /* assert entries present */ false);

      name = dealii::Patterns::Tools::Convert<Equation>::to_string(equation_);

      switch (equation_) {
      case Equation::euler: {
        TimeLoop<Euler::Description, DIM, NUMBER> time_loop(mpi_communicator);
        ParameterAcceptor::initialize(parameter_file);
        time_loop.run();
      }; break;
      case Equation::euler_aeos: {
        TimeLoop<EulerAEOS::Description, DIM, NUMBER> time_loop(
            mpi_communicator);
        ParameterAcceptor::initialize(parameter_file);
        time_loop.run();
      }; break;
      case Equation::navier_stokes: {
        TimeLoop<NavierStokes::Description, DIM, NUMBER> time_loop(
            mpi_communicator);
        ParameterAcceptor::initialize(parameter_file);
        time_loop.run();
      }; break;
      }
    }

  private:
    Equation equation_;
  };


  namespace internal
  {
    template <typename description>
    void create_parameter_template(const std::string &name,
                                   const MPI_Comm &mpi_communicator)
    {
      {
        /*
         * Create temporary objects for the sole purpose of populating the
         * ParameterAcceptor::prm object.
         */
        ryujin::EquationDispatch equation_dispatch;
        ryujin::TimeLoop<description, DIM, NUMBER> time_loop(mpi_communicator);

        /* Fix up "equation" entry: */
        auto &prm = dealii::ParameterAcceptor::prm;
        prm.enter_subsection("B - Equation");
        prm.set("equation", name);
        prm.leave_subsection();

        std::string base_name = name;
        std::replace(base_name.begin(), base_name.end(), ' ', '_');

        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
          const auto full_name = base_name + "-default_parameters.prm";
          if (!std::filesystem::exists(full_name))
            prm.print_parameters(full_name,
                                 dealii::ParameterHandler::OutputStyle::PRM);

          const auto short_name = base_name + "-brief.prm";
          if (!std::filesystem::exists(short_name))
            prm.print_parameters(short_name,
                                 dealii::ParameterHandler::OutputStyle::Short);
        }
        // all objects have to go out of scope, see
        // https://github.com/dealii/dealii/issues/15111
      }
      dealii::ParameterAcceptor::clear();
    }
  } // namespace internal


  void create_parameter_templates(const std::string &parameter_file,
                                  const MPI_Comm &mpi_communicator)
  {
    internal::create_parameter_template<Euler::Description>("euler",
                                                            mpi_communicator);
    std::filesystem::copy("euler-brief.prm", parameter_file);

    internal::create_parameter_template<EulerAEOS::Description>(
        "euler aeos", mpi_communicator);
    internal::create_parameter_template<NavierStokes::Description>(
        "navier stokes", mpi_communicator);
  }
} // namespace ryujin
