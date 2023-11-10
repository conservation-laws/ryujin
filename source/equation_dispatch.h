//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "patterns_conversion.h"
#include "time_loop.h"

#include "euler/description.h"
#include "euler_aeos/description.h"
#include "navier_stokes/description.h"
#include "scalar_conservation/description.h"
#include "shallow_water/description.h"

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

    /**
     * A scalar conservation equation with a user-specified flux depending
     * on the state.
     */
    scalar_conservation,

    /**
     * The shallow water equations
     */
    shallow_water,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Equation,
             LIST({ryujin::Equation::euler, "euler"},
                  {ryujin::Equation::euler_aeos, "euler aeos"},
                  {ryujin::Equation::navier_stokes, "navier stokes"},
                  {ryujin::Equation::scalar_conservation,
                   "scalar conservation"},
                  {ryujin::Equation::shallow_water, "shallow water"},
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
      dimension_ = 2;
      add_parameter("dimension", dimension_, "The spatial dimension");

      equation_ = Equation::euler;
      add_parameter("equation", equation_, "The PDE system");
    }

    void run(const std::string &parameter_file, const MPI_Comm &mpi_comm)
    {
      ParameterAcceptor::prm.parse_input(parameter_file,
                                         "",
                                         /* skip undefined */ true,
                                         /* assert entries present */ false);

      AssertThrow(
          dimension_ >= 1 && dimension_ <= 3,
          dealii::ExcMessage("Dave, this conversation can serve no purpose "
                             "anymore. Goodbye.\nThe dimension parameter needs "
                             "to be either 1, 2, or 3."));

      switch (equation_) {
      case Equation::euler:
        if (dimension_ == 1) {
          TimeLoop<Euler::Description, 1, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 2) {
          TimeLoop<Euler::Description, 2, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 3) {
          TimeLoop<Euler::Description, 3, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else
          __builtin_unreachable();
        break;
      case Equation::euler_aeos:
        if (dimension_ == 1) {
          TimeLoop<EulerAEOS::Description, 1, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 2) {
          TimeLoop<EulerAEOS::Description, 2, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 3) {
          TimeLoop<EulerAEOS::Description, 3, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else
          __builtin_unreachable();
        break;
      case Equation::navier_stokes:
        if (dimension_ == 1) {
          TimeLoop<NavierStokes::Description, 1, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 2) {
          TimeLoop<NavierStokes::Description, 2, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 3) {
          TimeLoop<NavierStokes::Description, 3, NUMBER> time_loop(mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else
          __builtin_unreachable();
        break;
      case Equation::scalar_conservation:
        if (dimension_ == 1) {
          TimeLoop<ScalarConservation::Description, 1, NUMBER> time_loop(
              mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 2) {
          TimeLoop<ScalarConservation::Description, 2, NUMBER> time_loop(
              mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 3) {
          TimeLoop<ScalarConservation::Description, 3, NUMBER> time_loop(
              mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else
          __builtin_unreachable();
        break;
      case Equation::shallow_water:
        if (dimension_ == 1) {
          TimeLoop<ShallowWater::Description, 1, NUMBER> time_loop(
              mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 2) {
          TimeLoop<ShallowWater::Description, 2, NUMBER> time_loop(
              mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else if (dimension_ == 3) {
          TimeLoop<ShallowWater::Description, 3, NUMBER> time_loop(
              mpi_comm);
          ParameterAcceptor::initialize(parameter_file);
          time_loop.run();
        } else
          __builtin_unreachable();
        break;
      default:
        __builtin_trap();
      }
    }

  private:
    int dimension_;
    Equation equation_;
  };


  namespace internal
  {
    template <int dim, typename description>
    void create_prm_files(const std::string &name,
                          const MPI_Comm &mpi_communicator,
                          bool write_detailed_description)
    {
      {
        /*
         * Create temporary objects for the sole purpose of populating the
         * ParameterAcceptor::prm object.
         */
        ryujin::EquationDispatch equation_dispatch;
        ryujin::TimeLoop<description, dim, NUMBER> time_loop(mpi_communicator);

        /* Fix up "equation" entry: */
        auto &prm = dealii::ParameterAcceptor::prm;
        prm.enter_subsection("B - Equation");
        prm.set("dimension", std::to_string(dim));
        prm.set("equation", name);
        prm.leave_subsection();

        std::string base_name = name;
        std::replace(base_name.begin(), base_name.end(), ' ', '_');
        base_name += "-" + std::to_string(dim) + "d";

        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
          const auto full_name =
              "default_parameters-" + base_name + "-description.prm";
          if (write_detailed_description)
            prm.print_parameters(full_name,
                                 dealii::ParameterHandler::OutputStyle::PRM);

          const auto short_name = "default_parameters-" + base_name + ".prm";
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
    internal::create_prm_files<1, Euler::Description>(
        "euler", mpi_communicator, false);
    internal::create_prm_files<2, Euler::Description>(
        "euler", mpi_communicator, true);
    internal::create_prm_files<3, Euler::Description>(
        "euler", mpi_communicator, false);
    std::filesystem::copy("default_parameters-euler-2d.prm", parameter_file);

    internal::create_prm_files<1, EulerAEOS::Description>(
        "euler aeos", mpi_communicator, false);
    internal::create_prm_files<2, EulerAEOS::Description>(
        "euler aeos", mpi_communicator, true);
    internal::create_prm_files<3, EulerAEOS::Description>(
        "euler aeos", mpi_communicator, false);

    internal::create_prm_files<1, NavierStokes::Description>(
        "navier stokes", mpi_communicator, false);
    internal::create_prm_files<2, NavierStokes::Description>(
        "navier stokes", mpi_communicator, true);
    internal::create_prm_files<3, NavierStokes::Description>(
        "navier stokes", mpi_communicator, false);

    internal::create_prm_files<1, ScalarConservation::Description>(
        "scalar conservation", mpi_communicator, false);
    internal::create_prm_files<2, ScalarConservation::Description>(
        "scalar conservation", mpi_communicator, true);
    internal::create_prm_files<3, ScalarConservation::Description>(
        "scalar conservation", mpi_communicator, false);

    internal::create_prm_files<1, ShallowWater::Description>(
        "shallow water", mpi_communicator, false);
    internal::create_prm_files<2, ShallowWater::Description>(
        "shallow water", mpi_communicator, true);
    internal::create_prm_files<3, ShallowWater::Description>(
        "shallow water", mpi_communicator, false);
  }
} // namespace ryujin
