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
     * The shallow water equations
     */
    //shallow_water
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Equation,
             LIST({ryujin::Equation::euler, "euler"},
                  {ryujin::Equation::euler_aeos, "euler aeos"},
                  // {ryujin::Equation::shallow_water, "shallow water"},
                  ));
#endif

namespace ryujin
{
#ifndef DOXYGEN
  /*
   * FIXME: This typetrait feels redundant...
   */

  template <typename description>
  constexpr Equation get_equation() = delete;

  template <>
  constexpr Equation get_equation<Euler::Description>()
  {
    return Equation::euler;
  }

  template <>
  constexpr Equation get_equation<EulerAEOS::Description>()
  {
    return Equation::euler_aeos;
  }

//   template <>
//   constexpr Equation get_equation<ShallowWater::Description>()
//   {
//     return Equation::shallow_water;
//   }
#endif


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
        ryujin::TimeLoop<ryujin::Euler::Description, DIM, NUMBER> time_loop(
            mpi_communicator);
        ParameterAcceptor::initialize(parameter_file);
        time_loop.run();
      }; break;
      case Equation::euler_aeos: {
        ryujin::TimeLoop<ryujin::EulerAEOS::Description, DIM, NUMBER> time_loop(
            mpi_communicator);
        ParameterAcceptor::initialize(parameter_file);
        time_loop.run();
      }; break;
//       case Equation::shallow_water: {
//         ryujin::TimeLoop<ryujin::ShallowWater::Description, DIM, NUMBER>
//             time_loop(mpi_communicator);
//         ParameterAcceptor::initialize(parameter_file);
//         time_loop.run();
//       }; break;
      }
    }

  private:
    Equation equation_;
  };


  namespace internal
  {
    template <typename description>
    void create_parameter_template(const std::string &parameter_file,
                                   const MPI_Comm &mpi_communicator)
    {
      constexpr auto equation = get_equation<description>();
      const std::string name =
          dealii::Patterns::Tools::Convert<Equation>::to_string(equation);

      std::string suffix = "." + name;
      std::replace(suffix.begin(), suffix.end(), ' ', '_');

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

        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
          const auto template_file = parameter_file + suffix;

          if (!std::filesystem::exists(template_file))
            prm.print_parameters(template_file,
                                 dealii::ParameterHandler::OutputStyle::PRM);
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
    internal::create_parameter_template<Euler::Description>(parameter_file,
                                                            mpi_communicator);

    internal::create_parameter_template<EulerAEOS::Description>(
        parameter_file, mpi_communicator);

//     internal::create_parameter_template<ShallowWater::Description>(
//         parameter_file, mpi_communicator);
  }
} // namespace ryujin
