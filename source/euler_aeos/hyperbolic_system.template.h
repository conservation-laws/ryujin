//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include "equation_of_state_library.h"

namespace ryujin
{
  HyperbolicSystem::HyperbolicSystem(
      const std::string &subsection /*= "HyperbolicSystem"*/)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&HyperbolicSystem::parse_parameters_callback, this));

    equation_of_state_ = "polytropic gas";
    add_parameter("equation of state",
                  equation_of_state_,
                  "The equation of state. Valid names are given by any of the "
                  "subsections defined below.");

    gamma_ = 7. / 5.;
    add_parameter("gamma", gamma_, "The ratio of specific heats");

    reference_density_ = 1.;
    add_parameter("reference density",
                  reference_density_,
                  "Problem specific density reference");

    vacuum_state_relaxation_ = 10000.;
    add_parameter("vacuum state relaxation",
                  vacuum_state_relaxation_,
                  "Problem specific vacuum relaxation parameter");

    /*
     * And finally populate the equation of state list with all equation of
     * state configurations defined in the EquationOfState namespace:
     */
    EquationOfStateLibrary::populate_equation_of_state_list(
        equation_of_state_list_, subsection);

    parse_parameters_callback();
  }


  void HyperbolicSystem::parse_parameters_callback()
  {
    gamma_inverse_ = 1. / gamma_;
    gamma_plus_one_inverse_ = 1. / (gamma_ + 1.);

    bool initialized = false;
    for (auto &it : equation_of_state_list_)
      if (it->name() == equation_of_state_) {
        pressure_oracle_ = [&it](double rho, double e) {
          return it->pressure_oracle(rho, e);
        };

        problem_name = "Compressible Euler equations (" + it->name() + " EOS)";

        initialized = true;
        break;
      }

    AssertThrow(
        initialized,
        dealii::ExcMessage(
            "Could not find an equation of state description with name \"" +
            equation_of_state_ + "\""));

  }


#ifndef DOXYGEN
  template <>
  const std::array<std::string, 3> HyperbolicSystem::component_names<1>{
      {"rho", "m", "E"}};

  template <>
  const std::array<std::string, 4> HyperbolicSystem::component_names<2>{
      {"rho", "m_1", "m_2", "E"}};

  template <>
  const std::array<std::string, 5> HyperbolicSystem::component_names<3>{
      {"rho", "m_1", "m_2", "m_3", "E"}};

  template <>
  const std::array<std::string, 3>
      HyperbolicSystem::primitive_component_names<1>{{"rho", "u", "e"}};

  template <>
  const std::array<std::string, 4>
      HyperbolicSystem::primitive_component_names<2>{
          {"rho", "v_1", "v_2", "e"}};

  template <>
  const std::array<std::string, 5>
      HyperbolicSystem::primitive_component_names<3>{
          {"rho", "v_1", "v_2", "v_3", "e"}};

  template <>
  const std::array<std::string, 4> HyperbolicSystem::precomputed_names<1>{
      {"p", "gamma", "surrogate_entropy", "eta_harten"}};

  template <>
  const std::array<std::string, 4> HyperbolicSystem::precomputed_names<2>{
      {"p", "gamma", "surrogate_entropy", "eta_harten"}};

  template <>
  const std::array<std::string, 4> HyperbolicSystem::precomputed_names<3>{
      {"p", "gamma", "surrogate_entropy", "eta_harten"}};

  template <>
  const std::array<std::string, 0>
      HyperbolicSystem::precomputed_initial_names<1>{};

  template <>
  const std::array<std::string, 0>
      HyperbolicSystem::precomputed_initial_names<2>{};

  template <>
  const std::array<std::string, 0>
      HyperbolicSystem::precomputed_initial_names<3>{};
#endif

} /* namespace ryujin */
