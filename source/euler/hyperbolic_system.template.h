//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

namespace ryujin
{
  HyperbolicSystem::HyperbolicSystem(
      const std::string &subsection /*= "HyperbolicSystem"*/)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&HyperbolicSystem::parse_parameters_callback, this));

    gamma_ = 7. / 5.;
    add_parameter("gamma", gamma_, "The ratio of specific heats");

    parse_parameters_callback();
  }


  void HyperbolicSystem::parse_parameters_callback()
  {
    /*
     * Precompute a number of derived gamma coefficients that contain
     * divisions:
     */
    gamma_inverse_ = 1. / gamma_;
    gamma_plus_one_inverse_ = 1. / (gamma_ + 1.);
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
      HyperbolicSystem::primitive_component_names<1>{{"rho", "u", "p"}};

  template <>
  const std::array<std::string, 4>
      HyperbolicSystem::primitive_component_names<2>{
          {"rho", "v_1", "v_2", "p"}};

  template <>
  const std::array<std::string, 5>
      HyperbolicSystem::primitive_component_names<3>{
          {"rho", "v_1", "v_2", "v_3", "p"}};
#endif

} /* namespace ryujin */
