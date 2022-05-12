//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
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

    gravity_ = 9.81;
    add_parameter("gravity", gravity_, "Gravitational constant [m/s^2]");

    reference_water_depth_ = 1.;
    add_parameter("reference water depth",
                  reference_water_depth_,
                  "Problem specific water depth reference");

    dry_state_tolerance_ = 1e-8;
    add_parameter("dry state tolerance",
                  dry_state_tolerance_,
                  "Tolerance to define h_tiny to account for dry states");

    mannings_ = 0.;
    add_parameter(
        "mannings", mannings_, "Roughness coefficient for friction source");

    parse_parameters_callback();
  }


  void HyperbolicSystem::parse_parameters_callback()
  {
    /*
     * Precompute some stuff:
     */
    h_tiny_ = reference_water_depth_ * dry_state_tolerance_;
    g_mannings_sqd_ = gravity_ * mannings_ * mannings_;
    reference_speed_ = std::sqrt(gravity_ * reference_water_depth_);
    h_kinetic_energy_tiny_ =
        1.e-9 * (0.5 * std::pow(reference_water_depth_ * reference_speed_, 2));
    tiny_entropy_number_ = 1.e-9 * std::pow(reference_speed_, 3);
  }


#ifndef DOXYGEN
  template <>
  const std::array<std::string, 2> HyperbolicSystem::component_names<1>{
      {"h", "m"}};

  template <>
  const std::array<std::string, 3> HyperbolicSystem::component_names<2>{
      {"h", "m_1", "m_2"}};

  template <>
  const std::array<std::string, 2>
      HyperbolicSystem::primitive_component_names<1>{{"h", "v"}};

  template <>
  const std::array<std::string, 3>
      HyperbolicSystem::primitive_component_names<2>{{"h", "v_1", "v_2"}};

  template <>
  const std::array<std::string, 1> HyperbolicSystem::precomputed_names<1>{
      {"bathymetry"}};

  template <>
  const std::array<std::string, 1> HyperbolicSystem::precomputed_names<2>{
      {"bathymetry"}};
#endif

} /* namespace ryujin */
