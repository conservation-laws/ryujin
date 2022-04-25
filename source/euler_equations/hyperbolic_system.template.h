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
    add_parameter("gamma", gamma_, "Euler: Ratio of specific heats");

    b_ = 0.;
    if constexpr (equation_of_state_ == EquationOfState::van_der_waals) {
      add_parameter("b", b_, "Euler: Covolume");
    }

    mu_ = 1.e-3;
    add_parameter("mu", mu_, "Navier Stokes: Shear viscosity");

    lambda_ = 0.;
    add_parameter("lambda", lambda_, "Navier Stokes: Bulk viscosity");

    cv_inverse_kappa_ = 1.866666666666666e-2;
    add_parameter("kappa",
                  cv_inverse_kappa_,
                  "Navier Stokes: Scaled thermal conductivity c_v^{-1} kappa");

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

    static_assert(equation_of_state_ == EquationOfState::ideal_gas,
                  "not implemented");
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
