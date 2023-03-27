//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

namespace ryujin
{
  namespace Euler
  {
    HyperbolicSystem::HyperbolicSystem(
        const std::string &subsection /*= "HyperbolicSystem"*/)
        : ParameterAcceptor(subsection)
    {
      ParameterAcceptor::parse_parameters_call_back.connect(
          [this] { parse_parameters_callback(); });

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
        HyperbolicSystem::primitive_component_names<1>{{"rho", "v", "p"}};


    template <>
    const std::array<std::string, 4>
        HyperbolicSystem::primitive_component_names<2>{
            {"rho", "v_1", "v_2", "p"}};


    template <>
    const std::array<std::string, 5>
        HyperbolicSystem::primitive_component_names<3>{
            {"rho", "v_1", "v_2", "v_3", "p"}};


    template <>
    const std::array<std::string, 2> HyperbolicSystem::precomputed_names<1>{
        {"s", "eta_h"}};


    template <>
    const std::array<std::string, 2> HyperbolicSystem::precomputed_names<2>{
        {"s", "eta_h"}};


    template <>
    const std::array<std::string, 2> HyperbolicSystem::precomputed_names<3>{
        {"s", "eta_h"}};


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
  } // namespace Euler
} /* namespace ryujin */
