//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "parabolic_system.h"

namespace ryujin
{
  ParabolicSystem::ParabolicSystem(
      const std::string &subsection /*= "ParabolicSystem"*/)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&ParabolicSystem::parse_parameters_callback, this));

    mu_ = 1.e-3;
    add_parameter("mu", mu_, "The shear viscosity");

    lambda_ = 0.;
    add_parameter("lambda", lambda_, "The bulk viscosity");

    cv_inverse_kappa_ = 1.866666666666666e-2;
    add_parameter("kappa",
                  cv_inverse_kappa_,
                  "The scaled thermal conductivity c_v^{-1} kappa");

    parse_parameters_callback();
  }


  void ParabolicSystem::parse_parameters_callback()
  {
  }
} /* namespace ryujin */
