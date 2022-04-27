//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#include "parabolic_system.template.h"

namespace ryujin
{
  const std::array<std::string, ParabolicSystem::n_implicit_systems>
      ParabolicSystem::implicit_system_names{{"vel", "int"}};
} /* namespace ryujin */
