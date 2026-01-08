//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include "electrostatic_configuration_constant.h"
#include "electrostatic_configuration_function.h"
#include "electrostatic_configuration_library.h"

namespace ryujin
{
  namespace ElectrostaticConfigurationLibrary
  {
    template <int dim, typename Number>
    using electrostatic_configuration_list_type =
        std::set<std::shared_ptr<ElectrostaticConfiguration<dim, Number>>>;

    template <int dim, typename Number>
    void populate_electrostatic_configuration_list(
        electrostatic_configuration_list_type<dim, Number> &list,
        const std::string &subsection)
    {
      auto add = [&](auto &&object) { list.emplace(std::move(object)); };
      add(std::make_shared<Constant<dim, Number>>(subsection));
      add(std::make_shared<Function<dim, Number>>(subsection));
    }
  } // namespace ElectrostaticConfigurationLibrary
} // namespace ryujin
