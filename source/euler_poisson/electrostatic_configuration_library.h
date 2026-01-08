//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "electrostatic_configuration.h"

namespace ryujin
{
  namespace ElectrostaticConfigurationLibrary
  {
    template <int dim, typename Number>
    using electrostatic_configuration_list_type =
        std::set<std::shared_ptr<ElectrostaticConfiguration<dim, Number>>>;

    /**
     * Populate a given container with all equation of states defined in
     * this namespace.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number>
    void populate_electrostatic_configuration_list(
        electrostatic_configuration_list_type<dim, Number> &list,
        const std::string &subsection);

  } // namespace ElectrostaticConfigurationLibrary
} // namespace ryujin
