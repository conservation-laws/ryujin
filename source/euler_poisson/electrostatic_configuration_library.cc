//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include "electrostatic_configuration_library.template.h"

namespace ryujin
{
  namespace ElectrostaticConfigurationLibrary
  {
    template void populate_electrostatic_configuration_list<1, double>(
        electrostatic_configuration_list_type<1, double> &,
        const std::string &);
    template void populate_electrostatic_configuration_list<2, double>(
        electrostatic_configuration_list_type<2, double> &,
        const std::string &);
    template void populate_electrostatic_configuration_list<3, double>(
        electrostatic_configuration_list_type<3, double> &,
        const std::string &);

    template void populate_electrostatic_configuration_list<1, float>(
        electrostatic_configuration_list_type<1, float> &, const std::string &);
    template void populate_electrostatic_configuration_list<2, float>(
        electrostatic_configuration_list_type<2, float> &, const std::string &);
    template void populate_electrostatic_configuration_list<3, float>(
        electrostatic_configuration_list_type<3, float> &, const std::string &);
  } // namespace ElectrostaticConfigurationLibrary
} // namespace ryujin
