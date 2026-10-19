//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/euler_poisson/electrostatic_configuration_constant.h>
#include <ryujin/euler_poisson/electrostatic_configuration_function.h>
#include <ryujin/euler_poisson/electrostatic_configuration_library.h>

namespace ryujin
{
  namespace ElectrostaticConfigurations
  {
    /**
     * The built-in electrostatic configurations.
     */
    template <int dim, typename Number>
    auto defaults()
    {
      using L = ElectrostaticConfigurationLibrary<dim, Number>;
      return std::vector<typename L::Creator>{
          L::template creator<Constant<dim, Number>>(),
          L::template creator<Function<dim, Number>>(),
      };
    }
  } // namespace ElectrostaticConfigurations

  template <>
  auto ElectrostaticConfigurationLibrary<1, double>::defaults()
      -> std::vector<Creator>
  {
    return ElectrostaticConfigurations::defaults<1, double>();
  }

  template <>
  auto ElectrostaticConfigurationLibrary<2, double>::defaults()
      -> std::vector<Creator>
  {
    return ElectrostaticConfigurations::defaults<2, double>();
  }

  template <>
  auto ElectrostaticConfigurationLibrary<3, double>::defaults()
      -> std::vector<Creator>
  {
    return ElectrostaticConfigurations::defaults<3, double>();
  }

  template <>
  auto ElectrostaticConfigurationLibrary<1, float>::defaults()
      -> std::vector<Creator>
  {
    return ElectrostaticConfigurations::defaults<1, float>();
  }

  template <>
  auto ElectrostaticConfigurationLibrary<2, float>::defaults()
      -> std::vector<Creator>
  {
    return ElectrostaticConfigurations::defaults<2, float>();
  }

  template <>
  auto ElectrostaticConfigurationLibrary<3, float>::defaults()
      -> std::vector<Creator>
  {
    return ElectrostaticConfigurations::defaults<3, float>();
  }

  /* instantiations */

  template class Library<ElectrostaticConfiguration<1, double>,
                         const std::string &>;
  template class Library<ElectrostaticConfiguration<2, double>,
                         const std::string &>;
  template class Library<ElectrostaticConfiguration<3, double>,
                         const std::string &>;
  template class Library<ElectrostaticConfiguration<1, float>,
                         const std::string &>;
  template class Library<ElectrostaticConfiguration<2, float>,
                         const std::string &>;
  template class Library<ElectrostaticConfiguration<3, float>,
                         const std::string &>;
} // namespace ryujin
