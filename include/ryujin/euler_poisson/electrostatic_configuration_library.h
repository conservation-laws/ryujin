//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/library.h>
#include <ryujin/euler_poisson/electrostatic_configuration.h>

#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The library of electrostatic configurations: a registry of all
   * electrostatic configurations that the EulerPoisson::ParabolicModule
   * can select by name. The built-in configurations of the namespace
   * ElectrostaticConfigurations are registered by default.
   *
   * @ingroup EulerPoissonEquations
   */
  template <int dim, typename Number = double>
  using ElectrostaticConfigurationLibrary =
      Library<ElectrostaticConfiguration<dim, Number>, const std::string &>;

#ifndef DOXYGEN
  template <>
  auto ElectrostaticConfigurationLibrary<1, double>::defaults()
      -> std::vector<Creator>;
  template <>
  auto ElectrostaticConfigurationLibrary<2, double>::defaults()
      -> std::vector<Creator>;
  template <>
  auto ElectrostaticConfigurationLibrary<3, double>::defaults()
      -> std::vector<Creator>;
  template <>
  auto ElectrostaticConfigurationLibrary<1, float>::defaults()
      -> std::vector<Creator>;
  template <>
  auto ElectrostaticConfigurationLibrary<2, float>::defaults()
      -> std::vector<Creator>;
  template <>
  auto ElectrostaticConfigurationLibrary<3, float>::defaults()
      -> std::vector<Creator>;

  extern template class Library<ElectrostaticConfiguration<1, double>,
                                const std::string &>;
  extern template class Library<ElectrostaticConfiguration<2, double>,
                                const std::string &>;
  extern template class Library<ElectrostaticConfiguration<3, double>,
                                const std::string &>;
  extern template class Library<ElectrostaticConfiguration<1, float>,
                                const std::string &>;
  extern template class Library<ElectrostaticConfiguration<2, float>,
                                const std::string &>;
  extern template class Library<ElectrostaticConfiguration<3, float>,
                                const std::string &>;
#endif
} // namespace ryujin
