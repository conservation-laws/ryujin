//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/library.h>
#include <ryujin/geometries/geometry.h>

#include <string>
#include <vector>

namespace ryujin
{
  /**
   * The library of geometries: a registry of all geometries that the
   * Discretization can select by name. The built-in geometries of the
   * namespace Geometries are registered by default. A client adds a
   * geometry of its own with
   * @code
   * GeometryLibrary<dim>::add<MyGeometry<dim>>();
   * @endcode
   *
   * @ingroup Geometries
   */
  template <int dim>
  using GeometryLibrary = Library<Geometry<dim>, const std::string &>;

#ifndef DOXYGEN
  template <>
  auto GeometryLibrary<1>::defaults() -> std::vector<Creator>;
  template <>
  auto GeometryLibrary<2>::defaults() -> std::vector<Creator>;
  template <>
  auto GeometryLibrary<3>::defaults() -> std::vector<Creator>;

  extern template class Library<Geometry<1>, const std::string &>;
  extern template class Library<Geometry<2>, const std::string &>;
  extern template class Library<Geometry<3>, const std::string &>;
#endif
} // namespace ryujin
