//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#include <ryujin/geometries/geometry_airfoil.h>
#include <ryujin/geometries/geometry_annulus.h>
#include <ryujin/geometries/geometry_cylinder.h>
#include <ryujin/geometries/geometry_disk.h>
#include <ryujin/geometries/geometry_elevation_profile.h>
#include <ryujin/geometries/geometry_geotiff_profile.h>
#include <ryujin/geometries/geometry_library.h>
#include <ryujin/geometries/geometry_reader.h>
#include <ryujin/geometries/geometry_rectangular_domain.h>
#include <ryujin/geometries/geometry_step.h>
#include <ryujin/geometries/geometry_tank.h>
#include <ryujin/geometries/geometry_two_tanks.h>
#include <ryujin/geometries/geometry_wall.h>

namespace ryujin
{
  namespace Geometries
  {
    /**
     * The built-in geometries.
     */
    template <int dim>
    auto defaults()
    {
      return std::vector<typename GeometryLibrary<dim>::Creator>{
          GeometryLibrary<dim>::template creator<Airfoil<dim>>(),
          GeometryLibrary<dim>::template creator<Annulus<dim>>(),
          GeometryLibrary<dim>::template creator<Cylinder<dim>>(),
          GeometryLibrary<dim>::template creator<Disk<dim>>(),
          GeometryLibrary<dim>::template creator<ElevationProfile<dim>>(),
          GeometryLibrary<dim>::template creator<GeoTIFFProfile<dim>>(),
          GeometryLibrary<dim>::template creator<Reader<dim>>(),
          GeometryLibrary<dim>::template creator<RectangularDomain<dim>>(),
          GeometryLibrary<dim>::template creator<Step<dim>>(),
          GeometryLibrary<dim>::template creator<TwoTanks<dim>>(),
          GeometryLibrary<dim>::template creator<Wall<dim>>(),
          GeometryLibrary<dim>::template creator<WaveTank<dim>>(),
      };
    }
  } // namespace Geometries

  template <>
  auto GeometryLibrary<1>::defaults() -> std::vector<Creator>
  {
    return Geometries::defaults<1>();
  }

  template <>
  auto GeometryLibrary<2>::defaults() -> std::vector<Creator>
  {
    return Geometries::defaults<2>();
  }

  template <>
  auto GeometryLibrary<3>::defaults() -> std::vector<Creator>
  {
    return Geometries::defaults<3>();
  }

  /* instantiations */

  template class Library<Geometry<1>, const std::string &>;
  template class Library<Geometry<2>, const std::string &>;
  template class Library<Geometry<3>, const std::string &>;
} // namespace ryujin
