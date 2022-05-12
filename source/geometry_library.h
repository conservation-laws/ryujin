//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "geometry.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tensor_product_manifold.h>
#include <deal.II/grid/tria.h>


namespace ryujin
{
  /**
   * This namespace provides a collection of functions for generating
   * triangulations for some benchmark configurations.
   *
   * @ingroup Mesh
   */
  namespace GridGenerator
  {
    using namespace dealii::GridGenerator;
  } /* namespace GridGenerator */
} /* namespace ryujin */


#include "geometry_airfoil.h"
#include "geometry_cylinder.h"
#include "geometry_rectangular_domain.h"
#include "geometry_step.h"
#include "geometry_wall.h"


namespace ryujin
{
  /**
   * A namespace for a number of benchmark geometries and dealii::GridIn
   * wrappers.
   *
   * @ingroup Mesh
   */
  namespace Geometries
  {
    /**
     * Populate a given container with all initial state defined in this
     * namespace
     *
     * @ingroup Mesh
     */
    template <int dim, typename T>
    void populate_geometry_list(T &geometry_list, const std::string &subsection)
    {
      auto add = [&](auto &&object) {
        geometry_list.emplace(std::move(object));
      };

      add(std::make_unique<Cylinder<dim>>(subsection));
      add(std::make_unique<Step<dim>>(subsection));
      add(std::make_unique<Wall<dim>>(subsection));
      add(std::make_unique<RectangularDomain<dim>>(subsection));
      add(std::make_unique<Airfoil<dim>>(subsection));
    }
  } /* namespace Geometries */
} /* namespace ryujin */
