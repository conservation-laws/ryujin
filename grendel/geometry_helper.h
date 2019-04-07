#ifndef GEOMETRY_HELPER_H
#define GEOMETRY_HELPER_H

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/manifold_lib.h>

namespace grendel
{
  /*
   * This header file contains a number of helper functions to create
   * varios different meshes.  It is used in the Discretization class.
   */


  /**
   * Create a 2D triangulation consisting of a rectangle with a prescribed
   * length and height with an insribed obstacle given by a centered,
   * equilateral triangle.
   *
   * Even though this function has a template parameter @p dim, it is only
   * implemented for dimension 2.
   */
  template <int dim>
  void create_coarse_grid_triangle(
      dealii::parallel::distributed::Triangulation<dim> &,
      const double /*length*/,
      const double /*height*/,
      const double /*object_height*/)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }


  template <>
  void create_coarse_grid_triangle<2>(
      dealii::parallel::distributed::Triangulation<2> &triangulation,
      const double length,
      const double height,
      const double object_height)
  {
    constexpr int dim = 2;

    using namespace dealii;

    const double object_length = object_height * std::sqrt(3) / 2.;

    const std::vector<Point<dim>> vertices{
        {0., 0.},                                     // 0, bottom left
        {(length - object_length) / 2., 0.},          // 1, bottom center left
        {(length + object_length) / 2., 0.},          // 2, bottom center right
        {length, 0.},                                 // 3, bottom right
        {0., height / 2.},                            // 4, middle left
        {(length - object_length) / 2., height / 2.}, // 5, middle center left
        {(length + object_length) / 2.,
         (height - object_height) / 2.},         // 6, middle lower center right
        {length, (height - object_height) / 2.}, // 7, middle lower right
        {(length + object_length) / 2.,
         (height + object_height) / 2.},         // 8, middle upper center right
        {length, (height + object_height) / 2.}, // 9, middle upper right
        {0., height},                            // 10, top left
        {(length - object_length) / 2., height}, // 11, top center left
        {(length + object_length) / 2., height}, // 12, top center right
        {length, height}                         // 13, top right
    };

    std::vector<CellData<dim>> cells(7);
    {
      const auto assign = [](auto b, std::array<unsigned int, 4> a) {
        std::copy(a.begin(), a.end(), b);
      };
      assign(cells[0].vertices, {0, 1, 4, 5});
      assign(cells[1].vertices, {1, 2, 5, 6});
      assign(cells[2].vertices, {2, 3, 6, 7});
      assign(cells[3].vertices, {4, 5, 10, 11});
      assign(cells[4].vertices, {5, 8, 11, 12});
      assign(cells[5].vertices, {8, 9, 12, 13});
      assign(cells[6].vertices, {6, 7, 8, 9});
    }

    triangulation.create_triangulation(vertices, cells, SubCellData());

    /*
     * Set boundary ids:
     */

    for (auto cell : triangulation.active_cell_iterators()) {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if (!face->at_boundary())
          continue;

        /*
         * We want slip boundary conditions (i.e. indicator 1) at top
         * bottom and on the triangle. On the left and right side we leave
         * the boundary indicator at 0, i.e. do nothing.
         */
        const auto center = face->center();
        if (center[0] > 0. && center[0] < length)
          face->set_boundary_id(1);
      }
    }
  }


  /**
   * Create an nD tube with a given length and diameter. More precisely,
   * this is a line in 1D, a rectangle in 2D, and a cylinder in 3D.
   *
   * We set boundary indicator 0 on the left and right side to indicate "do
   * nothing" boundary conditions, and boundary indicator 1 at the top and
   * bottom side in 2D, as well as the curved portion of the boundary in 3D
   * to indicate "slip boundary conditions".
   */

  template <int dim>
  void
  create_coarse_grid_tube(dealii::parallel::distributed::Triangulation<dim> &,
                          const double /*length*/,
                          const double /*diameter*/) = delete;


  template <>
  void create_coarse_grid_tube<1>(
      dealii::parallel::distributed::Triangulation<1> &triangulation,
      const double length,
      const double /*diameter*/)
  {
    dealii::GridGenerator::hyper_cube(triangulation, 0., length);
    triangulation.begin_active()->face(0)->set_boundary_id(0);
    triangulation.begin_active()->face(1)->set_boundary_id(0);
  }


  template <>
  void create_coarse_grid_tube<2>(
      dealii::parallel::distributed::Triangulation<2> &triangulation,
      const double length,
      const double diameter)
  {
    using namespace dealii;

    GridGenerator::hyper_rectangle(triangulation,
                                   Point<2>(-length / 2., -diameter / 2.),
                                   Point<2>(length / 2., diameter / 2.));

    /*
     * Set boundary ids:
     */

    for (auto cell : triangulation.active_cell_iterators()) {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if (!face->at_boundary())
          continue;

        /*
         * We want slip boundary conditions (i.e. indicator 1) at top and
         * bottom of the rectangle. On the left side we enforce initial
         * conditionas and leave the boundary indicator on the right side
         * at 0, i.e. do nothing.
         */

        const auto center = face->center();
        if (center[0] < -length / 2. + 1.e-6)
          face->set_boundary_id(2);
        else if (std::abs(center[1]) > diameter / 2. - 1.e-6)
          face->set_boundary_id(1);
      }
    }
  }


  template <>
  void create_coarse_grid_tube<3>(
      dealii::parallel::distributed::Triangulation<3> &triangulation,
      const double length,
      const double diameter)
  {
    using namespace dealii;

    GridGenerator::cylinder(triangulation, diameter / 2., length / 2.);

    /*
     * Set boundary ids:
     */

    for (auto cell : triangulation.active_cell_iterators()) {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if (!face->at_boundary())
          continue;

        /*
         * We want slip boundary conditions (i.e. indicator 1) at top and
         * bottom of the rectangle. On the left side we enforce initial
         * conditionas and leave the boundary indicator on the right side
         * at 0, i.e. do nothing.
         */

        const auto center = face->center();
        if (center[0] < -length / 2. + 1.e-6)
          face->set_boundary_id(2);
        else if (center[0] > length / 2. - 1.e-6)
          face->set_boundary_id(0);

        face->set_boundary_id(1);
      }
    }
  }


  /**
   * Create the 2D mach step triangulation.
   *
   * Even though this function has a template parameter @p dim, it is only
   * implemented for dimension 2.
   */

  template <int dim>
  void
  create_coarse_grid_step(dealii::parallel::distributed::Triangulation<dim> &,
                          const double /*length*/,
                          const double /*height*/,
                          const double /*step_position*/,
                          const double /*step_height*/)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }


  template <>
  void create_coarse_grid_step<2>(
      dealii::parallel::distributed::Triangulation<2> &triangulation,
      const double length,
      const double height,
      const double step_position,
      const double step_height)
  {
    constexpr int dim = 2;

    using namespace dealii;

    Triangulation<dim> tria1;

    GridGenerator::subdivided_hyper_rectangle(
        tria1, {15, 4}, Point<2>(0., step_height), Point<2>(length, height));

    Triangulation<dim> tria2;

    GridGenerator::subdivided_hyper_rectangle(
        tria2, {3, 1}, Point<2>(0., 0.), Point<2>(step_position, step_height));

    GridGenerator::merge_triangulations(tria1, tria2, triangulation);

    /*
     * Set boundary ids:
     */

    for (auto cell : triangulation.active_cell_iterators()) {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if (!face->at_boundary())
          continue;

        /*
         * We want slip boundary conditions (i.e. indicator 1) at top
         * and bottom of the rectangle. On the left we set inflow
         * conditions (i.e. indicator 2), and we do nothing on the right
         * side (i.e. indicator 0).
         */

        const auto center = face->center();

        if (center[0] > 0. + 1.e-6 && center[0] < length - 1.e-6)
          face->set_boundary_id(1);

        if (center[0] < 0. + 1.e-06)
          face->set_boundary_id(2);
      }
    }

    /*
     * Refine four times and round off corner with radius 0.0125:
     */

    triangulation.refine_global(4);

    Point<dim> point(step_position + 0.0125, step_height - 0.0125);
    triangulation.set_manifold(1, SphericalManifold<dim>(point));

    for (auto cell : triangulation.active_cell_iterators())
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        double distance =
            (cell->vertex(v) - Point<dim>(step_position, step_height)).norm();
        if (distance < 1.e-6) {
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            const auto face = cell->face(f);
            if(face->at_boundary())
              face->set_manifold_id(1);
            cell->set_manifold_id(1); // temporarily for second loop
          }
        }
      }

    for (auto cell : triangulation.active_cell_iterators()) {
      if (cell->manifold_id() != 1)
        continue;

      cell->set_manifold_id(0); // reset manifold id again

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        auto &vertex = cell->vertex(v);

        if (std::abs(vertex[0] - step_position) < 1.e-6 &&
            vertex[1] > step_height - 1.e-6)
          vertex[0] = step_position + 0.0125 * (1 - std::sqrt(1. / 2.));

        if (std::abs(vertex[1] - step_height) < 1.e-6 &&
            vertex[0] < step_position + 0.005)
          vertex[1] = step_height - 0.0125 * (1 - std::sqrt(1. / 2.));
      }
    }
  }


  /**
   * Create the 2D cylinder configuration
   *
   * Even though this function has a template parameter @p dim, it is only
   * implemented for dimension 2.
   */

  template <int dim>
  void create_coarse_grid_cylinder(
      dealii::parallel::distributed::Triangulation<dim> &,
      const double /*length*/,
      const double /*height*/,
      const double /*step_position*/,
      const double /*step_height*/)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }


  template <>
  void create_coarse_grid_cylinder<2>(
      dealii::parallel::distributed::Triangulation<2> &triangulation,
      const double length,
      const double height,
      const double disc_position,
      const double disc_diameter)
  {
    constexpr int dim = 2;

    using namespace dealii;

    Triangulation<dim> tria1, tria2, tria3, tria4;

    GridGenerator::hyper_cube_with_cylindrical_hole(
        tria1, disc_diameter / 2., disc_diameter, 0.5, 1, false);

    GridGenerator::subdivided_hyper_rectangle(
        tria2,
        {2, 1},
        Point<2>(-disc_diameter, disc_diameter),
        Point<2>(disc_diameter, height / 2.));

    GridGenerator::subdivided_hyper_rectangle(
        tria3,
        {2, 1},
        Point<2>(-disc_diameter, -disc_diameter),
        Point<2>(disc_diameter, -height / 2.));

    GridGenerator::subdivided_hyper_rectangle(
        tria4,
        {6, 4},
        Point<2>(disc_diameter, -height / 2.),
        Point<2>(length - disc_position, height / 2.));

    GridGenerator::merge_triangulations(
        {&tria1, &tria2, &tria3, &tria4}, triangulation, 1.e-12, true);

    /* Restore polar manifold for disc: */

    triangulation.set_manifold(0, PolarManifold<2>(Point<2>()));

    /* Fix up position of left boundary: */

    for (auto cell : triangulation.active_cell_iterators())
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        auto &vertex = cell->vertex(v);
        if (vertex[0] <= -disc_diameter + 1.e-6)
          vertex[0] = -disc_position;
      }

    /*
     * Set boundary ids:
     */

    for (auto cell : triangulation.active_cell_iterators()) {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if (!face->at_boundary())
          continue;

        /*
         * We want slip boundary conditions (i.e. indicator 1) at top and
         * bottom of the channel, as well as on the obstacle. On the left
         * side we set inflow conditions (indicator 2) and on the right
         * side we set indicator 0, i.e. do nothing.
         */

        const auto center = face->center();

        if (center[0] > length - disc_position - 1.e-6)
          continue;

        if (center[0] < -disc_position + 1.e-6) {
          face->set_boundary_id(2);
          continue;
        }

        // the rest:
        face->set_boundary_id(1);
      }
    }

  }


  /**
   * Create a 2D wall configuration: A rectangular domain with given length
   * and height.
   *
   */

  template <int dim>
  void
  create_coarse_grid_wall(dealii::parallel::distributed::Triangulation<dim> &,
                          const double /*length*/,
                          const double /*height*/,
                          const double /*wall_position*/)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }


  template <>
  void create_coarse_grid_wall<2>(
      dealii::parallel::distributed::Triangulation<2> &triangulation,
      const double length,
      const double height,
      const double wall_position)
  {
    using namespace dealii;

    Triangulation<2> tria1;

    GridGenerator::subdivided_hyper_rectangle(
        tria1, {18, 6}, Point<2>(wall_position, 0), Point<2>(length, height));

    Triangulation<2> tria2;

    GridGenerator::subdivided_hyper_rectangle(
        tria2, {1, 6}, Point<2>(0., 0.), Point<2>(wall_position, height));

    GridGenerator::merge_triangulations(tria1, tria2, triangulation);

    /*
     * Set boundary ids:
     */

    for (auto cell : triangulation.active_cell_iterators()) {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if (!face->at_boundary())
          continue;

        /*
         * We want slip boundary conditions (i.e. indicator 1) at the
         * bottom starting at position wall_position. We do nothing on the
         * right boundary and enforce inflow conditions elsewhere
         */

        const auto center = face->center();

        if (center[0] > wall_position && center[1] < 1.e-6)
          face->set_boundary_id(1);
        else if (center[0] > length - 1.e-6)
          continue;
        else
          // the rest:
          face->set_boundary_id(2);
      }
    }

  }

} // namespace grendel

#endif /* GEOMETRY_HELPER_H */
