#ifndef GEOMETRY_HELPER_H
#define GEOMETRY_HELPER_H

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

namespace grendel
{

  template <int dim>
  void create_coarse_grid_triangle(
      dealii::parallel::distributed::Triangulation<dim> &,
      const double,
      const double,
      const double)
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
  }

} // namespace grendel

#endif /* GEOMETRY_HELPER_H */
