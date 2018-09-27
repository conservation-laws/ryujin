#ifndef GEOMETRY_HELPER_H
#define GEOMETRY_HELPER_H

namespace grendel
{

  template <int dim, typename T>
  void create_coarse_grid_shard(T &triangulation,
                                const double length_,
                                const double height_,
                                const double object_height_)
  {
    using namespace dealii;

    const double baserect = length_;
    const double heighrect = height_;
    const double triangheight = object_height_;
    const double triangside = object_height_ / std::cos(M_PI / 6.0);
    const double tiptriang = object_height_;

    Triangulation<dim> tria1; // left rectangle
    Triangulation<dim> tria2; // lower irregular quadrilateral
    Triangulation<dim> tria3; // upper irregular quadrilateral
    Triangulation<dim> tria4; // right lowest regular quadrilateral
    Triangulation<dim> tria5; // right middle regular quadrilateral
    Triangulation<dim> tria6; // upper regular quadrilateral

    Triangulation<dim> merge1;
    Triangulation<dim> merge2;

    {
      std::vector<unsigned int> repetitions(2);
      repetitions[0] = 1;
      repetitions[1] = 2;
      GridGenerator::subdivided_hyper_rectangle(tria1,
                                                repetitions,
                                                Point<2>(0.0, 0.0),
                                                Point<2>(tiptriang, heighrect));
    }

    {
      std::vector<unsigned int> repetitions(2);
      repetitions[0] = 1;
      repetitions[1] = 1;
      GridGenerator::subdivided_hyper_rectangle(
          tria4,
          repetitions,
          Point<2>(tiptriang + triangheight, 0.0),
          Point<2>(baserect, 0.5 * (heighrect - triangside)));
    }

    {
      std::vector<unsigned int> repetitions(2);
      repetitions[0] = 1;
      repetitions[1] = 1;
      GridGenerator::subdivided_hyper_rectangle(
          tria5,
          repetitions,
          Point<2>(tiptriang + triangheight, 0.5 * (heighrect - triangside)),
          Point<2>(baserect, 0.5 * (heighrect + triangside)));
    }

    {
      std::vector<unsigned int> repetitions(2);
      repetitions[0] = 1;
      repetitions[1] = 1;
      GridGenerator::subdivided_hyper_rectangle(
          tria6,
          repetitions,
          Point<2>(tiptriang + triangheight, 0.5 * (heighrect + triangside)),
          Point<2>(baserect, heighrect));
    }

    {
      static const Point<2> vertices1[] = {
          Point<dim>(0.0, 0.0),
          Point<dim>(triangheight, 0.0),
          Point<dim>(0.0, 0.5 * heighrect),
          Point<dim>(triangheight, 0.5 * (heighrect - triangside))};

      const unsigned int n_vertices = sizeof(vertices1) / sizeof(vertices1[0]);
      const std::vector<Point<dim>> vertices(&vertices1[0],
                                             &vertices1[n_vertices]);

      static const int cell_vertices[][GeometryInfo<dim>::vertices_per_cell] = {
          {0, 1, 2, 3}};
      const unsigned int n_cells =
          sizeof(cell_vertices) / sizeof(cell_vertices[0]);

      std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
      for (unsigned int i = 0; i < n_cells; ++i) {
        for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j)
          cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
      }

      tria2.create_triangulation(vertices, cells, SubCellData());

      Tensor<1, dim> moveright;
      moveright[0] = tiptriang;
      moveright[1] = 0.0;
      GridTools::shift(moveright, tria2);
    }

    {
      static const Point<2> vertices1[] = {
          Point<dim>(0.0, 0.0),
          Point<dim>(triangheight, 0.0),
          Point<dim>(0.0, -0.5 * heighrect),
          Point<dim>(triangheight, -0.5 * (heighrect - triangside))};

      const unsigned int n_vertices = sizeof(vertices1) / sizeof(vertices1[0]);
      const std::vector<Point<dim>> vertices(&vertices1[0],
                                             &vertices1[n_vertices]);

      static const int cell_vertices[][GeometryInfo<dim>::vertices_per_cell] = {
          {2, 3, 0, 1}};
      const unsigned int n_cells =
          sizeof(cell_vertices) / sizeof(cell_vertices[0]);

      std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
      for (unsigned int i = 0; i < n_cells; ++i) {
        for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j)
          cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
      }

      tria3.create_triangulation(vertices, cells, SubCellData());

      Tensor<1, dim> moveright;
      moveright[0] = tiptriang;
      moveright[1] = heighrect;
      GridTools::shift(moveright, tria3);
    }

    GridGenerator::merge_triangulations(tria1, tria3, merge1);
    GridGenerator::merge_triangulations(merge1, tria2, merge2);
    merge1.clear();
    GridGenerator::merge_triangulations(merge2, tria4, merge1);
    merge2.clear();
    GridGenerator::merge_triangulations(merge1, tria5, merge2);
    merge1.clear();
    GridGenerator::merge_triangulations(merge2, tria6, triangulation);
  }

} // namespace grendel

#endif /* GEOMETRY_HELPER_H */
