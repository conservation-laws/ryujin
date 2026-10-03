#pragma once

#include <geometries/cut_cell_primitives.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

/*
 * Common helpers for the cut_cell_primitives_* tests: align a mesh of
 * 12 x 8 unit cells with a given height function, create the
 * triangulation with CutCellDecomposition and print a summary over the
 * locally owned cells, reduced to rank 0: the number of cells per
 * reference cell type, the number of boundary faces per boundary id (1
 * left, 2 right, 3 bottom, 4 top, 7 profile), the smallest cell measure,
 * and the vertices of all triangles.
 */

namespace
{
  constexpr int dim = 2;
  using dealii::Point;

  /* A linear ramp from 0 to 1 over a width around s = 0: */
  double ramp(const double s, const double width)
  {
    return std::clamp(s / width + 0.5, 0., 1.);
  }


  template <typename Temporary, typename Target>
  void run_cases()
  {
    const auto comm = MPI_COMM_WORLD;
    const bool root = dealii::Utilities::MPI::this_mpi_process(comm) == 0;

    ryujin::MeshAlignment<dim> alignment("mesh alignment");
    ryujin::CutCellDecomposition<dim> decomposition;
    std::stringstream parameters;
    dealii::ParameterAcceptor::initialize(parameters);

    const auto run = [&](const std::string &name, const auto &height) {
      const auto create = [&](auto *type) {
        using Triangulation = std::remove_pointer_t<decltype(type)>;
        if constexpr (std::is_same_v<Triangulation, dealii::Triangulation<dim>>)
          return std::make_unique<Triangulation>();
        else
          return std::make_unique<Triangulation>(comm);
      };
      const auto temporary = create(static_cast<Temporary *>(nullptr));
      const auto triangulation = create(static_cast<Target *>(nullptr));

      dealii::GridGenerator::subdivided_hyper_rectangle(
          *temporary, {12, 8}, Point<dim>(0., 0.), Point<dim>(12., 8.), true);
      for (const auto &cell : temporary->active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            face->set_boundary_id(face->boundary_id() + 1);

      alignment.align_with_elevation_profile(*temporary, height);
      decomposition.create_triangulation(*triangulation, *temporary, height, 7);

      std::map<std::string, unsigned int> cells;
      std::map<unsigned int, unsigned int> boundary_faces;
      double min_measure = std::numeric_limits<double>::max();
      std::vector<std::pair<Point<dim>, std::string>> triangles;

      for (const auto &cell : triangulation->active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;
        ++cells[cell->reference_cell().to_string()];
        min_measure = std::min(min_measure, cell->measure());
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            ++boundary_faces[face->boundary_id()];
        if (cell->n_vertices() == 3) {
          std::ostringstream vertices;
          vertices << std::fixed << std::setprecision(3);
          for (unsigned int i = 0; i < 3; ++i)
            vertices << (i == 0 ? "" : " ") << cell->vertex(i);
          triangles.emplace_back(cell->center(), vertices.str());
        }
      }

      min_measure = dealii::Utilities::MPI::min(min_measure, comm);
      const auto all_cells = dealii::Utilities::MPI::gather(comm, cells, 0);
      const auto all_boundary_faces =
          dealii::Utilities::MPI::gather(comm, boundary_faces, 0);
      const auto all_triangles =
          dealii::Utilities::MPI::gather(comm, triangles, 0);

      if (!root)
        return;

      cells.clear();
      boundary_faces.clear();
      triangles.clear();
      for (unsigned int r = 0; r < all_cells.size(); ++r) {
        for (const auto &[type, n] : all_cells[r])
          cells[type] += n;
        for (const auto &[id, n] : all_boundary_faces[r])
          boundary_faces[id] += n;
        triangles.insert(
            triangles.end(), all_triangles[r].begin(), all_triangles[r].end());
      }
      std::sort(triangles.begin(), triangles.end(), [](auto &a, auto &b) {
        return std::make_pair(a.first[0], a.first[1]) <
               std::make_pair(b.first[0], b.first[1]);
      });

      std::cout << "\n== " << name << " ==\n  cells:";
      for (const auto &[type, n] : cells)
        std::cout << " " << type << " " << n;
      std::cout << "\n  boundary faces:";
      for (const auto &[id, n] : boundary_faces)
        std::cout << " id " << id << ": " << n;
      std::cout << "\n  min measure: " << std::setprecision(4) << min_measure
                << "\n";
      for (const auto &[center, vertices] : triangles)
        std::cout << "    " << vertices << "\n";
    };

    run("flat at level 3", [](const Point<dim> &) { return 3.; });
    run("flat at level 3.5", [](const Point<dim> &) { return 3.5; });
    run("slope 0.25", [](const Point<dim> &p) { return 1. + 0.25 * p[0]; });

    for (const double position : {0.15, 5.0, 5.5})
      for (const double h : {0.5, 1.0, 1.5}) {
        std::stringstream name;
        name << "cliff at x = " << position << " of height " << h;
        run(name.str(), [=](const Point<dim> &p) {
          return 2. + h * ramp(p[0] - position, 0.3);
        });
      }
  }
} // namespace
