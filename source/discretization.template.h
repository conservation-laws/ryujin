//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef DISCRETIZATION_TEMPLATE_H
#define DISCRETIZATION_TEMPLATE_H

#include <compile_time_options.h>

#include "discretization.h"
#include "grid_generator.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <fstream>

namespace ryujin
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const MPI_Comm &mpi_communicator,
                                      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , triangulation_(std::make_unique<Triangulation>(mpi_communicator_))
  {
    /* Options: */

    geometry_ = "cylinder";
    add_parameter("geometry",
                  geometry_,
                  "Name of the geometry used to create the mesh. Valid names "
                  "are given by any of the subsections defined below.");

    refinement_ = 5;
    add_parameter("mesh refinement",
                  refinement_,
                  "number of refinement of global refinement steps");

    mesh_distortion_ = 0.;
    add_parameter(
        "mesh distortion", mesh_distortion_, "Strength of mesh distortion");

    repartitioning_ = true;
    add_parameter("mesh repartitioning",
                  repartitioning_,
                  "try to equalize workload by repartitioning the mesh");

    geometry_list_.emplace(
        std::make_unique<Geometries::Cylinder<dim>>(subsection));
    geometry_list_.emplace(std::make_unique<Geometries::Step<dim>>(subsection));
    geometry_list_.emplace(std::make_unique<Geometries::Wall<dim>>(subsection));
    geometry_list_.emplace(
        std::make_unique<Geometries::ShockTube<dim>>(subsection));
    geometry_list_.emplace(
        std::make_unique<Geometries::Validation<dim>>(subsection));
    geometry_list_.emplace(
        std::make_unique<Geometries::Airfoil<dim>>(subsection));
  }


  template <int dim>
  void Discretization<dim>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Discretization<dim>::prepare()" << std::endl;
#endif

    auto &triangulation = *triangulation_;
    triangulation.clear();

    {
      bool initialized = false;
      for (auto &it : geometry_list_)
        if (it->name() == geometry_) {
          it->create_triangulation(triangulation);
          initialized = true;
          break;
        }

      AssertThrow(
          initialized,
          ExcMessage("Could not find a geometry description with name \"" +
                     geometry_ + "\""));
    }

    /* Handle periodic faces: */

    const auto bdy_ids = triangulation.get_boundary_ids();
    if constexpr (dim != 1) {
      if (std::find(bdy_ids.begin(), bdy_ids.end(), Boundary::periodic) !=
          bdy_ids.end()) {

#ifdef DEBUG_OUTPUT
        std::cout << "        collecting periodic faces" << std::endl;
#endif

        std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>>
            periodic_faces;

        for (int i = 0; i < dim; ++i)
          GridTools::collect_periodic_faces(triangulation,
                                            /*b_id */ Boundary::periodic,
                                            /*direction*/ i,
                                            periodic_faces);

        triangulation.add_periodicity(periodic_faces);
      }
    }

#ifdef USE_SIMD
    if (repartitioning_) {
      /*
       * Try to partition the mesh equilibrating the workload. The usual mesh
       * partitioning heuristic that tries to partition the mesh such that
       * every MPI rank has roughly the same number of locally owned degrees
       * of freedom does not work well in our case due to the fact that
       * boundary dofs are not SIMD parallelized. (In fact, every dof with
       * "non-standard connectivity" is not SIMD parallelized. Those are
       * however exceedingly rare (point irregularities in 2D, line
       * irregularities in 3D) and we simply ignore them.)
       *
       * For the mesh partitioning scheme we have to supply an additional
       * weight that gets added to the default weight of a cell which is
       * 1000. Asymptotically we have one boundary dof per boundary cell (in
       * any dimension). A rough benchmark reveals that the speedup due to
       * SIMD vectorization is typically less than VectorizedArray::size() /
       * 2. Boundary dofs are more expensive due to certain special treatment
       * (additional symmetrization of d_ij, boundary fixup) so it should be
       * safe to assume that the cost incurred is at least
       * VectorizedArray::size() / 2.
       */
      constexpr auto speedup = dealii::VectorizedArray<NUMBER>::size() / 2u;
      constexpr unsigned int weight = 1000u;

      triangulation.signals.cell_weight.connect(
          [](const auto &cell, const auto /*status*/) -> unsigned int {
            if (cell->at_boundary())
              return weight * (speedup == 0u ? 0u : speedup - 1u);
            else
              return 0u;
          });

      triangulation.repartition();
    }
#endif

    triangulation.refine_global(refinement_);

    if (std::abs(mesh_distortion_) > 1.0e-10)
      GridTools::distort_random(mesh_distortion_, triangulation);

    mapping_ = std::make_unique<MappingQ<dim>>(order_mapping);
    finite_element_ = std::make_unique<FE_Q<dim>>(order_finite_element);
    quadrature_ = std::make_unique<QGauss<dim>>(order_quadrature);
    quadrature_1d_ = std::make_unique<QGauss<1>>(order_quadrature);
  }

} /* namespace ryujin */

#endif /* DISCRETIZATION_TEMPLATE_H */
