//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "geometries/geometry_library.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_out.h>

#include <random>

namespace ryujin
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const MPIEnsemble &mpi_ensemble,
                                      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
  {
    /* Options: */

    ansatz_ = Ansatz::cg_q1;
    add_parameter("finite element ansatz",
                  ansatz_,
                  "The finite element ansatz used for discretization. valid "
                  "choices are cG Q1, cG Q2, cG Q3.");

    geometry_ = "cylinder";
    add_parameter("geometry",
                  geometry_,
                  "Name of the geometry used to create the mesh. Valid names "
                  "are given by any of the subsections defined below.");

    refinement_ = 5;
    add_parameter("mesh refinement",
                  refinement_,
                  "number of refinement of global refinement steps");

    mesh_writeout_ = true;
    add_parameter("mesh writeout",
                  mesh_writeout_,
                  "Write out shared coarse mesh to a GMSH *.msh file.");

    mesh_distortion_ = 0.;
    add_parameter(
        "mesh distortion", mesh_distortion_, "Strength of mesh distortion");

    Geometries::populate_geometry_list<dim>(geometry_list_, subsection);
  }


  template <int dim>
  void Discretization<dim>::prepare(const std::string &base_name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Discretization<dim>::prepare()" << std::endl;
#endif

    const auto smoothing =
        dealii::Triangulation<dim>::limit_level_difference_at_vertices;

    // FIXME: This information will ultimately be provided by the Geometry.
    const auto selection =
        (dim == 1 ? MeshType::parallel_shared : MeshType::parallel_distributed);

    switch (selection) {
    case MeshType::parallel_fullydistributed: {
      triangulation_ = std::make_unique<
          dealii::parallel::fullydistributed::Triangulation<dim>>(
          mpi_ensemble_.ensemble_communicator());
      triangulation_->set_mesh_smoothing(smoothing);
    } break;

    case MeshType::parallel_distributed: {
      const auto settings = dealii::parallel::distributed::Triangulation<
          dim>::Settings::construct_multigrid_hierarchy;
      triangulation_ =
          std::make_unique<dealii::parallel::distributed::Triangulation<dim>>(
              mpi_ensemble_.ensemble_communicator(), smoothing, settings);
    } break;

    case MeshType::parallel_shared: {
      const auto settings = static_cast<
          typename dealii::parallel::shared::Triangulation<dim>::Settings>(
          dealii::parallel::shared::Triangulation<dim>::partition_auto |
          dealii::parallel::shared::Triangulation<
              dim>::construct_multigrid_hierarchy);
      /* Beware of the boolean: */
      triangulation_ =
          std::make_unique<dealii::parallel::shared::Triangulation<dim>>(
              mpi_ensemble_.ensemble_communicator(),
              smoothing,
              /*artificial cells*/ true,
              settings);
    } break;

    default:
      __builtin_trap();
    }

    auto &triangulation = *triangulation_;

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

    if (mesh_writeout_ && dealii::Utilities::MPI::this_mpi_process(
                              mpi_ensemble_.ensemble_communicator()) == 0) {
#ifdef DEAL_II_GMSH_WITH_API
      GridOut grid_out;
      grid_out.write_msh(triangulation, base_name + "-coarse_grid.msh");
#else
      GridOut grid_out;
      GridOutFlags::Msh flags(/* write faces */ true, /* write lines */ true);
      grid_out.set_flags(flags);
      std::ofstream file(base_name + "-coarse_grid.msh");
      grid_out.write_msh(triangulation, file);
#endif
    }

    triangulation.refine_global(refinement_);

    if (std::abs(mesh_distortion_) > 1.0e-10)
      GridTools::distort_random(
          mesh_distortion_, triangulation, false, std::random_device()());

    const auto fe_degree = polynomial_degree();
    const auto mapping_degree = fe_degree;
    const auto quadrature_degree = fe_degree + 1;

    if (have_discontinuous_ansatz()) {
      finite_element_ =
          std::make_unique<hp::FECollection<dim>>(FE_DGQ<dim>(fe_degree));
      finite_element_cg_ =
          std::make_unique<hp::FECollection<dim>>(FE_Q<dim>(fe_degree));
    } else {
      finite_element_ =
          std::make_unique<hp::FECollection<dim>>(FE_Q<dim>(fe_degree));
      finite_element_cg_ =
          std::make_unique<hp::FECollection<dim>>(FE_Q<dim>(fe_degree));
    }

    mapping_ = std::make_unique<dealii::hp::MappingCollection<dim>>(
        MappingQ<dim>(mapping_degree));

    quadrature_ =
        std::make_unique<hp::QCollection<dim>>(QGauss<dim>(quadrature_degree));
    quadrature_high_order_ = std::make_unique<hp::QCollection<dim>>(
        QGauss<dim>(quadrature_degree + 1));
    nodal_quadrature_ = std::make_unique<hp::QCollection<dim>>(
        QGaussLobatto<dim>(quadrature_degree));

    quadrature_1d_ =
        std::make_unique<hp::QCollection<1>>(QGauss<1>(quadrature_degree));

    face_quadrature_ = std::make_unique<hp::QCollection<dim - 1>>(
        QGauss<dim - 1>(quadrature_degree));
    face_nodal_quadrature_ = std::make_unique<hp::QCollection<dim - 1>>(
        QGaussLobatto<dim - 1>(quadrature_degree));
  }

} /* namespace ryujin */
