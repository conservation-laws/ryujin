//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <boost/random/detail/polynomial.hpp>
#include <compile_time_options.h>

#include "discretization.h"
#include "geometries/geometry_library.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe.h>
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
                  "The finite element ansatz used for discretization. Valid "
                  "choices are cG Q1, cG Q2, cG Q3.");

    mesh_type_ =
        (dim == 1 ? MeshType::parallel_shared : MeshType::parallel_distributed);
    add_parameter("mesh type",
                  mesh_type_,
                  "The triangulation class used. Valid choices are \"serial\", "
                  "\"parallel shared\", \"parallel distributed\", \"parallel "
                  "fullydistributed\".");

    if constexpr (dim == 1) {
      geometry_ = "rectangular domain";
    } else {
      geometry_ = "cylinder";
    }
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

    /* Select geometry: */

    {
      bool initialized = false;
      for (auto &it : geometry_list_)
        if (it->name() == geometry_) {
          selected_geometry_ = it;
          initialized = true;
          break;
        }

      AssertThrow(
          initialized,
          ExcMessage("Could not find a geometry description with name \"" +
                     geometry_ + "\""));
    }

    /* Set up Triangulation object: */

    const auto smoothing =
        dealii::Triangulation<dim>::limit_level_difference_at_vertices;

    switch (mesh_type_) {
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

    case MeshType::serial: {
      AssertThrow(
          mpi_ensemble_.n_ensemble_ranks() == 1,
          ExcMessage(
              "The serial triangulation can only be used for serial "
              "computations. If you want to run simulations with more than one "
              "rank per ensemble, then please set \"mesh type\" to one of the "
              "parallel triangulations supported by deal.II"));

      triangulation_ = std::make_unique<dealii::Triangulation<dim>>(smoothing);

    } break;

    default:
      __builtin_trap();
    }

    /* Create and distribute mesh: */

    auto &triangulation = *triangulation_;
    selected_geometry_->create_coarse_triangulation(triangulation);

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

    /*
     * First, let the selected geometry populate our hp::*Collection
     * objects. If the method returns standard_quarilaterls, or
     * standard_simplices, however, we need to do the setup ourselves:
     */

    const auto collection_type =
        selected_geometry_->populate_hp_collections(fe_degree, collection_);

    switch (collection_type) {
    case Geometry<dim>::HP_Collection::populated_by_geometry: {
      /*
       * The geometry already populated the hp::*Collections
       */

      Assert(collection_.mapping, dealii::ExcInternalError());
      Assert(collection_.finite_element_cg, dealii::ExcInternalError());
      Assert(collection_.finite_element_dg, dealii::ExcInternalError());
      Assert(collection_.quadrature, dealii::ExcInternalError());
      Assert(collection_.quadrature_high_order, dealii::ExcInternalError());
      Assert(collection_.nodal_quadrature, dealii::ExcInternalError());
      Assert(collection_.quadrature_1d, dealii::ExcInternalError());
      Assert(collection_.nodal_quadrature_1d, dealii::ExcInternalError());
      Assert(collection_.face_quadrature, dealii::ExcInternalError());
      Assert(collection_.face_nodal_quadrature, dealii::ExcInternalError());
    } break;

    case Geometry<dim>::HP_Collection::standard_quadrilaterals: {
      /*
       * Populate all collections with appropriate objects for the cG Qk, dG
       * Qk finite element on purely quadrilateral, or hexahedral meshes:
       */

      collection_.finite_element_cg =
          std::make_unique<hp::FECollection<dim>>(FE_Q<dim>(fe_degree));
      collection_.finite_element_dg =
          std::make_unique<hp::FECollection<dim>>(FE_DGQ<dim>(fe_degree));

      collection_.mapping =
          std::make_unique<dealii::hp::MappingCollection<dim>>(
              MappingQ<dim>(mapping_degree));

      collection_.quadrature = std::make_unique<hp::QCollection<dim>>(
          QGauss<dim>(quadrature_degree));
      collection_.quadrature_high_order =
          std::make_unique<hp::QCollection<dim>>(
              QGauss<dim>(quadrature_degree + 1));
      collection_.nodal_quadrature = std::make_unique<hp::QCollection<dim>>(
          QGaussLobatto<dim>(quadrature_degree));
      collection_.quadrature_1d =
          std::make_unique<hp::QCollection<1>>(QGauss<1>(quadrature_degree));
      collection_.nodal_quadrature_1d = std::make_unique<hp::QCollection<1>>(
          QGaussLobatto<1>(quadrature_degree));
      collection_.face_quadrature = std::make_unique<hp::QCollection<dim - 1>>(
          QGauss<dim - 1>(quadrature_degree));
      collection_.face_nodal_quadrature =
          std::make_unique<hp::QCollection<dim - 1>>(
              QGaussLobatto<dim - 1>(quadrature_degree));
    } break;

    case Geometry<dim>::HP_Collection::standard_simplices: {
      /*
       * Populate all collections with appropriate objects for the cG Pk, dG
       * Pk finite element on purely quadrilateral, or hexahedral meshes:
       */

      collection_.finite_element_cg =
          std::make_unique<hp::FECollection<dim>>(FE_SimplexP<dim>(fe_degree));
      collection_.finite_element_dg = std::make_unique<hp::FECollection<dim>>(
          FE_SimplexDGP<dim>(fe_degree));

      collection_.mapping =
          std::make_unique<dealii::hp::MappingCollection<dim>>(
              MappingFE<dim>(FE_SimplexP<dim>(mapping_degree)));

      collection_.quadrature = std::make_unique<hp::QCollection<dim>>(
          QGaussSimplex<dim>(quadrature_degree));
      collection_.quadrature_high_order =
          std::make_unique<hp::QCollection<dim>>(
              QGaussSimplex<dim>(quadrature_degree + 1));
#if DEAL_II_VERSION_GTE(9, 7, 0)
      collection_.nodal_quadrature = std::make_unique<hp::QCollection<dim>>(
          FETools::compute_nodal_quadrature(
              FE_SimplexP<dim>(quadrature_degree)));
#else
      AssertThrow(false,
                  dealii::ExcMessage("Discretization: Simplex support requires "
                                     "deal.II version 9.7.0 or newer"));

#endif
      collection_.quadrature_1d = std::make_unique<hp::QCollection<1>>(
          QGaussSimplex<1>(quadrature_degree));
#if DEAL_II_VERSION_GTE(9, 7, 0)
      collection_.nodal_quadrature_1d = std::make_unique<hp::QCollection<1>>(
          QGaussLobatto<1>(quadrature_degree));
#endif
      collection_.face_quadrature = std::make_unique<hp::QCollection<dim - 1>>(
          QGaussSimplex<dim - 1>(quadrature_degree));
      if constexpr (dim == 1) {
        collection_.face_nodal_quadrature =
            std::make_unique<hp::QCollection<dim - 1>>(
                QGaussLobatto<dim - 1>(quadrature_degree));
      } else {
#if DEAL_II_VERSION_GTE(9, 7, 0)
        collection_.face_nodal_quadrature =
            std::make_unique<hp::QCollection<dim - 1>>(
                FETools::compute_nodal_quadrature(
                    FE_SimplexP<dim - 1>(quadrature_degree)));
#endif
      }

      return;
    } break;
    default:
      __builtin_trap();
    }
  }

} /* namespace ryujin */
