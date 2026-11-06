//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/geometries/geometry_library.h>

#include <boost/random/detail/polynomial.hpp>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe.h>
#if DEAL_II_VERSION_GTE(9, 7, 0)
#include <deal.II/fe/mapping_p1.h>
#endif
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <fstream>
#include <random>

namespace ryujin
{
  using namespace dealii;

  template <int structdim, int dim>
  Discretization<structdim, dim>::Discretization(
      const MPIEnsemble &mpi_ensemble, const std::string &subsection)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
  {
    /* Options: */

    ansatz_ = Ansatz::cg_q1;
    add_parameter("finite element ansatz",
                  ansatz_,
                  "The finite element ansatz used for discretization. Valid "
                  "choices are cG Q1, cG Q2, cG Q3.");

    mesh_type_ = (structdim == 1 ? MeshType::parallel_shared
                                 : MeshType::parallel_distributed);
    add_parameter("mesh type",
                  mesh_type_,
                  "The triangulation class used. Valid choices are \"serial\", "
                  "\"parallel shared\", \"parallel distributed\", \"parallel "
                  "fullydistributed\".");

    geometry_ = "rectangular domain";
    geometry_list_ = GeometryLibrary<structdim, dim>::create(subsection);

    add_parameter("geometry",
                  geometry_,
                  "Name of the geometry used to create the mesh. Valid names "
                  "are given by any of the subsections defined below.",
                  this->prm,
                  GeometryLibrary<structdim, dim>::selection(geometry_list_));

    refinement_ = 5;
    add_parameter("mesh refinement",
                  refinement_,
                  "number of refinement of global refinement steps");

    local_refinement_ = 0;
    add_parameter("mesh local refinement",
                  local_refinement_,
                  "number of local refinement steps performed after global "
                  "refinement on the region given by \"mesh local refinement "
                  "region\"");

    local_refinement_region_ = "1";
    add_parameter("mesh local refinement region",
                  local_refinement_region_,
                  "expression in x, y, z: cells with a positive value at the "
                  "cell center are locally refined");

    mesh_writeout_ = true;
    add_parameter("mesh writeout",
                  mesh_writeout_,
                  "Write out shared coarse mesh to a GMSH *.msh file.");

    mesh_distortion_ = 0.;
    add_parameter(
        "mesh distortion", mesh_distortion_, "Strength of mesh distortion");
  }


  template <int structdim, int dim>
  void Discretization<structdim, dim>::prepare(const std::string &base_name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Discretization<structdim, dim>::prepare()" << std::endl;
#endif

    AssertThrow(
        mesh_type_ != MeshType::parallel_fullydistributed || refinement_ == 0,
        ExcMessage("The fully distributed mesh type does not support global "
                   "refinement. The geometry must create a properly refined "
                   "mesh."));

    /* Select geometry: */

    selected_geometry_ =
        &GeometryLibrary<structdim, dim>::select(geometry_list_, geometry_);

    /*
     * Set up Triangulation object. DoFHandler objects attached to a
     * previous triangulation have to be released first:
     */

    dof_handler_cg_.reset();
    dof_handler_dg_.reset();

    const auto smoothing =
        dealii::Triangulation<structdim,
                              dim>::limit_level_difference_at_vertices;

    switch (mesh_type_) {
    case MeshType::parallel_fullydistributed: {
      const auto settings = dealii::TriangulationDescription::Settings::
          construct_multigrid_hierarchy;
      auto triangulation = std::make_unique<
          dealii::parallel::fullydistributed::Triangulation<structdim, dim>>(
          mpi_ensemble_.ensemble_communicator());
      triangulation->set_mesh_smoothing(smoothing);
      triangulation->set_partitioner(
          [](dealii::Triangulation<structdim, dim> &tria,
             const unsigned int n_partitions) {
            GridTools::partition_triangulation_zorder(n_partitions, tria);
          },
          settings);
      triangulation_ = std::move(triangulation);
    } break;

    case MeshType::parallel_distributed: {
      const auto settings =
          dealii::parallel::distributed::Triangulation<structdim, dim>::
              Settings::construct_multigrid_hierarchy;
      triangulation_ = std::make_unique<
          dealii::parallel::distributed::Triangulation<structdim, dim>>(
          mpi_ensemble_.ensemble_communicator(), smoothing, settings);
    } break;

    case MeshType::parallel_shared: {
      const auto settings =
          static_cast<typename dealii::parallel::shared::
                          Triangulation<structdim, dim>::Settings>(
              dealii::parallel::shared::Triangulation<structdim,
                                                      dim>::partition_auto |
              dealii::parallel::shared::Triangulation<structdim, dim>::
                  construct_multigrid_hierarchy);
      /* Beware of the boolean: */
      triangulation_ = std::make_unique<
          dealii::parallel::shared::Triangulation<structdim, dim>>(
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

      triangulation_ =
          std::make_unique<dealii::Triangulation<structdim, dim>>(smoothing);

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

    if (local_refinement_ > 0) {
      const FunctionParser<dim> region(local_refinement_region_);
      for (unsigned int step = 0; step < local_refinement_; ++step) {
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned() && region.value(cell->center()) > 0.)
            cell->set_refine_flag();
        triangulation.execute_coarsening_and_refinement();
      }
    }

    if (std::abs(mesh_distortion_) > 1.0e-10) {
      /*
       * A parallel::shared::Triangulation is distorted independently on
       * every rank, so all ranks have to use the same seed:
       */
      const auto seed = Utilities::MPI::broadcast(
          mpi_ensemble_.ensemble_communicator(), std::random_device()());
      GridTools::distort_random(mesh_distortion_, triangulation, false, seed);
    }

#ifdef DEBUG
    /*
     * Without a partner cell for a periodic face we would silently end up
     * with an open boundary. This checks for a bug in deal.II, see
     * https://github.com/dealii/dealii/pull/20359
     */

    using cell_iterator =
        typename dealii::Triangulation<structdim, dim>::cell_iterator;
    const auto &periodic_faces = triangulation.get_periodic_face_map();

    for (const auto &cell : triangulation.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      for (const auto f : cell->face_indices()) {
        const auto face = cell->face(f);
        if (!face->at_boundary() || face->boundary_id() != Boundary::periodic)
          continue;

        const auto it = periodic_faces.find({cell_iterator(cell), f});
        Assert(it != periodic_faces.end(),
               ExcMessage("The partner cell of a periodic face is not "
                          "available on this MPI rank."));

        const auto &partner = it->second.first.first;
        Assert(!partner->is_active() || !partner->is_artificial(),
               ExcMessage("The partner cell of a periodic face is an "
                          "artificial cell on this MPI rank."));
      }
    }
#endif

    const auto fe_degree = polynomial_degree();
    const auto mapping_degree = fe_degree;
    const auto quadrature_degree = fe_degree + 1;

    /*
     * First, let the selected geometry populate our hp::*Collection
     * objects. If the method returns standard_quarilaterls, or
     * standard_simplices, however, we need to do the setup ourselves:
     */

    transformation_ = selected_geometry_->transformation();

    collection_type_ =
        selected_geometry_->populate_hp_collections(fe_degree, collection_);

    switch (collection_type_) {
    case Geometry<structdim, dim>::HP_Collection::populated_by_geometry: {
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

    case Geometry<structdim, dim>::HP_Collection::standard_quadrilaterals: {
      /*
       * Populate all collections with appropriate objects for the cG Qk, dG
       * Qk finite element on purely quadrilateral, or hexahedral meshes:
       */

      collection_.finite_element_cg =
          std::make_unique<hp::FECollection<structdim, dim>>(
              FE_Q<structdim, dim>(fe_degree));
      collection_.finite_element_dg =
          std::make_unique<hp::FECollection<structdim, dim>>(
              FE_DGQ<structdim, dim>(fe_degree));

      /*
       * If the geometry describes the mesh via a forward transformation of
       * the undeformed triangulation we use a MappingQCache that is filled
       * in update_mapping():
       */
      if (transformation_) {
        create_mapping_cache();
      } else {
        collection_.mapping =
            std::make_unique<dealii::hp::MappingCollection<structdim, dim>>(
                MappingQ<structdim, dim>(mapping_degree));
      }

      collection_.quadrature = std::make_unique<hp::QCollection<structdim>>(
          QGauss<structdim>(quadrature_degree));
      collection_.quadrature_high_order =
          std::make_unique<hp::QCollection<structdim>>(
              QGauss<structdim>(quadrature_degree + 1));
      collection_.nodal_quadrature =
          std::make_unique<hp::QCollection<structdim>>(
              QGaussLobatto<structdim>(quadrature_degree));
      collection_.quadrature_1d =
          std::make_unique<hp::QCollection<1>>(QGauss<1>(quadrature_degree));
      collection_.nodal_quadrature_1d = std::make_unique<hp::QCollection<1>>(
          QGaussLobatto<1>(quadrature_degree));
      using QCF = hp::QCollection<structdim - 1>;
      collection_.face_quadrature = std::make_unique<std::vector<QCF>>(
          1, QCF(QGauss<structdim - 1>(quadrature_degree)));
      collection_.face_nodal_quadrature = std::make_unique<std::vector<QCF>>(
          1, QCF(QGaussLobatto<structdim - 1>(quadrature_degree)));
    } break;

    case Geometry<structdim, dim>::HP_Collection::standard_simplices: {
      /*
       * Populate all collections with appropriate objects for the cG Pk, dG
       * Pk finite element on purely quadrilateral, or hexahedral meshes:
       */

      collection_.finite_element_cg =
          std::make_unique<hp::FECollection<structdim, dim>>(
              FE_SimplexP<structdim, dim>(fe_degree));
      collection_.finite_element_dg =
          std::make_unique<hp::FECollection<structdim, dim>>(
              FE_SimplexDGP<structdim, dim>(fe_degree));

      if (mapping_degree == 1) {
#if DEAL_II_VERSION_GTE(9, 7, 0)
        collection_.mapping =
            std::make_unique<hp::MappingCollection<structdim, dim>>(
                MappingP1<structdim, dim>());
#else
        collection_.mapping =
            std::make_unique<hp::MappingCollection<structdim, dim>>(
                MappingFE<structdim, dim>(
                    FE_SimplexP<structdim, dim>(fe_degree)));
#endif
      } else {
        collection_.mapping =
            std::make_unique<hp::MappingCollection<structdim, dim>>(
                MappingFE<structdim, dim>(
                    FE_SimplexP<structdim, dim>(fe_degree)));
      }

      collection_.quadrature = std::make_unique<hp::QCollection<structdim>>(
          QGaussSimplex<structdim>(quadrature_degree));
      collection_.quadrature_high_order =
          std::make_unique<hp::QCollection<structdim>>(
              QGaussSimplex<structdim>(quadrature_degree + 1));
#if DEAL_II_VERSION_GTE(9, 7, 0)
      collection_.nodal_quadrature =
          std::make_unique<hp::QCollection<structdim>>(
              FETools::compute_nodal_quadrature(
                  FE_SimplexP<structdim>(quadrature_degree)));
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
      using QCF = hp::QCollection<structdim - 1>;
      collection_.face_quadrature = std::make_unique<std::vector<QCF>>(
          1, QCF(QGaussSimplex<structdim - 1>(quadrature_degree)));
      if constexpr (structdim == 1) {
        collection_.face_nodal_quadrature = std::make_unique<std::vector<QCF>>(
            1, QCF(QGaussLobatto<structdim - 1>(quadrature_degree)));
      } else {
#if DEAL_II_VERSION_GTE(9, 7, 0)
        const auto quadrature = FETools::compute_nodal_quadrature(
            FE_SimplexP<structdim - 1>(fe_degree));
        collection_.face_nodal_quadrature =
            std::make_unique<std::vector<QCF>>(1, QCF(quadrature));
#endif
      }
    } break;
    default:
      __builtin_trap();
    }

    distribute_dofs();
    update_mapping();
  }


  template <int structdim, int dim>
  void Discretization<structdim, dim>::update_mapping()
  {
    if (!mapping_cache_) {
      if (!transformation_)
        return;

      /* The geometry or a MeshAdaptor has set a transformation: */
      using HP_Collection = typename Geometry<structdim, dim>::HP_Collection;
      AssertThrow(
          collection_type_ == HP_Collection::standard_quadrilaterals,
          ExcMessage("Discretization: A transformation (set by the geometry "
                     "or a mesh adaptor) is only supported for the standard "
                     "Q_k setup on quadrilateral or hexahedral meshes. It "
                     "cannot be used with simplex meshes or with a geometry "
                     "that populates the hp collections itself."));
      create_mapping_cache();
    }

    const MappingQ<structdim, dim> mapping(mapping_cache_->get_degree());
    if (transformation_)
      mapping_cache_->initialize(mapping,
                                 *triangulation_,
                                 transformation_,
                                 /*function_describes_relative_displacement*/
                                 false);
    else
      mapping_cache_->initialize(mapping, *triangulation_);
  }


  template <int structdim, int dim>
  void Discretization<structdim, dim>::distribute_dofs()
  {
    if (!dof_handler_cg_)
      dof_handler_cg_ =
          std::make_unique<dealii::DoFHandler<structdim, dim>>(*triangulation_);

    if (!dof_handler_dg_)
      dof_handler_dg_ =
          std::make_unique<dealii::DoFHandler<structdim, dim>>(*triangulation_);

    /*
     * Set active FE indices: This information depends on the selected
     * geometry. Therefore, let the selected geometry object handle the
     * setup. For a standard geometry that has only one reference element
     * the method simply does nothing.
     */

    selected_geometry_->update_dof_handler(*dof_handler_cg_);
    selected_geometry_->update_dof_handler(*dof_handler_dg_);

    dof_handler_cg_->distribute_dofs(finite_element_cg());
    dof_handler_dg_->distribute_dofs(finite_element_dg());
  }


  template <int structdim, int dim>
  void Discretization<structdim, dim>::create_mapping_cache()
  {
    mapping_cache_ =
        std::make_shared<MappingQCache<structdim, dim>>(polynomial_degree());

    auto mapping = std::make_unique<hp::MappingCollection<structdim, dim>>();
    hp::Collection<Mapping<structdim, dim>> &base = *mapping;
    base.push_back(mapping_cache_);
    collection_.mapping = std::move(mapping);
  }

} // namespace ryujin
