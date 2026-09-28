//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "cut_cell_primitives.h"
#include "geometry_rectangular_domain.h"
#include "geotiff_reader.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

namespace ryujin
{
  namespace Geometries
  {
    /**
     * A modified rectangular domain, where the bottom boundary is
     * described by an elevation profile read from a GeoTIFF file. By
     * convention, the negative y-direction points to the bottom boundary.
     *
     * The rectangular mesh is aligned with the profile (see MeshAlignment)
     * and the cells above the profile form the new mesh, where cut cells
     * are replaced by triangles (see CutCellDecomposition).
     *
     * @note Only implemented in 2D.
     *
     * @ingroup Mesh
     */
    template <int dim>
    class GeoTIFFProfile : public RectangularDomain<dim>
    {
    public:
      GeoTIFFProfile(const std::string &subsection)
          : RectangularDomain<dim>("geotiff profile", subsection)
          , geotiff_reader_(subsection + "/geotiff profile/geotiff")
          , mesh_alignment_(subsection + "/geotiff profile/mesh alignment")
      {
        reference_y_coordinate_ = 0.;
        this->add_parameter(
            "reference y coordinate",
            reference_y_coordinate_,
            "GeoTIFF: select the value for y-coordinate in 2D. That is, the "
            "1D profile for the lower boundary is queried from the 2D "
            "geotiff image at coordinates (x, y=constant)");

        refinement_ = 0;
        this->add_parameter("refinement",
                            refinement_,
                            "number of global refinement steps applied to the "
                            "coarse mesh before aligning it with the elevation "
                            "profile");
      }


      void create_coarse_triangulation(
          dealii::Triangulation<dim> &triangulation) const final
      {
        AssertThrow(
            dim == 2,
            dealii::ExcMessage(
                "The geotiff profile ng geometry is only implemented in 2D."));

        /*
         * The mesh is refined and aligned on a temporary triangulation (a
         * parallel::distributed one for a fully distributed triangulation)
         * from which the cut cell decomposition creates the triangulation:
         */
        using Base = dealii::parallel::TriangulationBase<dim>;
        using FD = dealii::parallel::fullydistributed::Triangulation<dim>;
        const auto distributed = dynamic_cast<const Base *>(&triangulation);
        const auto fully_distributed = dynamic_cast<const FD *>(&triangulation);
        AssertThrow(
            fully_distributed != nullptr || distributed == nullptr,
            dealii::ExcMessage("The geotiff profile ng geometry only supports "
                               "serial and fully distributed triangulations."));

        if constexpr (dim == 2) {
          std::unique_ptr<dealii::Triangulation<dim>> temporary;
          if (fully_distributed != nullptr)
            temporary =
                std::make_unique<FD>(triangulation.get_mpi_communicator());
          else
            temporary = std::make_unique<dealii::Triangulation<dim>>();

          RectangularDomain<dim>::create_coarse_triangulation(*temporary);
          temporary->refine_global(refinement_);

          /*
           * The 1D profile is extracted from the geotiff image along the
           * line y = reference_y_coordinate_:
           */

          const auto height = [&](const dealii::Point<dim> &vertex) {
            return geotiff_reader_.compute_height(
                dealii::Point<2>(vertex[0], reference_y_coordinate_));
          };

          mesh_alignment_.align_with_elevation_profile(*temporary, height);

          cut_cell_decomposition_.create_triangulation(
              triangulation, *temporary, height, this->boundary_bottom_);

        } else {

          __builtin_trap();
        }
      }


      void update_dof_handler(dealii::DoFHandler<dim> &dof_handler) const final
      {
        /* Select the simplex finite element (index 1) on all triangles: */
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned() && cell->reference_cell().is_simplex())
            cell->set_active_fe_index(1);
      }


      Geometry<dim>::HP_Collection
      populate_hp_collections(const unsigned int fe_degree,
                              typename ryujin::Discretization<dim>::Collection
                                  &collection) const final
      {
        using namespace dealii;

        /*
         * Every collection has the cG Qk / dG Qk finite element for
         * quadrilaterals at index 0 and the cG Pk / dG Pk finite element
         * for the triangles created by the cut cell decomposition at index
         * 1:
         */

        if constexpr (dim == 2) {
          const auto mapping_degree = fe_degree;
          const auto quadrature_degree = fe_degree + 1;

          const auto make = [](auto collection,
                               const auto &quadrilateral,
                               const auto &triangle) {
            collection.push_back(quadrilateral);
            collection.push_back(triangle);
            return std::make_unique<decltype(collection)>(
                std::move(collection));
          };

          collection.finite_element_cg = make(hp::FECollection<dim>(),
                                              FE_Q<dim>(fe_degree),
                                              FE_SimplexP<dim>(fe_degree));
          collection.finite_element_dg = make(hp::FECollection<dim>(),
                                              FE_DGQ<dim>(fe_degree),
                                              FE_SimplexDGP<dim>(fe_degree));

          collection.mapping =
              make(hp::MappingCollection<dim>(),
                   MappingQ<dim>(mapping_degree),
                   MappingFE<dim>(FE_SimplexP<dim>(mapping_degree)));

          collection.quadrature = make(hp::QCollection<dim>(),
                                       QGauss<dim>(quadrature_degree),
                                       QGaussSimplex<dim>(quadrature_degree));
          collection.quadrature_high_order =
              make(hp::QCollection<dim>(),
                   QGauss<dim>(quadrature_degree + 1),
                   QGaussSimplex<dim>(quadrature_degree + 1));
          collection.nodal_quadrature =
              make(hp::QCollection<dim>(),
                   QGaussLobatto<dim>(quadrature_degree),
                   FETools::compute_nodal_quadrature(
                       FE_SimplexP<dim>(quadrature_degree)));

          collection.quadrature_1d = make(hp::QCollection<1>(),
                                          QGauss<1>(quadrature_degree),
                                          QGaussSimplex<1>(quadrature_degree));
          collection.nodal_quadrature_1d =
              make(hp::QCollection<1>(),
                   QGaussLobatto<1>(quadrature_degree),
                   QGaussLobatto<1>(quadrature_degree));

          /* One face quadrature collection per finite element: */
          using QCF = hp::QCollection<dim - 1>;
          collection.face_quadrature =
              make(std::vector<QCF>(),
                   QCF(QGauss<dim - 1>(quadrature_degree)),
                   QCF(QGaussSimplex<dim - 1>(quadrature_degree)));
          collection.face_nodal_quadrature =
              make(std::vector<QCF>(),
                   QCF(QGaussLobatto<dim - 1>(quadrature_degree)),
                   QCF(QGaussLobatto<dim - 1>(quadrature_degree)));

        } else {

          __builtin_trap();
        }

        return Geometry<dim>::HP_Collection::populated_by_geometry;
      }

    private:
      GeoTIFFReader geotiff_reader_;
      MeshAlignment<dim> mesh_alignment_;
      CutCellDecomposition<dim> cut_cell_decomposition_;

      double reference_y_coordinate_;
      unsigned int refinement_;
    };

  } /* namespace Geometries */
} /* namespace ryujin */
