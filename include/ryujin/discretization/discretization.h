//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/geometries/geometry.h>
#include <ryujin/interface/discretization.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <memory>
#include <vector>

namespace ryujin
{
  /**
   * An enum class for setting the type of Triangulation that should be
   * constructed.
   *
   * @ingroup MeshAndOfflineData
   */
  enum class MeshType {
    /** Use serial dealii::Triangulation<dim> */
    serial,
    /** Use parallel dealii::parallel::shared::Triangulation<dim> */
    parallel_shared,
    /** Use parallel dealii::parallel::distributed::Triangulation<dim> */
    parallel_distributed,
    /** Use parallel dealii::parallel::fullydistributed::Triangulation<dim> */
    parallel_fullydistributed
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::MeshType,
             LIST({ryujin::MeshType::serial, "serial"},
                  {ryujin::MeshType::parallel_shared, "parallel shared"},
                  {ryujin::MeshType::parallel_distributed,
                   "parallel distributed"},
                  {ryujin::MeshType::parallel_fullydistributed,
                   "parallel fullydistributed"}));
#endif

namespace ryujin
{
  /**
   * This class is as a container for data related to the discretization,
   * this includes the triangulation, finite element, mapping, and
   * quadrature. After prepare() is called, the getter functions
   * Discretization::triangulation(), Discretization::finite_element(),
   * Discretization::mapping(), and Discretization::quadrature() return
   * valid const references to the mentioned objects.
   *
   * The class uses dealii::ParameterAcceptor to handle a multitude of
   * parameters to control the creation of meshes for a variety of
   * benchmark configurations and to read in meshes in one of the formats
   * supported by the deal.II library.
   *
   * @ingroup MeshAndOfflineData
   */
  template <int dim>
  class Discretization final : public Interface::Discretization<dim>
  {
  public:
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    Discretization(const MPIEnsemble &mpi_ensemble,
                   const std::string &subsection = "/Discretization");

    /**
     * Create the triangulation and set up the finite element, mapping and
     * quadrature objects.
     */
    void prepare(const std::string &base_name) override;

    /**
     * Recompute the cached mapping after the triangulation, or the
     * transformation(), has changed.
     *
     * The mapping is a MappingQCache if the selected geometry provides a
     * Geometry::transformation(), or once a transformation has been set.
     */
    void update_mapping() override;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    MeshType mesh_type_;

    std::string geometry_;

    bool mesh_writeout_;
    double mesh_distortion_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    std::vector<std::unique_ptr<Geometry<dim>>> geometry_list_;

    typename Geometry<dim>::HP_Collection collection_type_ =
        Geometry<dim>::HP_Collection::standard_quadrilaterals;

    //@}

    /**
     * Create a MappingQCache and set it as the mapping of the collection.
     */
    void create_mapping_cache();
  };
} // namespace ryujin
