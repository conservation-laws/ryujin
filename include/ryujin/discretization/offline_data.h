//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/interface/offline_data.h>


#include <memory>
#include <string>
#include <vector>

namespace ryujin
{
  /**
   * A class to store all data that can be precomputed offline.
   *
   * This class takes a reference to a Discretization object (that itself
   * holds a Triangulation, FiniteElement, Mapping, Quadrature, and
   * DoFHandler objects).
   *
   * Most notably this class renumbers the degrees of freedom of the
   * DoFHandler objects owned by the Discretization, sets up the
   * SparsityPattern, various IndexSet objects to hold locally
   * owned and locally relevant indices, and precomputes all matrices (mass
   * matrix, lumped mass matrix, $c_{ij}$ matrices, and $n_{ij}$ matrices).
   *
   * After @p prepare() is called, all getter functions return valid
   * references.
   *
   * @note The offline data precomputed in this class is problem
   * independent, it only depends on the chosen geometry and ansatz stored
   * in the Discretization class.
   *
   * @note Data structures in OfflineData are initialized with the ensemble
   * subrange communicator stored in MPIEnsemble.
   *
   * The template parameters @p structdim and @p dim follow the deal.II
   * convention for dealii::DoFHandler<structdim, dim>. All data that the
   * algebraic part of ryujin operates on is accessible through the base
   * class Interface::OfflineData<dim, Number>, which only depends on the
   * space dimension @p dim.
   *
   * @ingroup MeshAndOfflineData
   */
  template <int structdim, int dim, typename Number = double>
  class OfflineData : public Interface::OfflineData<dim, Number>
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using Base = Interface::OfflineData<dim, Number>;

    using BoundaryDescription = Base::BoundaryDescription;
    using CouplingDescription = Base::CouplingDescription;
    using BoundaryMap = Base::BoundaryMap;
    using CouplingBoundaryPairs = Base::CouplingBoundaryPairs;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor
     */
    OfflineData(const MPIEnsemble &mpi_ensemble,
                Discretization<structdim, dim> &discretization,
                const std::string &subsection = "/OfflineData");

    /**
     * Prepare offline data. A call to prepare() renumbers the degrees of
     * freedom of the Discretization, sets up storage, sparsity patterns,
     * and assembles all matrices. The degrees of freedom must have been
     * distributed with Discretization::prepare() or
     * Discretization::distribute_dofs().
     *
     * The problem_dimension and n_precomputed_values parameters is used to
     * set up appropriately sized vector partitioners for the state and
     * precomputed MultiComponentVector.
     */
    void prepare(const unsigned int problem_dimension,
                 const unsigned int n_precomputed_values) override;

    //@}
    /**
     * @name Information and statistics
     */
    //@{

    /**
     * The boundary map on all levels of the grid in case multilevel
     * support was enabled.
     */
    ACCESSOR_READ_ONLY(level_boundary_map)

    /**
     * The lumped mass matrix on all levels of the grid in case multilevel
     * support was enabled.
     */
    ACCESSOR_READ_ONLY(level_lumped_mass_matrix)

    /**
     * Returns a reference of the underlying Discretization object.
     */
    ACCESSOR_READ_ONLY(discretization)

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    bool treat_fe_nothing_as_boundary_;
    double incidence_relaxation_even_;
    double incidence_relaxation_odd_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<Discretization<structdim, dim>> discretization_;

    std::vector<BoundaryMap> level_boundary_map_;

    std::vector<ScalarHostVector<float>> level_lumped_mass_matrix_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Renumber the hyperbolic DoFHandler for use with our SIMD sparsity
     * pattern. Internally used in prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void renumber_for_simd();

    /**
     * Set up affine constraints and sparsity pattern. Internally used in
     * prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void create_constraints_and_sparsity_pattern();

    /**
     * Ensure that the SIMD stride of the renumbered degrees of freedom
     * remains consistent across all MPI ranks. Internally used in
     * prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void ensure_simd_stride_consistency();

    /**
     * Create partitioner. Internally used in prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void create_partitioner_and_simd_sparsity(
        const unsigned int problem_dimension,
        const unsigned int n_precomputed_values);

    /**
     * Set up dg_to_cg_index_map() and cg_to_dg_index_map(). Internally
     * used in prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void create_index_maps();

    /**
     * Set up support_points(). Internally used in prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void create_support_points();

    /**
     * Assemble all matrices. Internally used in prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void create_matrices();

    /**
     * Create multigrid data. Internally used in prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void create_multigrid_data();

    /**
     * Construct a boundary map for a given set of DoFHandler iterators.
     */
    template <typename ITERATOR1, typename ITERATOR2>
    BoundaryMap construct_boundary_map(
        const ITERATOR1 &begin,
        const ITERATOR2 &end,
        const dealii::Utilities::MPI::Partitioner &partitioner) const;

    /**
     * Collect coupling pairs of locally owned (and locally relevant)
     * boundary degrees of freedom.
     */
    template <typename ITERATOR1, typename ITERATOR2>
    CouplingBoundaryPairs collect_coupling_boundary_pairs(
        const ITERATOR1 &begin,
        const ITERATOR2 &end,
        const dealii::Utilities::MPI::Partitioner &partitioner) const;

    //@}
  };

} // namespace ryujin
