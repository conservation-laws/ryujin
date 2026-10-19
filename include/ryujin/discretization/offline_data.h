//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <ryujin/interface/offline_data.h>

#include "convenience_macros.h"
#include "discretization.h"
#include "gpu.h"
#include "mpi_ensemble.h"
#include "sparse_matrix.h"
#include "sparsity_pattern.h"
#include "state_vector.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>

namespace ryujin
{
  /**
   * A class to store all data that can be precomputed offline.
   *
   * This class takes a reference to a Discretization object (that itself
   * holds a Triangulation, FiniteElement, Mapping, and Quadrature object).
   *
   * Most notably this class sets up a DoFHandler, the
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
   * @ingroup Mesh
   */
  template <int dim, typename Number = double>
  class OfflineData final : public Interface::OfflineData<dim, Number>
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using typename Interface::OfflineData<dim, Number>::BoundaryDescription;

    using typename Interface::OfflineData<dim, Number>::CouplingDescription;

    using typename Interface::OfflineData<dim, Number>::BoundaryMap;

    using typename Interface::OfflineData<dim, Number>::CouplingBoundaryPairs;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor
     */
    OfflineData(const MPIEnsemble &mpi_ensemble,
                const Interface::Discretization<dim> &discretization,
                const std::string &subsection = "/OfflineData");

    /**
     * Prepare offline data. A call to prepare() sets up storage, dof
     * handlers, sparsity patterns, and assembles all matrices.
     *
     * The problem_dimension and n_precomputed_values parameters is used to
     * set up appropriately sized vector partitioners for the state and
     * precomputed MultiComponentVector.
     */
    void prepare(const unsigned int problem_dimension,
                 const unsigned int n_precomputed_values) override;

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

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Return a writable reference to the dof handler.
     */
    dealii::DoFHandler<dim> &writable_dof_handler()
    {
      if (this->discretization_->have_discontinuous_ansatz()) {
        Assert(this->dof_handler_dg_, dealii::ExcInternalError());
        return *this->dof_handler_dg_;
      } else {
        Assert(this->dof_handler_cg_, dealii::ExcInternalError());
        return *this->dof_handler_cg_;
      }
    }

    /**
     * Set up DoFHandlers. Internally used in prepare().
     *
     * @note This method populates various OfflineData internal data structures.
     */
    void create_dof_handlers();

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

} /* namespace ryujin */
