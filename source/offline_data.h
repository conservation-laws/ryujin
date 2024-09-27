//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "discretization.h"
#include "sparse_matrix_simd.h"
#include "state_vector.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

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
   * @ingroup Mesh
   */
  template <int dim, typename Number = double>
  class OfflineData : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc ryujin::ScalarVector
     */
    using ScalarVector = Vectors::ScalarVector<Number>;

    /**
     * Scalar vector storing single-precision floats
     */
    using ScalarVectorFloat = Vectors::ScalarVector<float>;

    /**
     * A tuple describing (local) dof index, boundary normal, normal mass,
     * boundary mass, boundary id, and position of the boundary degree of
     * freedom.
     */
    using BoundaryDescription =
        std::tuple<unsigned int /*i*/,
                   dealii::Tensor<1, dim, Number> /*normal*/,
                   Number /*normal mass*/,
                   Number /*boundary mass*/,
                   dealii::types::boundary_id /*id*/,
                   dealii::Point<dim>> /*position*/;

    /**
     * A tuple describing coupling boundary degrees of freedom on directly
     * enforced boundaries for which we have to symmetrize the d_ij matrix.
     */
    using CouplingDescription = std::tuple<unsigned int /*i*/, //
                                           unsigned int /*col_idx*/,
                                           unsigned int /*j*/>;

    /**
     * Constructor
     */
    OfflineData(const MPI_Comm &mpi_communicator,
                const Discretization<dim> &discretization,
                const std::string &subsection = "/OfflineData");

    /**
     * Prepare offline data. A call to prepare() internally calls setup()
     * and assemble().
     *
     * The problem_dimension and n_precomputed_values parameters is used to
     * set up appropriately sized vector partitioners for the state and
     * precomputed MultiComponentVector.
     */
    void prepare(const unsigned int problem_dimension,
                 const unsigned int n_precomputed_values,
                 const unsigned int n_auxiliary_state_vectors)
    {
      setup(problem_dimension, n_precomputed_values);
      assemble();
      create_multigrid_data();

      n_auxiliary_state_vectors_ = n_auxiliary_state_vectors;
    }

    /**
     * The DofHandler for our (scalar) CG ansatz space in (deal.II typical)
     * global numbering.
     */
    ACCESSOR_READ_ONLY(dof_handler)

    /**
     * An AffineConstraints object storing constraints in (Deal.II typical)
     * global numbering.
     */
    ACCESSOR_READ_ONLY(affine_constraints)

    /**
     * An MPI partitioner for the (scalar) Vector storing a scalar-valued
     * quantity.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(scalar_partitioner)

    /**
     * An MPI partitioner for the MultiComponentVector storing a
     * vector-valued quantity of size HyperbolicSystem::problem_dimension.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(hyperbolic_vector_partitioner)

    /**
     * An MPI partitioner for the MultiComponentVector storing a
     * vector-valued quantity of size HyperbolicSystem::problem_dimension.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(precomputed_vector_partitioner)

    /**
     * The block size of the auxiliary state vector.
     */
    ACCESSOR_READ_ONLY(n_auxiliary_state_vectors);

    /**
     * The subinterval \f$[0,\texttt{n_export_indices()})\f$ contains all
     * (SIMD-vectorized) indices of the interval
     * \f$[0,\texttt{n_locally_internal()})\f$ that are exported to
     * neighboring MPI ranks.
     *
     * @note The interval \f$[\texttt{n_locally_internal()},
     * \texttt{n_locally_relevant()})\f$ (consisting of non-SIMD-vectorized
     * indices) contains additional degrees of freedom that might have to
     * be exported to neighboring MPI ranks.
     */
    ACCESSOR_READ_ONLY(n_export_indices)

    /**
     * Number of locally owned internal degrees of freedom: In (MPI rank)
     * local numbering all indices in the half open interval [0,
     * n_locally_internal_) are owned by this processor, have standard
     * connectivity, and are not situated at a boundary.
     */
    ACCESSOR_READ_ONLY(n_locally_internal)

    /**
     * Number of locally owned degrees of freedom: In (MPI rank) local
     * numbering all indices in the half open interval [0,
     * n_locally_owned_) are owned by this processor.
     */
    ACCESSOR_READ_ONLY(n_locally_owned)

    /**
     * Number of locally relevant degrees of freedom: This number is the
     * toal number of degrees of freedom we store locally on this MPI rank.
     * I.e.,  we can access the half open interval [0, n_locally_relevant_)
     * on this machine.
     */
    ACCESSOR_READ_ONLY(n_locally_relevant)

    /**
     * The boundary map. Local numbering.
     *
     * For every degree of freedom that has nonzero support at the boundary
     * we record the global degree of freedom index along with a weighted
     * boundary normal, the associated boundary id, and position.
     *
     * This map is later used in OfflineData to handle boundary
     * degrees of freedom after every time step (for example to implement
     * reflective boundary conditions).
     */
    ACCESSOR_READ_ONLY(boundary_map)

    /**
     * A vector of tuples describing coupling degrees of freedom i and j
     * where both degrees of freedom are collocated at the boundary (and
     * hence the d_ij matrix has to be symmetrized). The function returns a
     * reference to a vector of tuples consisting of (i, col_idx, j).
     */
    ACCESSOR_READ_ONLY(coupling_boundary_pairs)

    /**
     * The boundary map on all levels of the grid in case multilevel
     * support was enabled.
     */
    ACCESSOR_READ_ONLY(level_boundary_map)

    /**
     * A sparsity pattern for (standard deal.II) matrices storing indices
     * in (Deal.II typical) global numbering.
     */
    ACCESSOR_READ_ONLY(sparsity_pattern)

    /**
     * A sparsity pattern for matrices in vectorized format. Local
     * numbering.
     */
    ACCESSOR_READ_ONLY(sparsity_pattern_simd)

    /**
     * The mass matrix. (SIMD storage, local numbering)
     */
    ACCESSOR_READ_ONLY(mass_matrix)

    /**
     * The inverse mass matrix. (SIMD storage, local numbering)
     *
     * This matrix is only available for a discontinuous finite Element
     * ansatz.
     */
    ACCESSOR_READ_ONLY(mass_matrix_inverse)

    /**
     * The lumped mass matrix. (stored as vector, local numbering)
     */
    ACCESSOR_READ_ONLY(lumped_mass_matrix)

    /**
     * The inverse of the lumped mass matrix. (stored as vector, local
     * numbering)
     */
    ACCESSOR_READ_ONLY(lumped_mass_matrix_inverse)

    /**
     * The lumped mass matrix on all levels of the grid in case multilevel
     * support was enabled.
     */
    ACCESSOR_READ_ONLY(level_lumped_mass_matrix)

    /**
     * The \f$(c_{ij})\f$ matrix. (SIMD storage, local numbering)
     */
    ACCESSOR_READ_ONLY(cij_matrix)

    /**
     * The incidence matrix \f$(beta_{ij})\f$: 1 for coupling face degrees
     * of freedom that share the same support point coordinate, 0 otherwise.
     *
     * (SIMD storage, local numbering)
     *
     * This matrix is only available for a discontinuous finite Element
     * ansatz.
     */
    ACCESSOR_READ_ONLY(incidence_matrix)

    /**
     * Size of computational domain.
     */
    ACCESSOR_READ_ONLY(measure_of_omega)

    /**
     * Returns a reference of the underlying Discretization object.
     */
    ACCESSOR_READ_ONLY(discretization)

  private:
    /**
     * Private methods used in prepare()
     */
    //@{

    /**
     * Set up affine constraints and sparsity pattern. Internally used in
     * setup().
     */
    void create_constraints_and_sparsity_pattern();

    /**
     * Set up DoFHandler, all IndexSet objects and the SparsityPattern.
     * Initialize matrix storage.
     *
     * The problem_dimension parameter is used to setup up an appropriately
     * sized vector partitioner for the MultiComponentVector.
     */
    void setup(const unsigned int problem_dimension,
               const unsigned int n_precomputed_values);

    /**
     * Assemble all matrices.
     */
    void assemble();

    /**
     * Create multigrid data.
     */
    void create_multigrid_data();

    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_;

    dealii::AffineConstraints<Number> affine_constraints_;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        scalar_partitioner_;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        hyperbolic_vector_partitioner_;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        precomputed_vector_partitioner_;

    unsigned int n_auxiliary_state_vectors_;

    unsigned int n_export_indices_;
    unsigned int n_locally_internal_;
    unsigned int n_locally_owned_;
    unsigned int n_locally_relevant_;

    using BoundaryMap = std::vector<BoundaryDescription>;
    BoundaryMap boundary_map_;
    std::vector<BoundaryMap> level_boundary_map_;

    using CouplingBoundaryPairs = std::vector<CouplingDescription>;
    CouplingBoundaryPairs coupling_boundary_pairs_;

    dealii::DynamicSparsityPattern sparsity_pattern_;

    SparsityPatternSIMD<dealii::VectorizedArray<Number>::size()>
        sparsity_pattern_simd_;

    SparseMatrixSIMD<Number> mass_matrix_;
    SparseMatrixSIMD<Number> mass_matrix_inverse_;

    ScalarVector lumped_mass_matrix_;
    ScalarVector lumped_mass_matrix_inverse_;

    std::vector<ScalarVectorFloat> level_lumped_mass_matrix_;

    SparseMatrixSIMD<Number, dim> cij_matrix_;
    SparseMatrixSIMD<Number> incidence_matrix_;

    Number measure_of_omega_;

    dealii::SmartPointer<const Discretization<dim>> discretization_;

    const MPI_Comm &mpi_communicator_;

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
    /**
     * @name Run time options
     */
    //@{

    double incidence_relaxation_even_;
    double incidence_relaxation_odd_;

    //@}
  };

} /* namespace ryujin */
