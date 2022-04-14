//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "discretization.h"
#include "multicomponent_vector.h"
#include "problem_description.h"
#include "sparse_matrix_simd.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/partitioner.h>
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
   * holds a @ref Triangulation, @ref FiniteElement, @ref Mapping, and @ref
   * Quadrature object).
   *
   * Most notably this class sets up a @ref DoFHandler, the
   * @ref SparsityPattern, various @ref IndexSet objects to hold locally
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
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class OfflineData : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription::problem_dimension<dim>;
    // clang-format on

    /**
     * Shorthand typedef for
     * dealii::LinearAlgebra::distributed::Vector<Number>.
     */
    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;

    /**
     * Shorthand typedef for a MultiComponentVector storing the current
     * simulation state.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * A tuple describing global dof index, boundary normal, normal mass,
     * boundary mass, boundary id, and position of the boundary degree of
     * freedom.
     */
    using boundary_description =
        std::tuple<dealii::Tensor<1, dim, Number> /*normal*/,
                   Number /*normal mass*/,
                   Number /*boundary mass*/,
                   dealii::types::boundary_id /*id*/,
                   dealii::Point<dim>> /*position*/;

    /**
     * Constructor
     */
    OfflineData(const MPI_Comm &mpi_communicator,
                const ryujin::Discretization<dim> &discretization,
                const std::string &subsection = "OfflineData");

    /**
     * Prepare offline data. A call to @ref prepare() internally calls
     * @ref setup() and @ref assemble().
     */
    void prepare()
    {
      setup();
      assemble();
      create_multigrid_data();
    }

    /**
     * Set up DoFHandler, all IndexSet objects and the SparsityPattern.
     * Initialize matrix storage.
     */
    void setup();

    /**
     * Assemble all matrices.
     */
    void assemble();

    /**
     * Create multigrid data.
     */
    void create_multigrid_data();

  private:
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_;

    dealii::AffineConstraints<Number> affine_constraints_;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        scalar_partitioner_;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        vector_partitioner_;

    unsigned int n_export_indices_;
    unsigned int n_locally_internal_;
    unsigned int n_locally_owned_;
    unsigned int n_locally_relevant_;

    using boundary_map_type =
        std::multimap<dealii::types::global_dof_index, boundary_description>;
    using coupling_boundary_pairs_type =
        std::vector<std::tuple<dealii::types::global_dof_index,
                               unsigned int,
                               dealii::types::global_dof_index>>;
    boundary_map_type boundary_map_;
    coupling_boundary_pairs_type coupling_boundary_pairs_;

    std::vector<boundary_map_type> level_boundary_map_;

    dealii::DynamicSparsityPattern sparsity_pattern_;

    SparsityPatternSIMD<dealii::VectorizedArray<Number>::size()>
        sparsity_pattern_simd_;

    SparseMatrixSIMD<Number> mass_matrix_;

    dealii::LinearAlgebra::distributed::Vector<Number> lumped_mass_matrix_;
    dealii::LinearAlgebra::distributed::Vector<Number>
        lumped_mass_matrix_inverse_;

    std::vector<dealii::LinearAlgebra::distributed::Vector<float>>
        level_lumped_mass_matrix_;

    SparseMatrixSIMD<Number> betaij_matrix_;
    SparseMatrixSIMD<Number, dim> cij_matrix_;

    Number measure_of_omega_;

    dealii::SmartPointer<const ryujin::Discretization<dim>> discretization_;

    const MPI_Comm &mpi_communicator_;

    /**
     * Construct a boundary map for a given set of DoFHandler iterators.
     */
    template <typename ITERATOR1, typename ITERATOR2>
    boundary_map_type construct_boundary_map(
        const ITERATOR1 &begin,
        const ITERATOR2 &end,
        const dealii::Utilities::MPI::Partitioner &partitioner) const;

  protected:
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
     * An MPI partitioner for all parallel distributed vectors storing a
     * scalar quantity.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(scalar_partitioner)

    /**
     * An MPI partitioner for all parallel distributed vectors storing a
     * vector-valued quantity of size
     * ProblemDescription::problem_dimension.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(vector_partitioner)

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
     * This map is later used in @ref OfflineData to handle boundary
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
     * The lumped mass matrix.
     */
    ACCESSOR_READ_ONLY(lumped_mass_matrix)

    /**
     * The inverse of the lumped mass matrix.
     */
    ACCESSOR_READ_ONLY(lumped_mass_matrix_inverse)

    /**
     * The lumped mass matrix on all levels of the grid in case multilevel
     * support was enabled.
     */
    ACCESSOR_READ_ONLY(level_lumped_mass_matrix)

    /**
     * The stiffness matrix \f$(beta_{ij})\f$:
     *   \f$\beta_{ij} = \nabla\varphi_{j}\cdot\nabla\varphi_{i}\f$
     * (SIMD storage, local numbering)
     */
    ACCESSOR_READ_ONLY(betaij_matrix)

    /**
     * The \f$(c_{ij})\f$ matrix. (SIMD storage, local numbering)
     */
    ACCESSOR_READ_ONLY(cij_matrix)

    /**
     * Size of computational domain.
     */
    ACCESSOR_READ_ONLY(measure_of_omega)

    /**
     * Returns a reference of the underlying Discretization object.
     */
    ACCESSOR_READ_ONLY(discretization)
  };

} /* namespace ryujin */
