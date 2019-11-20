#ifndef OFFLINE_DATA_H
#define OFFLINE_DATA_H

#include "discretization.h"
#include "helper.h"
#include "sparse_matrix_simd.h"
#include <compile_time_options.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

namespace grendel
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
   * independent, it only depends on the Discretization.
   */
  template <int dim, typename Number = double>
  class OfflineData : public dealii::ParameterAcceptor
  {
  public:
    OfflineData(const MPI_Comm &mpi_communicator,
                const grendel::Discretization<dim> &discretization,
                const std::string &subsection = "OfflineData");

    /**
     * Prepare offline data. A call to @ref prepare() internally calls
     * @ref setup() and @ref assemble().
     */
    void prepare()
    {
      setup();
      assemble();
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

  protected:
    /**
     * The DofHandler for our (scalar) CG ansatz space in global numbering.
     */
    dealii::DoFHandler<dim> dof_handler_;
    ACCESSOR_READ_ONLY(dof_handler)

    /**
     * An AffineConstraints object storing constraints in global numbering.
     */
    dealii::AffineConstraints<Number> affine_constraints_;
    ACCESSOR_READ_ONLY(affine_constraints)

    /**
     * The MPI partitioner used in all distributed Vectors.
     */
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner_;
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(partitioner)

    /**
     * FIXME
     */
    unsigned int n_import_indices_;
    ACCESSOR_READ_ONLY(n_import_indices)

    /**
     * Number of locally owned internal degrees of freedom: In (MPI rank)
     * local numbering all indices in the half open interval [0,
     * n_locally_internal_) are owned by this processor, have standard
     * connectivity, and are not situated at a boundary.
     */
    unsigned int n_locally_internal_;
    ACCESSOR_READ_ONLY(n_locally_internal)

    /**
     * Number of locally owned degrees of freedom: In (MPI rank) local
     * numbering all indices in the half open interval [0,
     * n_locally_owned_) are owned by this processor.
     */
    unsigned int n_locally_owned_;
    ACCESSOR_READ_ONLY(n_locally_owned)

    /**
     * Number of locally relevant degrees of freedom: This number is the
     * toal number of degrees of freedom we store locally on this MPI rank.
     * I.e.,  we can access the half open interval [0, n_locally_relevant_)
     * on this machine.
     */
    unsigned int n_locally_relevant_;
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
    std::map<dealii::types::global_dof_index,
             std::tuple<dealii::Tensor<1, dim, Number>,
                        dealii::types::boundary_id,
                        dealii::Point<dim>>>
        boundary_normal_map_;
    ACCESSOR_READ_ONLY(boundary_normal_map)

    /**
     * A sparsity pattern for matrices in vectorized format. Local
     * numbering.
     */
    SparsityPatternSIMD<dealii::VectorizedArray<Number>::n_array_elements>
        sparsity_pattern_simd_;
    ACCESSOR_READ_ONLY(sparsity_pattern_simd)

    /**
     * The mass matrix. (SIMD storage, local numbering)
     */
    SparseMatrixSIMD<Number> mass_matrix_;
    ACCESSOR_READ_ONLY(mass_matrix)

    /**
     * The lumped mass matrix.
     */
    dealii::LinearAlgebra::distributed::Vector<Number> lumped_mass_matrix_;
    ACCESSOR_READ_ONLY(lumped_mass_matrix)

    /**
     * The inverse of the lumped mass matrix.
     */
    dealii::LinearAlgebra::distributed::Vector<Number>
        lumped_mass_matrix_inverse_;
    ACCESSOR_READ_ONLY(lumped_mass_matrix_inverse)

    /**
     * The stiffness matrix $(beta_{ij})$:
     *   $\beta_{ij} = \nabla\varphi_{j}\cdot\nabla\varphi_{i}$
     * (SIMD storage, local numbering)
     */
    SparseMatrixSIMD<Number> betaij_matrix_;
    ACCESSOR_READ_ONLY(betaij_matrix)

    /**
     * The $(c_{ij})$ matrix. (SIMD storage, local numbering)
     */
    SparseMatrixSIMD<Number, dim> cij_matrix_;
    ACCESSOR_READ_ONLY(cij_matrix)

    /**
     * Size of computational domain.
     */
    Number measure_of_omega_;
    ACCESSOR_READ_ONLY(measure_of_omega)

  private:
    /* Scratch storage: */
    dealii::SparsityPattern sparsity_pattern_assembly_;
    dealii::AffineConstraints<Number> affine_constraints_assembly_;

    const MPI_Comm &mpi_communicator_;

    /**
     * A pointer to a discretization object.
     */
    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
    ACCESSOR_READ_ONLY(discretization)
  };

} /* namespace grendel */

#endif /* OFFLINE_DATA_H */
