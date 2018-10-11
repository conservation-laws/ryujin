#ifndef OFFLINE_DATA_H
#define OFFLINE_DATA_H

#include "boilerplate.h"
#include "discretization.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
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
  template <int dim>
  class OfflineData : public dealii::ParameterAcceptor
  {
  public:
    OfflineData(const MPI_Comm &mpi_communicator,
                dealii::TimerOutput &computing_timer,
                const grendel::Discretization<dim> &discretization,
                const std::string &subsection = "OfflineData");

    virtual ~OfflineData() final = default;

    /* Interface for computation: */

    virtual void prepare()
    {
      setup();
      assemble();
    }

    void setup();
    void assemble();

  protected:

    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    /**
     * A read-only reference to the underlying discretization.
     */
    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
    ACCESSOR_READ_ONLY(discretization)

    /**
     * The DofHandler for our (scalar) CG ansatz space.
     */
    dealii::DoFHandler<dim> dof_handler_;
    ACCESSOR_READ_ONLY(dof_handler)

    /**
     * An IndexSet storing all locally owned indices.
     */
    dealii::IndexSet locally_owned_;
    ACCESSOR_READ_ONLY(locally_owned)

    /**
     * An IndexSet storing all locally relevant indices.
     */
    dealii::IndexSet locally_relevant_;
    ACCESSOR_READ_ONLY(locally_relevant)

    /**
     * The SparsityPattern of our FiniteElement ansatz.
     */
    dealii::SparsityPattern sparsity_pattern_;
    ACCESSOR_READ_ONLY(sparsity_pattern)

    /**
     * 
     */
    std::map<dealii::types::global_dof_index,
             std::tuple<dealii::Tensor<1, dim>, dealii::types::boundary_id>>
        boundary_normal_map_;
    ACCESSOR_READ_ONLY(boundary_normal_map)

    /**
     * The AffineConstraints object is currently unused.
     */
    dealii::AffineConstraints<double> affine_constraints_;
    ACCESSOR_READ_ONLY(affine_constraints)

    /**
     * The mass matrix.
     */
    dealii::SparseMatrix<double> mass_matrix_;
    ACCESSOR_READ_ONLY(mass_matrix)

    /**
     * The lumped mass matrix.
     */
    dealii::SparseMatrix<double> lumped_mass_matrix_;
    ACCESSOR_READ_ONLY(lumped_mass_matrix)

    /**
     * The $(c_{ij})$ matrix.
     *
     * Departing from the mathematical formulation, where an entry $c_ij$
     * is itself a vector-valued element of $\mathbb{R}^{\text{dim}}$ we
     * store the matrix as a $p dim dimensional array of scalar-valued,
     * regular matrices.
     */
    std::array<dealii::SparseMatrix<double>, dim> cij_matrix_;
    ACCESSOR_READ_ONLY(cij_matrix)

    /**
     * The $(n_{ij})$ matrix.
     *
     * This matrix is defined as $n_{ij} = c_{ij} / |c_{ij}|$.
     */
    std::array<dealii::SparseMatrix<double>, dim> nij_matrix_;
    ACCESSOR_READ_ONLY(nij_matrix)

    /**
     * We also store the norm of all $c_{ij}$s separately in a @p norm
     * matrix_
     */
    dealii::SparseMatrix<double> norm_matrix_;
    ACCESSOR_READ_ONLY(norm_matrix)
  };

} /* namespace grendel */

#endif /* OFFLINE_DATA_H */
