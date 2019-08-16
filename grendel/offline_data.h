#ifndef OFFLINE_DATA_H
#define OFFLINE_DATA_H

#include "discretization.h"
#include "helper.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/partitioner.h>
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

    /**
     * Prepare offline data. A call to @ref prepare() internally calls @ref
     * setup() and @ref assemble().
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
     * The DofHandler for our (scalar) CG ansatz space.
     */
    dealii::DoFHandler<dim> dof_handler_;
    ACCESSOR_READ_ONLY(dof_handler)

    /**
     * FIXME Description.
     */
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner_;
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(partitioner)

    /**
     * FIXME Description.
     */
    unsigned int n_locally_owned_;
    ACCESSOR_READ_ONLY(n_locally_owned)

    /**
     * FIXME Description.
     */
    unsigned int n_locally_relevant_;
    ACCESSOR_READ_ONLY(n_locally_relevant)

    /**
     * The SparsityPattern.
     */
    dealii::SparsityPattern sparsity_pattern_;
    ACCESSOR_READ_ONLY(sparsity_pattern)

    /**
     * The boundary map.
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
             std::tuple<dealii::Tensor<1, dim>,
                        dealii::types::boundary_id,
                        dealii::Point<dim>>>
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
     * Size of computational domain.
     */
    double measure_of_omega_;
    ACCESSOR_READ_ONLY(measure_of_omega)


    /**
     * The $(b_{ij})$ matrix:
     *   $b_{ij} = \delta_{ij} - m_{ij}/m{j}$
     */
    dealii::SparseMatrix<double> bij_matrix_;
    ACCESSOR_READ_ONLY(bij_matrix)

    /**
     * The $(beta_{ij})$ matrix.
     */
    dealii::SparseMatrix<double> betaij_matrix_;
    ACCESSOR_READ_ONLY(betaij_matrix)

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

  private:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
  };

} /* namespace grendel */

#endif /* OFFLINE_DATA_H */
