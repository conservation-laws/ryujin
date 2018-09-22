#ifndef OFFLINE_DATA_H
#define OFFLINE_DATA_H

#include "boilerplate.h"
#include "discretization.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

namespace grendel
{

  template <int dim>
  class OfflineData : public dealii::ParameterAcceptor
  {
  public:
    OfflineData(const MPI_Comm &mpi_communicator,
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

    void clear();

  protected:

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
    A_RO(discretization)

    /* Implementation: */

    dealii::DoFHandler<dim> dof_handler_;
    A_RO(dof_handler)

    dealii::IndexSet locally_owned_;
    A_RO(locally_owned)

    dealii::IndexSet locally_relevant_;
    A_RO(locally_relevant)

    dealii::SparsityPattern sparsity_pattern_;
    A_RO(sparsity_pattern)

    dealii::AffineConstraints<double> affine_constraints_;
    A_RO(affine_constraints)

    dealii::SparseMatrix<double> mass_matrix_;
    A_RO(mass_matrix)

    dealii::SparseMatrix<double> lumped_mass_matrix_;
    A_RO(lumped_mass_matrix)

    dealii::SparseMatrix<double> norm_matrix_;
    A_RO(norm_matrix)

    std::array<dealii::SparseMatrix<double>, dim> cij_matrix_;
    A_RO(cij_matrix)

    std::array<dealii::SparseMatrix<double>, dim> nij_matrix_;
    A_RO(nij_matrix)
  };

} /* namespace grendel */

#endif /* OFFLINE_DATA_H */
