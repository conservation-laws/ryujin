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

    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
    ACCESSOR_READ_ONLY(discretization)

    /* Implementation: */

    dealii::DoFHandler<dim> dof_handler_;
    ACCESSOR_READ_ONLY(dof_handler)

    dealii::IndexSet locally_owned_;
    ACCESSOR_READ_ONLY(locally_owned)

    dealii::IndexSet locally_relevant_;
    ACCESSOR_READ_ONLY(locally_relevant)

    dealii::SparsityPattern sparsity_pattern_;
    ACCESSOR_READ_ONLY(sparsity_pattern)

    std::map<dealii::types::global_dof_index,
             std::tuple<dealii::Tensor<1, dim>, dealii::types::boundary_id>>
        boundary_normal_map_;
    ACCESSOR_READ_ONLY(boundary_normal_map)

    dealii::AffineConstraints<double> affine_constraints_;
    ACCESSOR_READ_ONLY(affine_constraints)

    dealii::SparseMatrix<double> mass_matrix_;
    ACCESSOR_READ_ONLY(mass_matrix)

    dealii::SparseMatrix<double> lumped_mass_matrix_;
    ACCESSOR_READ_ONLY(lumped_mass_matrix)

    dealii::SparseMatrix<double> norm_matrix_;
    ACCESSOR_READ_ONLY(norm_matrix)

    std::array<dealii::SparseMatrix<double>, dim> cij_matrix_;
    ACCESSOR_READ_ONLY(cij_matrix)

    std::array<dealii::SparseMatrix<double>, dim> nij_matrix_;
    ACCESSOR_READ_ONLY(nij_matrix)
  };

} /* namespace grendel */

#endif /* OFFLINE_DATA_H */
