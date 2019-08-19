#ifndef MATRIX_COMMUNICATOR_H
#define MATRIX_COMMUNICATOR_H

#include "offline_data.h"

#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

namespace grendel
{

  template <int dim, typename Number = double>
  class MatrixCommunicator
  {
  public:
    MatrixCommunicator(const MPI_Comm &mpi_communicator,
                       dealii::TimerOutput &computing_timer,
                       const grendel::OfflineData<dim, Number> &offline_data,
                       dealii::SparseMatrix<Number> &matrix);

    void prepare();
    void synchronize();

  private:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim, Number>> offline_data_;

    dealii::SparseMatrix<Number> &matrix_;

    dealii::SparseMatrix<unsigned int> indices_;
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
        matrix_temp_;
  };

} /* namespace grendel */

#endif /* MATRIX_COMMUNICATOR_H */
