//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef QUANTITIES_H
#define QUANTITIES_H

#include <compile_time_options.h>

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <fstream>

namespace ryujin
{

  /**
   * A postprocessor class to compute scalar quantities of interest.
   *
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class Quantities final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type = ProblemDescription::rank1_type<dim, Number>;

    /**
     * Type used to store a curl of an 2D/3D vector field. Departing from
     * mathematical rigor, in 2D this is a number (stored as
     * `Tensor<1,1>`), in 3D this is a rank 1 tensor.
     */
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * Constructor.
     */
    Quantities(const MPI_Comm &mpi_communicator,
               const ProblemDescription &problem_description,
               const OfflineData<dim, Number> &offline_data,
               const std::string &subsection = "Quantities");

    /**
     * Prepare evaluation. A call to @ref prepare() allocates temporary
     * storage and is necessary before compute() can be called.
     *
     * A file descriptor to a log file @ref name is opened.
     */
    void prepare(const std::string &name);

    /**
     * Given a state vector @p U and a scalar vector @p alpha (as well as a
     * file name prefix @p name, the current time @p t, and the current
     * output cycle @p cycle) schedule a solution postprocessing and
     * output.
     *
     * The function post-processes quantities synchronously and (depending
     * on configuration options)
     *
     * The booleans @p output_full controls whether the full vector field
     * is written out. Correspondingly, @p output_cutplanes controls
     * whether cells in the vicinity of predefined cutplanes are written
     * out.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void compute(const vector_type &U,
                 Number t);

  private:
    /**
     * @name Run time options
     */
    //@{

    bool compute_conserved_quantities_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    const unsigned int mpi_rank;

    const ProblemDescription &problem_description;
    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;

    std::ofstream output;

    //@}
  };

} /* namespace ryujin */

#endif /* QUANTITIES_H */
