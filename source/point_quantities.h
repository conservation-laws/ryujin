//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef POINT_QUANTITIES_H
#define POINT_QUANTITIES_H

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "simd.h"

#include "initial_values.h"
#include "offline_data.h"
#include "problem_description.h"
#include "sparse_matrix_simd.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ryujin
{
  /**
   * A postprocessor class to compute point values of quantities of
   * interest.
   *
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class PointQuantities final : public dealii::ParameterAcceptor
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
     * A distributed block vector used for temporary storage of the
     * velocity field.
     */
    using block_vector_type =
        dealii::LinearAlgebra::distributed::BlockVector<Number>;

    /**
     * Constructor.
     */
    PointQuantities(const MPI_Comm &mpi_communicator,
                    const ryujin::ProblemDescription &problem_description,
                    const ryujin::OfflineData<dim, Number> &offline_data,
                    const std::string &subsection = "PointQuantities");

    /**
     * Prepare evaluation. A call to @ref prepare() allocates temporary
     * storage and is necessary before compute() can be called.
     *
     * Calling prepare() allocates temporary storage for additional (3 *
     * dim + 1) scalar vectors of type OfflineData::scalar_type.
     *
     * The string parameter @ref name is used as base name for output files.
     */
    void prepare();

    /**
     * Takes a state vector @p U at time t (obtained at the end of a full
     * Strang step) and a velocity vector @p velocity computed at time
     * \f$t_(n+1/2)\f$ (@p t_interp) during the implicit parabolic step.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void compute(const vector_type &U,
                 const Number t,
                 std::string name,
                 unsigned int cycle);

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    using plane_description = std::tuple<dealii::Point<dim> /*origin*/,
                                         dealii::Tensor<1, dim> /*normal*/,
                                         double /*tolerance*/>;

    std::vector<plane_description> output_planes_;

    std::vector<std::map<dealii::types::global_dof_index, dealii::Point<dim>>>
        cutplane_map_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const ryujin::ProblemDescription> problem_description_;
    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;

    dealii::MatrixFree<dim, Number> matrix_free_;

    block_vector_type velocity_;
    block_vector_type vorticity_;
    block_vector_type boundary_stress_;
    scalar_type lumped_boundary_mass_;
    scalar_type pressure_;

    //@}
  };

} /* namespace ryujin */

#endif /* POINT_QUANTITIES_H */
