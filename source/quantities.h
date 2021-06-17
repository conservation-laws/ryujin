//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

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

namespace ryujin
{
  /**
   * A postprocessor class for quantities of interest.
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
     * @copydoc ProblemDescription::state_type
     */
    using state_type = ProblemDescription::state_type<dim, Number>;

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
    Quantities(const MPI_Comm &mpi_communicator,
               const ryujin::ProblemDescription &problem_description,
               const ryujin::OfflineData<dim, Number> &offline_data,
               const std::string &subsection = "Quantities");

    /**
     * Prepare evaluation. A call to @ref prepare() allocates temporary
     * storage and is necessary before compute() can be called.
     *
     * Calling prepare() allocates temporary storage for additional (3 *
     * dim + 1) scalar vectors of type OfflineData::scalar_type.
     *
     * The string parameter @ref name is used as base name for output files.
     */
    void prepare(std::string name);

    /**
     * Takes a state vector @p U at time t (obtained at the end of a full
     * Strang step) and accumulates statistics for quantities of interests
     * for all defined manifolds.
     */
    void accumulate(const vector_type &U, const Number t);

    /**
     * Write quantities of interest to designated output files.
     */
    void write_out(const vector_type &U, const Number t, unsigned int cycle);

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    std::vector<std::tuple<std::string, std::string, std::string>>
        interior_manifolds_;

    std::vector<std::tuple<std::string, std::string, std::string>>
        boundary_manifolds_;

    //@}
    /**
     * @name Internal data
     */
    //@{


    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const ProblemDescription> problem_description_;
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;

    /**
     * A tuple describing (local) dof index, boundary normal, normal mass,
     * boundary mass, boundary id, and position of the boundary degree of
     * freedom.
     */
    using boundary_point =
        std::tuple<dealii::types::global_dof_index /*local dof index*/,
                   dealii::Tensor<1, dim, Number> /*normal*/,
                   Number /*normal mass*/,
                   Number /*boundary mass*/,
                   dealii::types::boundary_id /*id*/,
                   dealii::Point<dim>> /*position*/;

    /**
     * The boundary map.
     */
    std::vector<std::tuple<std::string, std::vector<boundary_point>>>
        boundary_maps_;

    /**
     * A tuple describing boundary values we are interested in: the
     * primitive state and its second moment, boundary stresses and normal
     * pressure force.
     */
    using boundary_value =
        std::tuple<state_type /* primitive state */,
                   state_type /* primitive state second moment */,
                   dealii::Tensor<1, dim, Number> /* tau_n */,
                   dealii::Tensor<1, dim, Number> /* pn */>;

    /**
     * Tempoaral statistics we store for each boundary manifold.
     */
    using boundary_statistic =
        std::tuple<std::vector<boundary_value> /* values old */,
                   std::vector<boundary_value> /* values new */,
                   std::vector<boundary_value> /* values sum */,
                   Number /* t old */,
                   Number /* t new */,
                   Number /* t sum */>;

    /**
     * Associated statistics for The boundary map.
     */
    std::vector<boundary_statistic> boundary_statistics_;
    std::vector<std::vector<std::tuple<Number, boundary_value>>>
        boundary_time_series_;

    std::string base_name_;
    bool first_cycle_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    boundary_value
    accumulate_internal(const vector_type &U,
                        const std::vector<boundary_point> &boundary_map,
                        std::vector<boundary_value> &new_val);

    void write_out_internal(std::ostream &output,
                            const std::vector<boundary_value> &values,
                            const Number scale);

    void write_out_time_series(
        std::ostream &output,
        const std::vector<std::tuple<Number, boundary_value>> &values,
        bool append);

    //@}
  };

} /* namespace ryujin */
