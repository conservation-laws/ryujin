//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef DISSIPATION_MODULE_H
#define DISSIPATION_MODULE_H

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "simd.h"

#include "initial_values.h"
#include "offline_data.h"
#include "problem_description.h"
#include "sparse_matrix_simd.h"
#include "dissipation_gmg_operators.h"

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

namespace ryujin
{
  template <int, typename>
  class DiagonalMatrix;

  /**
   * Minimum entropy guaranteeing second-order time stepping for the
   * parabolic limit equations.
   *
   * @ingroup NavierModule
   */
  template <int dim, typename Number = double>
  class DissipationModule final : public dealii::ParameterAcceptor
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
    DissipationModule(const MPI_Comm &mpi_communicator,
                      std::map<std::string, dealii::Timer> &computing_timer,
                      const ryujin::ProblemDescription &problem_description,
                      const ryujin::OfflineData<dim, Number> &offline_data,
                      const ryujin::InitialValues<dim, Number> &initial_values,
                      const std::string &subsection = "DissipationModule");

    /**
     * Prepare time stepping. A call to @ref prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector U perform an implicit
     * update of the dissipative parabolic limiting problem and store the
     * result again in U. The function
     *
     *  - returns the chosen time step size tau (for compatibility with the
     *    EulerModule::euler_step() and Euler_Module::step() interfaces).
     *
     *  - performs a time step and populates the vector U_new by the
     *    result. The time step is performed with either tau_max (if tau ==
     *    0), or tau (if tau != 0). Here, tau_max is the computed maximal
     *    time step size and tau is the optional third parameter.
     */
    Number step(vector_type &U, Number t, Number tau, unsigned int cycle);

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    bool use_gmg_velocity_;
    ACCESSOR_READ_ONLY(use_gmg_velocity)

    bool use_gmg_internal_energy_;
    ACCESSOR_READ_ONLY(use_gmg_internal_energy)

    Number tolerance_;
    bool tolerance_linfty_norm_;

    Number shift_;

    unsigned int gmg_max_iter_vel_;
    unsigned int gmg_max_iter_en_;
    double gmg_smoother_range_vel_;
    double gmg_smoother_range_en_;
    double gmg_smoother_max_eig_vel_;
    double gmg_smoother_max_eig_en_;
    unsigned int gmg_smoother_degree_;
    unsigned int gmg_smoother_n_cg_iter_;
    unsigned int gmg_min_level_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const ryujin::ProblemDescription> problem_description_;
    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::InitialValues<dim, Number>>
        initial_values_;

    double n_iterations_velocity_;
    ACCESSOR_READ_ONLY(n_iterations_velocity)

    double n_iterations_internal_energy_;
    ACCESSOR_READ_ONLY(n_iterations_internal_energy)

    dealii::MatrixFree<dim, Number> matrix_free_;

    block_vector_type velocity_;
    ACCESSOR_READ_ONLY(velocity)

    block_vector_type velocity_rhs_;

    scalar_type internal_energy_;
    scalar_type internal_energy_rhs_;

    scalar_type density_;

    Number tau_;
    Number theta_;

    dealii::MGLevelObject<dealii::MatrixFree<dim, float>> level_matrix_free_;
    dealii::MGConstrainedDoFs mg_constrained_dofs_;
    dealii::MGLevelObject<dealii::LinearAlgebra::distributed::Vector<float>>
        level_density_;
    MGTransferVelocity<dim, float> mg_transfer_velocity_;
    dealii::MGLevelObject<VelocityMatrix<dim, float, Number>>
        level_velocity_matrices_;
    MGTransferEnergy<dim, float> mg_transfer_energy_;
    dealii::MGLevelObject<EnergyMatrix<dim, float, Number>>
        level_energy_matrices_;

    dealii::mg::SmootherRelaxation<
        dealii::PreconditionChebyshev<
            VelocityMatrix<dim, float, Number>,
            dealii::LinearAlgebra::distributed::BlockVector<float>,
            DiagonalMatrix<dim, float>>,
        dealii::LinearAlgebra::distributed::BlockVector<float>>
        mg_smoother_velocity_;

    dealii::mg::SmootherRelaxation<
        dealii::PreconditionChebyshev<
            EnergyMatrix<dim, float, Number>,
            dealii::LinearAlgebra::distributed::Vector<float>>,
        dealii::LinearAlgebra::distributed::Vector<float>>
        mg_smoother_energy_;

    //@}
  };

} /* namespace ryujin */

#endif /* DISSIPATION_MODULE_H */
