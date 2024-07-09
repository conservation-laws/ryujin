//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <offline_data.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/distributed/solution_transfer.h>

namespace ryujin
{
  /**
   * Controls the adaptation strategy used in MeshAdaptor.
   *
   * @ingroup Mesh
   */
  enum class AdaptationStrategy {
    /**
     * Performs a simple global refinement at specified timepoints.
     */
    global_refinement,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::AdaptationStrategy,
             LIST({ryujin::AdaptationStrategy::global_refinement,
                   "global refinement"}, ));
#endif

namespace ryujin
{
  /**
   * The MeshAdaptor class is responsible for performing global or local
   * mesh adaptation.
   *
   * @ingroup Mesh
   */
  template <typename Description, int dim, typename Number = double>
  class MeshAdaptor final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using StateVector = typename View::StateVector;

    using ScalarVector = Vectors::ScalarVector<Number>;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    MeshAdaptor(const MPI_Comm &mpi_communicator,
                const OfflineData<dim, Number> &offline_data,
                const HyperbolicSystem &hyperbolic_system,
                const ParabolicSystem &parabolic_system,
                const std::string &subsection = "/MeshAdaptor");

    /**
     * Prepare temporary storage and clean up internal data for the
     * analyze() facility.
     *
     * @note this function does not reset the internal state_ and
     * solution_transfer_ objects as those are needed to finalize the
     * solution transfer to a new mesh.
     */
    void prepare(const Number t);

    /**
     * Analyze the given StateVector with the configured adaptation strategy
     * and decide whether a mesh adaptation cycle should be performed.
     */
    void analyze(const StateVector &state_vector,
                 const Number t,
                 unsigned int cycle);

    /**
     * Perform mesh adaptation with the configured adaptation strategy. The
     * function will modify the supplied @p triangulation and
     * @p state_vector objects. The prepare_compute_kernels argument must
     * be a lambda that reinitializes all data structures after mesh
     * refinement.
     */
    template <typename Callable>
    void adapt_mesh_and_transfer_state_vector(
        dealii::Triangulation<dim> &triangulation,
        StateVector &state_vector,
        const Callable &prepare_compute_kernels) const
    {
      mark_cells_for_coarsening_and_refinement(triangulation);

      triangulation.prepare_coarsening_and_refinement();
      prepare_for_interpolation(state_vector);

      triangulation.execute_coarsening_and_refinement();
      prepare_compute_kernels();

      interpolate(state_vector);

      // need_mesh_adaptation_ is already set to false due to the call to
      // prepare_compute_kernels()
    }

    /**
     * A boolean indicating whether we should perform a mesh adapation step
     * in the current cycle. The analyze() method will set this boolean to
     * true whenever the selected adaptation strategy advices to perform an
     * adaptation cycle.
     */
    ACCESSOR_READ_ONLY(need_mesh_adaptation)

  private:
    /**
     * @name Run time options
     */
    //@{

    AdaptationStrategy adaptation_strategy_;
    std::vector<Number> t_global_refinements_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const ParabolicSystem> parabolic_system_;

    bool need_mesh_adaptation_;

    mutable std::unique_ptr<
        dealii::parallel::distributed::SolutionTransfer<dim, ScalarVector>>
        solution_transfer_;

    mutable std::vector<ScalarVector> state_;

    //@}
    /**
     * @name Private methods for adapt_mesh_and_transfer_state_vector()
     */
    //@{

    /**
     * Mark cells for coarsening and refinement with the configured marking
     * strategy.
     */
    void mark_cells_for_coarsening_and_refinement(
        dealii::Triangulation<dim> &triangulation) const;

    /**
     * Read in a state vector (in conserved quantities). The function
     * populates auxiliary distributed vectors that store the given
     * conserved state and then calls the underlying deal.II
     * SolutionTransfer::prepare_for_coarsening_and_refinement();
     *
     * @note This function has to be called before the actual grid refinement.
     *
     * @note This function initializes the internal state_ and
     * solution_transfer_ objects.
     */
    void prepare_for_interpolation(const StateVector &state_vector) const;

    /**
     * Finalize the state vector transfer by calling
     * SolutionTransfer::interpolate() and repopulating the state vector.
     *
     * @note This function has to be called after the actual grid refinement.
     *
     * @note After successful state vector transfer this function frees the
     * internal state_ and solution_transfer_ objects.
     */
    void interpolate(StateVector &state_vector) const;

    //@}
  };

} // namespace ryujin
