//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>

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
     * A boolean indicating whether we should perform a mesh adapation step
     * in the current cycle. The analyze() method will set this boolean to
     * true whenever the selected adaptation strategy advices to perform an
     * adaptation cycle.
     */
    ACCESSOR_READ_ONLY(need_mesh_adaptation)

    /**
     * Mark cells for coarsening and refinement with the configured marking
     * strategy.
     */
    void mark_cells_for_coarsening_and_refinement(
        dealii::Triangulation<dim> &triangulation) const;

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
    //@}
  };

} // namespace ryujin
