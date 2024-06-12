//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <offline_data.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  /**
   * The MeshAdaptor class is responsible for
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
     * Prepare MeshAdaptor. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     */
    void prepare();

    /**
     * Analyze the given StateVector and decide - depending on adaptation
     * strategy - whether a mesh adaptation cycle should be performed.
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

  private:
    /**
     * @name Run time options
     */
    //@{

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const ParabolicSystem> parabolic_system_;
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;

    bool need_mesh_adaptation_;

    //@}
  };

} // namespace ryujin
