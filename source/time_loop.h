//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "hyperbolic_module.h"
#include "initial_values.h"
#include "mesh_adaptor.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "postprocessor.h"
#include "quantities.h"
#include "time_integrator.h"
#include "vtu_output.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

#include <fstream>

namespace ryujin
{

  /**
   * The high-level time loop driving the computation.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class TimeLoop final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    using ParabolicSystem = typename Description::ParabolicSystem;

    using ParabolicSolver =
        typename Description::template ParabolicSolver<dim, Number>;

    using ScalarNumber = typename View::ScalarNumber;

    static constexpr auto problem_dimension = View::problem_dimension;

    static constexpr auto n_precomputed_values = View::n_precomputed_values;

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
    TimeLoop(const MPI_Comm &mpi_comm);

    /**
     * Run the high-level time loop.
     */
    void run();

  protected:
    /**
     * @name Private methods for run()
     */
    //@{

    /**
     * Performs a resume operation. Given a @p base_name the function tries
     * to locate correponding checkpoint files and will read in the saved
     * state @p state_vector at saved time @p t with saved output cycle
     * @p output_cycle.
     */
    template <typename Callable>
    void read_checkpoint(StateVector &state_vector,
                         const std::string &base_name,
                         Number &t,
                         unsigned int &output_cycle,
                         const Callable &prepare_compute_kernels);

    /**
     * Write out a checkpoint to disk. Given a @p base_name and a current
     * state @p U at time @p t and output cycle @p output_cycle the
     * function writes out the state to disk using boost::archive for
     * serialization.
     *
     * @pre the state_vector needs to be prepared.
     */
    void write_checkpoint(const StateVector &state_vector,
                          const std::string &base_name,
                          const Number &t,
                          const unsigned int &output_cycle);

    /**
     * Perform a mesh adaptation cycle according to the selected strategy
     * in the MeshAdaptor class. The state vector is transferred to the new
     * discretization.
     */
    template <typename Callable>
    void adapt_mesh_and_transfer_state_vector(
        StateVector &state_vector, const Callable &prepare_compute_kernels);

    void compute_error(StateVector &state_vector, Number t);

    void output(StateVector &state_vector,
                const std::string &name,
                const Number t,
                const unsigned int cycle);

    void print_parameters(std::ostream &stream);
    void print_mpi_partition(std::ostream &stream);
    void print_memory_statistics(std::ostream &stream);
    void print_timers(std::ostream &stream);
    void print_throughput(unsigned int cycle,
                          Number t,
                          std::ostream &stream,
                          bool final_time = false);

    void print_info(const std::string &header);
    void print_head(const std::string &header,
                    const std::string &secondary,
                    std::ostream &stream);

    void print_cycle_statistics(unsigned int cycle,
                                Number t,
                                unsigned int output_cycle,
                                bool write_to_logfile = false,
                                bool final_time = false);
    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    std::string base_name_;

    std::string debug_filename_;

    Number t_final_;
    Number timer_granularity_;

    bool enable_checkpointing_;
    bool enable_output_full_;
    bool enable_output_levelsets_;
    bool enable_compute_error_;
    bool enable_compute_quantities_;
    bool enable_mesh_adaptivity_;

    unsigned int timer_checkpoint_multiplier_;
    unsigned int timer_output_full_multiplier_;
    unsigned int timer_output_levelsets_multiplier_;
    unsigned int timer_compute_quantities_multiplier_;

    std::vector<std::string> error_quantities_;
    bool error_normalize_;

    bool resume_;
    bool resume_at_time_zero_;

    Number terminal_update_interval_;
    bool terminal_show_rank_throughput_;

    //@}
    /**
     * @name Internal data:
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    std::map<std::string, dealii::Timer> computing_timer_;

    HyperbolicSystem hyperbolic_system_;
    ParabolicSystem parabolic_system_;
    Discretization<dim> discretization_;
    OfflineData<dim, Number> offline_data_;
    InitialValues<Description, dim, Number> initial_values_;
    HyperbolicModule<Description, dim, Number> hyperbolic_module_;
    ParabolicModule<Description, dim, Number> parabolic_module_;
    TimeIntegrator<Description, dim, Number> time_integrator_;
    MeshAdaptor<Description, dim, Number> mesh_adaptor_;
    Postprocessor<Description, dim, Number> postprocessor_;
    VTUOutput<Description, dim, Number> vtu_output_;
    Quantities<Description, dim, Number> quantities_;

    const unsigned int mpi_rank_;
    const unsigned int n_mpi_processes_;

    std::ofstream logfile_; /* log file */

    //@}
  };

} // namespace ryujin
