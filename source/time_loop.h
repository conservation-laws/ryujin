//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "hyperbolic_module.h"
#include "initial_values.h"
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

    using HyperbolicSystem = Description::HyperbolicSystem;

    using View = Description::template HyperbolicSystemView<dim, Number>;

    using ParabolicSystem = Description::ParabolicSystem;

    using ParabolicSolver = Description::template ParabolicSolver<dim, Number>;

    using ScalarNumber = View::ScalarNumber;

    static constexpr auto problem_dimension = View::problem_dimension;

    static constexpr auto n_precomputed_values = View::n_precomputed_values;

    using StateVector = View::StateVector;

    using ScalarVector = ryujin::ScalarVector<Number>;

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

    void compute_error(const StateVector &state_vector, Number t);

    void output(StateVector &state_vector,
                const std::string &name,
                Number t,
                unsigned int cycle);

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

    Number t_final_;
    std::vector<Number> t_refinements_;

    Number output_granularity_;

    bool enable_checkpointing_;
    bool enable_output_full_;
    bool enable_output_levelsets_;
    bool enable_compute_error_;
    bool enable_compute_quantities_;

    unsigned int output_checkpoint_multiplier_;
    unsigned int output_full_multiplier_;
    unsigned int output_levelsets_multiplier_;
    unsigned int output_quantities_multiplier_;

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
    Postprocessor<Description, dim, Number> postprocessor_;
    VTUOutput<Description, dim, Number> vtu_output_;
    Quantities<Description, dim, Number> quantities_;

    const unsigned int mpi_rank_;
    const unsigned int n_mpi_processes_;

    std::ofstream logfile_; /* log file */

    //@}
  };

} // namespace ryujin
