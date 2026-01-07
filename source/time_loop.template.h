//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include "mpi_ensemble_container.h"
#include "scope.h"
#include "state_vector.h"
#include "time_loop.h"
#include "version_info.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>

#ifdef WITH_OPENMP
#include "omp.h"
#endif

using namespace dealii;

namespace ryujin
{
  template <typename Description, int dim, typename Number>
  TimeLoop<Description, dim, Number>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("/A - TimeLoop")
      , mpi_ensemble_(mpi_comm,
                      [] {
                        if constexpr (has_n_mpi_ensembles_v<Description>)
                          return Description::n_mpi_ensembles();
                        else
                          return 1;
                      }())
      , hyperbolic_system_(mpi_ensemble_, "/B - Equation")
      , parabolic_system_(mpi_ensemble_, "/B - Equation")
      , discretization_(mpi_ensemble_, "/C - Discretization")
      , offline_data_(mpi_ensemble_, discretization_, "/D - OfflineData")
      , initial_values_(mpi_ensemble_,
                        "/E - InitialValues",
                        mpi_ensemble_,
                        offline_data_,
                        hyperbolic_system_,
                        parabolic_system_)
      , hyperbolic_module_(mpi_ensemble_,
                           computing_timer_,
                           offline_data_,
                           hyperbolic_system_,
                           initial_values_,
                           "/F - HyperbolicModule")
      , parabolic_module_(mpi_ensemble_,
                          computing_timer_,
                          offline_data_,
                          hyperbolic_system_,
                          parabolic_system_,
                          initial_values_,
                          "/G - ParabolicModule")
      , time_integrator_(mpi_ensemble_,
                         offline_data_,
                         hyperbolic_module_,
                         parabolic_module_,
                         "/H - TimeIntegrator")
      , mesh_adaptor_(mpi_ensemble_,
                      offline_data_,
                      hyperbolic_system_,
                      parabolic_system_,
                      hyperbolic_module_.initial_precomputed(),
                      hyperbolic_module_.alpha(),
                      "/I - MeshAdaptor")
      , solution_transfer_(mpi_ensemble_,
                           offline_data_,
                           hyperbolic_system_,
                           parabolic_system_,
                           "/I - MeshAdaptor")
      , postprocessor_(mpi_ensemble_,
                       offline_data_,
                       hyperbolic_system_,
                       parabolic_system_,
                       "/J - VTUOutput")
      , vtu_output_(mpi_ensemble_,
                    offline_data_,
                    hyperbolic_system_,
                    parabolic_system_,
                    postprocessor_,
                    hyperbolic_module_.initial_precomputed(),
                    hyperbolic_module_.alpha(),
                    mesh_adaptor_.smoothness_indicators(),
                    "/J - VTUOutput")
      , quantities_(mpi_ensemble_,
                    offline_data_,
                    hyperbolic_system_,
                    parabolic_system_,
                    "/K - Quantities")
  {
    base_name_ = "test";
    add_parameter("basename", base_name_, "Base name for all output files");

    t_final_ = Number(5.);
    add_parameter("final time", t_final_, "Final time");

    enforce_t_final_ = false;
    add_parameter("enforce final time",
                  enforce_t_final_,
                  "Boolean indicating whether the final time should be "
                  "enforced strictly. If set to true the last time step is "
                  "shortened so that the simulation ends precisely at t_final");

    timer_granularity_ = Number(0.01);
    add_parameter("timer granularity",
                  timer_granularity_,
                  "The timer granularity specifies the time interval after "
                  "which compute, output, postprocessing, and mesh adaptation "
                  "routines are run. This \"baseline tick\" is further "
                  "modified by the corresponding \"*_multiplier\" options");

    enable_output_full_ = false;
    add_parameter("enable output full",
                  enable_output_full_,
                  "Write out full pvtu records. The frequency is determined by "
                  "\"timer granularity\" and \"timer output full multiplier\"");

    enable_output_levelsets_ = false;
    add_parameter(
        "enable output levelsets",
        enable_output_levelsets_,
        "Write out levelsets pvtu records. The frequency is determined by "
        "\"timer granularity\" and \"timer output levelsets multiplier\"");

    enable_compute_error_ = false;
    add_parameter("enable compute error",
                  enable_compute_error_,
                  "Flag to control whether we compute the Linfty Linf_norm of "
                  "the difference to an analytic solution. Implemented only "
                  "for certain initial state configurations.");

    enable_compute_quantities_ = false;
    add_parameter(
        "enable compute quantities",
        enable_compute_quantities_,
        "Flag to control whether we compute quantities of interest. The "
        "frequency how often quantities are logged is determined by \"timer "
        "granularity\" and \"timer compute quantities multiplier\"");

    enable_mesh_adaptivity_ = false;
    add_parameter(
        "enable mesh adaptivity",
        enable_mesh_adaptivity_,
        "Flag to control whether we use an adaptive mesh refinement strategy. "
        "The frequency how often we query MeshAdaptor::analyze() for deciding "
        "on adapting the mesh is determined by \"timer granularity\" and "
        "\"timer mesh refinement multiplier\"");

    timer_output_full_multiplier_ = 1;
    add_parameter("timer output full multiplier",
                  timer_output_full_multiplier_,
                  "Multiplicative modifier applied to \"timer granularity\" "
                  "that determines the full pvtu writeout granularity");

    timer_output_levelsets_multiplier_ = 1;
    add_parameter("timer output levelsets multiplier",
                  timer_output_levelsets_multiplier_,
                  "Multiplicative modifier applied to \"timer granularity\" "
                  "that determines the levelsets pvtu writeout granularity");

    timer_compute_quantities_multiplier_ = 1;
    add_parameter(
        "timer compute quantities multiplier",
        timer_compute_quantities_multiplier_,
        "Multiplicative modifier applied to \"timer granularity\" that "
        "determines the writeout granularity for quantities of interest");

    std::copy(std::begin(View::component_names),
              std::end(View::component_names),
              std::back_inserter(error_quantities_));

    add_parameter("error quantities",
                  error_quantities_,
                  "List of conserved quantities used in the computation of the "
                  "error norms.");

    error_normalize_ = true;
    add_parameter("error normalize",
                  error_normalize_,
                  "Flag to control whether the error should be normalized by "
                  "the corresponding norm of the analytic solution.");

    resume_ = false;
    add_parameter("resume", resume_, "Resume an interrupted computation");

    resume_at_time_zero_ = false;
    add_parameter("resume at time zero",
                  resume_at_time_zero_,
                  "Resume from the latest checkpoint but set the time to t=0.");

    terminal_update_interval_ = 5;
    add_parameter("terminal update interval",
                  terminal_update_interval_,
                  "Number of seconds after which output statistics are "
                  "recomputed and printed on the terminal. Setting the "
                  "interval to zero disables terminal output.");

    terminal_correct_for_hypertreadhing_ = true;
    add_parameter(
        "terminal correct for hyperthreading",
        terminal_correct_for_hypertreadhing_,
        "If set to true, the CPU throughput is corrected by dividing the total "
        "consumed CPU time by a factor of 2. This correction is only active if "
        "the number of threads (per MPI rank) is 2.");

    checkpoint_update_interval_ = 0;
    add_parameter(
        "checkpoint update interval",
        checkpoint_update_interval_,
        "Number of seconds after which a new checkpoint is written out to "
        "disk. Setting the interval to zero disables checkpointing.");

    debug_command_ = "";
    add_parameter("debug command",
                  debug_command_,
                  "If set to a nonempty string then the host environment's "
                  "command processor is invoked via std::system() with the "
                  "specified string as command parameter.");

    debug_filename_ = "";
    add_parameter("debug filename",
                  debug_filename_,
                  "If set to a nonempty string then we output the contents of "
                  "this file at the end. This is mainly useful in the "
                  "testsuite to output files we wish to compare");
  }


  /*
   * ---------------------------------------------------------------------------
   * Setup and main loop:
   * ---------------------------------------------------------------------------
   */


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::run()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::run()" << std::endl;
#endif

    {
      base_name_ensemble_ = base_name_;
      if (mpi_ensemble_.n_ensembles() > 1) {
        print_info("setting up MPI ensemble");
        unsigned int digits =
            dealii::Utilities::needed_digits(mpi_ensemble_.n_ensembles() - 1);
        base_name_ensemble_ +=
            "-ensemble_" +
            dealii::Utilities::int_to_string(mpi_ensemble_.ensemble(), digits);
      }
    }

    /* Attach log file and record runtime parameters: */

    if (mpi_ensemble_.world_rank() == 0)
      logfile_.open(base_name_ + ".log");

    print_parameters(logfile_);

    /*
     * Prepare data structures:
     */

    Number t = 0.;
    unsigned int timer_cycle = 0;
    StateVector state_vector;

    /* Create a small lambda for preparing compute kernels: */
    const auto prepare_compute_kernels = [&]() {
      print_info("preparing compute kernels");

      offline_data_.prepare(problem_dimension, n_precomputed_values);

      hyperbolic_module_.prepare();
      parabolic_module_.prepare();
      time_integrator_.prepare();
      mesh_adaptor_.prepare(/*needs current timepoint*/ t);
      postprocessor_.prepare();
      vtu_output_.prepare();
      quantities_.prepare(base_name_ensemble_);
      print_mpi_partition(logfile_);

      if (mpi_ensemble_.ensemble_rank() == 0)
        n_global_dofs_ = dealii::Utilities::MPI::sum(
            offline_data_.dof_handler().n_dofs(),
            mpi_ensemble_.ensemble_leader_communicator());
    };

    {
      Scope scope(computing_timer_, "(re)initialize data structures");
      print_info("initializing data structures");

      if (resume_) {
        print_info("resume: reading mesh and loading state vector");

        read_checkpoint(state_vector,
                        base_name_ensemble_,
                        t,
                        timer_cycle,
                        prepare_compute_kernels);

        if (resume_at_time_zero_) {
          /* Reset the current time t and the output cycle count to zero: */
          t = 0.;
          timer_cycle = 0;
        }

      } else {
        print_info("creating mesh and interpolating initial values");

        discretization_.prepare(base_name_ensemble_);

        prepare_compute_kernels();

        hyperbolic_module_.reinit_state_vector(state_vector);
        parabolic_module_.reinit_state_vector(state_vector);
        {
          Scope scope(computing_timer_,
                      "time step [X]   - interpolate data vectors");
          std::get<0>(state_vector) =
              initial_values_.get().interpolate_hyperbolic_vector();
        }
        Vectors::debug_poison_invalid_values(state_vector, offline_data_);
      }
    }

    /* Prepare the state vector for time stepping. */
    time_integrator_.prepare_state_vector(state_vector, t);

    /*
     * The honorable main loop:
     */

    Number last_terminal_output = terminal_update_interval_ == Number(0.)
                                      ? std::numeric_limits<Number>::max()
                                      : Number(0.);
    Number last_checkpoint = checkpoint_update_interval_ == Number(0.)
                                 ? std::numeric_limits<Number>::max()
                                 : Number(0.);

    print_info("entering main loop");
    computing_timer_["time loop"].start();

    constexpr Number relax =
        Number(1.) - Number(10.) * std::numeric_limits<Number>::epsilon();

    unsigned int cycle = 1;
    for (;; ++cycle) {

#ifdef DEBUG_OUTPUT
      std::cout << "\n\n###   cycle = " << cycle << "   ###\n\n" << std::endl;
#endif

      /* Accumulate quantities of interest: */

      if (enable_compute_quantities_) {
        Scope scope(computing_timer_,
                    "time step [X]   - accumulate quantities");
        quantities_.accumulate(state_vector, t);
      }

      /* Perform output tasks whenever we reach a timer tick: */

      if (t >= relax * timer_cycle * timer_granularity_) {
        if (enable_compute_error_) {
          /*
           * FIXME: We interpolate the analytic solution at every timer
           * tick. If we happen to actually not output anything then this
           * is terribly inefficient...
           */

          StateVector analytic;
          {
            Scope scope(computing_timer_,
                        "time step [X]   - interpolate data vectors");
            hyperbolic_module_.reinit_state_vector(analytic);
            parabolic_module_.reinit_state_vector(analytic);
            std::get<0>(analytic) =
                initial_values_.get().interpolate_hyperbolic_vector(t);
          }

          time_integrator_.prepare_state_vector(analytic, t);

          output(analytic,
                 base_name_ensemble_ + "-analytic_solution",
                 t,
                 timer_cycle);
        }

        output(state_vector, base_name_ensemble_ + "-solution", t, timer_cycle);

        if (enable_compute_quantities_ &&
            (timer_cycle % timer_compute_quantities_multiplier_ == 0)) {
          Scope scope(computing_timer_,
                      "time step [X]   - write out quantities");
          quantities_.write_out(state_vector, t, timer_cycle);
        }

        ++timer_cycle;
      }

      /* Break if we have reached the final time. */

      if (t >= relax * t_final_)
        break;

      /* Peform a mesh adaptation cycle: */

      if (enable_mesh_adaptivity_) {
        {
          Scope scope(computing_timer_,
                      "time step [X]   - analyze for mesh adaptation");

          mesh_adaptor_.analyze(state_vector, t, cycle);
        }

        if (mesh_adaptor_.need_mesh_adaptation()) {
          Scope scope_1(computing_timer_, "(re)initialize data structures");
          Scope scope_2(computing_timer_,
                        "time step [X]   - perform mesh adaptation");
          print_info("performing mesh adaptation");

          adapt_mesh_and_transfer_state_vector(state_vector,
                                               prepare_compute_kernels);

          /* Prepare the state vector for time stepping. */
          time_integrator_.prepare_state_vector(state_vector, t);
        }
      }

      /* Perform a time step: */

      const auto tau = time_integrator_.step(
          state_vector,
          t,
          enforce_t_final_
              ? std::min(t_final_, timer_cycle * timer_granularity_)
              : std::numeric_limits<Number>::max());

      t += tau;

      time_integrator_.prepare_state_vector(state_vector, t);

      /* Synchronize wall time: */

      auto wall_time = computing_timer_["time loop"].wall_time();
      {
        Scope scope(computing_timer_,
                    "time step [X] _ - synchronization barriers");
        wall_time =
            Utilities::MPI::max(wall_time, mpi_ensemble_.world_communicator());
      }

      /* Print and record cycle statistics: */

      const bool write_to_log_file =
          (terminal_update_interval_ != Number(0.)) && /* suppress output */
          (t >= relax * timer_cycle * timer_granularity_);

      const bool update_terminal =
          (wall_time >= last_terminal_output + terminal_update_interval_);

      if (write_to_log_file || update_terminal) {
        Scope scope(computing_timer_,
                    "time step [X] _ - synchronization barriers");
        print_cycle_statistics(cycle,
                               t,
                               timer_cycle,
                               last_checkpoint,
                               /*logfile*/ write_to_log_file);
        last_terminal_output = wall_time;
      }

      const bool update_checkpoint =
          (wall_time >= last_checkpoint + checkpoint_update_interval_);

      if (update_checkpoint) {
        Scope scop(computing_timer_, "time step [X]   - perform checkpointing");

        print_info("scheduling checkpointing");
        write_checkpoint(state_vector, base_name_ensemble_, t, timer_cycle);
        last_checkpoint = wall_time;
      }
    } /* end of loop */

    /* We have actually performed one cycle less. */
    --cycle;

    if (checkpoint_update_interval_ != Number(0.)) {
      Scope scope(computing_timer_, "time step [X]   - perform checkpointing");

      print_info("scheduling checkpointing");
      write_checkpoint(state_vector, base_name_ensemble_, t, timer_cycle);
    }

    computing_timer_["time loop"].stop();

    if (terminal_update_interval_ != Number(0.)) {
      /* Write final timing statistics to screen and logfile: */
      print_cycle_statistics(cycle,
                             t,
                             timer_cycle,
                             last_checkpoint,
                             /*logfile*/ true,
                             /*final*/ true);
    }

    if (enable_compute_error_) {
      /* Output final error: */
      compute_error(state_vector, t);
    }

    /*
     *
     */

    if (mpi_ensemble_.world_rank() == 0) {
      if (debug_command_ != "") {
        auto result [[maybe_unused]] = std::system(debug_command_.c_str());
      }

      if (debug_filename_ != "") {
        std::ifstream f(debug_filename_);
        if (f.is_open())
          std::cout << f.rdbuf();
      }
    }

#ifdef WITH_VALGRIND
    CALLGRIND_DUMP_STATS;
#endif
  }


  /*
   * ---------------------------------------------------------------------------
   * Checkpointing, VTK output, and compute error:
   * ---------------------------------------------------------------------------
   */


  template <typename Description, int dim, typename Number>
  template <typename Callable>
  void TimeLoop<Description, dim, Number>::read_checkpoint(
      StateVector &state_vector,
      const std::string &base_name,
      Number &t,
      unsigned int &timer_cycle,
      const Callable &prepare_compute_kernels)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::read_checkpoint()" << std::endl;
#endif

    /*
     * Initialize discretization, read in the mesh, and initialize everything:
     */

#if DEAL_II_VERSION_GTE(9, 6, 0)
    discretization_.refinement() = 0; /* do not refine */
    discretization_.prepare(base_name);
    discretization_.triangulation().load(base_name + "-checkpoint.mesh");

#else
    AssertThrow(false,
                dealii::ExcMessage("write_checkpoint() is not available with "
                                   "deal.II versions prior to 9.6.0"));
#endif

    prepare_compute_kernels();

    /*
     * Read in and broadcast metadata:
     */

    std::string name = base_name + "-checkpoint";

    unsigned int transfer_handle;
    if (mpi_ensemble_.ensemble_rank() == 0) {
      std::string meta = name + ".metadata";

      std::ifstream file(meta, std::ios::binary);
      boost::archive::binary_iarchive ia(file);
      ia >> t >> timer_cycle >> transfer_handle;
    }

    int ierr;
    if constexpr (std::is_same_v<Number, double>)
      ierr = MPI_Bcast(
          &t, 1, MPI_DOUBLE, 0, mpi_ensemble_.ensemble_communicator());
    else
      ierr =
          MPI_Bcast(&t, 1, MPI_FLOAT, 0, mpi_ensemble_.ensemble_communicator());
    AssertThrowMPI(ierr);

    ierr = MPI_Bcast(&timer_cycle,
                     1,
                     MPI_UNSIGNED,
                     0,
                     mpi_ensemble_.ensemble_communicator());
    AssertThrowMPI(ierr);

    ierr = MPI_Bcast(&transfer_handle,
                     1,
                     MPI_UNSIGNED,
                     0,
                     mpi_ensemble_.ensemble_communicator());
    AssertThrowMPI(ierr);

    /* Now read in the state vector: */

    hyperbolic_module_.reinit_state_vector(state_vector);
    parabolic_module_.reinit_state_vector(state_vector);

    solution_transfer_.set_handle(transfer_handle);
    solution_transfer_.project(state_vector);
    solution_transfer_.reset_handle();
    Vectors::debug_poison_invalid_values(state_vector, offline_data_);

    time_integrator_.prepare_state_vector(state_vector, t);
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::write_checkpoint(
      const StateVector &state_vector,
      const std::string &base_name,
      const Number &t,
      const unsigned int &timer_cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::write_checkpoint()" << std::endl;
#endif

    solution_transfer_.prepare_projection(state_vector);
    const auto transfer_handle = solution_transfer_.get_handle();
    solution_transfer_.reset_handle();

    std::string name = base_name + "-checkpoint";

    if (mpi_ensemble_.ensemble_rank() == 0) {
      for (const std::string suffix :
           {".mesh", ".mesh_fixed.data", ".mesh.info", ".metadata"})
        if (std::filesystem::exists(name + suffix))
          std::filesystem::rename(name + suffix, name + suffix + "~");
    }

#if DEAL_II_VERSION_GTE(9, 6, 0)
    const auto &triangulation = discretization_.triangulation();
    triangulation.save(name + ".mesh");

#else
    AssertThrow(false,
                dealii::ExcMessage("write_checkpoint() is not available with "
                                   "deal.II versions prior to 9.6.0"));
#endif

    /*
     * Now, write out metadata on rank 0:
     */

    if (mpi_ensemble_.ensemble_rank() == 0) {
      std::string meta = name + ".metadata";
      std::ofstream file(meta, std::ios::binary | std::ios::trunc);
      boost::archive::binary_oarchive oa(file);
      oa << t << timer_cycle << transfer_handle;
    }

    const int ierr = MPI_Barrier(mpi_ensemble_.ensemble_communicator());
    AssertThrowMPI(ierr);
  }


  template <typename Description, int dim, typename Number>
  template <typename Callable>
  void TimeLoop<Description, dim, Number>::adapt_mesh_and_transfer_state_vector(
      StateVector &state_vector, const Callable &prepare_compute_kernels)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::adapt_mesh_and_transfer_state_vector()"
              << std::endl;
#endif

    AssertThrow(mpi_ensemble_.n_ensembles() == 1, dealii::ExcNotImplemented());

    /*
     * Mark cells for coarsening and refinement and set up triangulation:
     */

    auto &triangulation = discretization_.triangulation();
    mesh_adaptor_.mark_cells_for_coarsening_and_refinement(triangulation);

    triangulation.prepare_coarsening_and_refinement();

    solution_transfer_.prepare_projection(state_vector);

    /* Execute mesh adaptation and project old state to new state vector: */

    triangulation.execute_coarsening_and_refinement();
    prepare_compute_kernels();

    hyperbolic_module_.reinit_state_vector(state_vector);
    parabolic_module_.reinit_state_vector(state_vector);

    solution_transfer_.project(state_vector);
    solution_transfer_.reset_handle();
    Vectors::debug_poison_invalid_values(state_vector, offline_data_);
  }


  template <typename Description, int dim, typename Number>
  void
  TimeLoop<Description, dim, Number>::compute_error(StateVector &state_vector,
                                                    const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::compute_error()" << std::endl;
#endif

    Vector<Number> difference_per_cell(
        discretization_.triangulation().n_active_cells());

    Number linf_norm = 0.;
    Number l1_norm = 0;
    Number l2_norm = 0;

    const auto analytic_U =
        initial_values_.get().interpolate_hyperbolic_vector(t);
    const auto &U = std::get<0>(state_vector);

    ScalarVector analytic_component;
    ScalarVector error_component;
    analytic_component.reinit(offline_data_.scalar_partitioner());
    error_component.reinit(offline_data_.scalar_partitioner());

    /* Loop over all selected components: */
    for (const auto &entry : error_quantities_) {
      const auto &names = View::component_names;
      const auto pos = std::find(std::begin(names), std::end(names), entry);
      if (pos == std::end(names)) {
        AssertThrow(
            false,
            dealii::ExcMessage("Unknown component name »" + entry + "«"));
        __builtin_trap();
      }

      const auto index = std::distance(std::begin(names), pos);

      analytic_U.extract_component(analytic_component, index);

      /* Compute norms of analytic solution: */

      Number linf_norm_analytic = 0.;
      Number l1_norm_analytic = 0.;
      Number l2_norm_analytic = 0.;

      if (error_normalize_) {
        linf_norm_analytic = analytic_component.linfty_norm();

        VectorTools::integrate_difference(
            discretization_.mapping(),
            offline_data_.dof_handler(),
            analytic_component,
            Functions::ZeroFunction<dim, Number>(),
            difference_per_cell,
            discretization_.quadrature_high_order(),
            VectorTools::L1_norm);

        l1_norm_analytic =
            Utilities::MPI::sum(difference_per_cell.l1_norm(),
                                mpi_ensemble_.ensemble_communicator());

        VectorTools::integrate_difference(
            discretization_.mapping(),
            offline_data_.dof_handler(),
            analytic_component,
            Functions::ZeroFunction<dim, Number>(),
            difference_per_cell,
            discretization_.quadrature_high_order(),
            VectorTools::L2_norm);

        l2_norm_analytic = Number(std::sqrt(
            Utilities::MPI::sum(std::pow(difference_per_cell.l2_norm(), 2),
                                mpi_ensemble_.ensemble_communicator())));
      }

      /* Compute norms of error: */

      U.extract_component(error_component, index);
      /* Populate constrained dofs due to periodicity: */
      offline_data_.affine_constraints().distribute(error_component);
      error_component.update_ghost_values();
      error_component -= analytic_component;

      const Number linf_norm_error = error_component.linfty_norm();

      VectorTools::integrate_difference(discretization_.mapping(),
                                        offline_data_.dof_handler(),
                                        error_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        discretization_.quadrature_high_order(),
                                        VectorTools::L1_norm);

      const Number l1_norm_error = Utilities::MPI::sum(
          difference_per_cell.l1_norm(), mpi_ensemble_.ensemble_communicator());

      VectorTools::integrate_difference(discretization_.mapping(),
                                        offline_data_.dof_handler(),
                                        error_component,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        discretization_.quadrature_high_order(),
                                        VectorTools::L2_norm);

      const Number l2_norm_error = Number(std::sqrt(
          Utilities::MPI::sum(std::pow(difference_per_cell.l2_norm(), 2),
                              mpi_ensemble_.ensemble_communicator())));

      if (error_normalize_) {
        linf_norm += linf_norm_error / linf_norm_analytic;
        l1_norm += l1_norm_error / l1_norm_analytic;
        l2_norm += l2_norm_error / l2_norm_analytic;
      } else {
        linf_norm += linf_norm_error;
        l1_norm += l1_norm_error;
        l2_norm += l2_norm_error;
      }
    }

    if (mpi_ensemble_.ensemble_rank() != 0)
      return;

    /*
     * Sum up over all participating MPI ranks. Note: we only perform this
     * operation on "peer" ranks zero:
     */

    if (mpi_ensemble_.n_ensembles() > 1) {
      linf_norm = Utilities::MPI::sum(
          linf_norm, mpi_ensemble_.ensemble_leader_communicator());
      l1_norm = Utilities::MPI::sum(
          l1_norm, mpi_ensemble_.ensemble_leader_communicator());
      l2_norm = Utilities::MPI::sum(
          l2_norm, mpi_ensemble_.ensemble_leader_communicator());
    }

    if (mpi_ensemble_.world_rank() != 0)
      return;

    logfile_ << std::endl << "Computed errors:" << std::endl << std::endl;
    logfile_ << std::setprecision(16);

    std::string description =
        error_normalize_ ? "Normalized consolidated" : "Consolidated";

    logfile_ << description + " Linf, L1, and L2 errors at final time \n";
    logfile_ << std::setprecision(16);
    logfile_ << "#dofs = " << n_global_dofs_ << std::endl;
    logfile_ << "t     = " << t << std::endl;
    logfile_ << "Linf  = " << linf_norm << std::endl;
    logfile_ << "L1    = " << l1_norm << std::endl;
    logfile_ << "L2    = " << l2_norm << std::endl;

    std::cout << description + " Linf, L1, and L2 errors at final time \n";
    std::cout << std::setprecision(16);
    std::cout << "#dofs = " << n_global_dofs_ << std::endl;
    std::cout << "t     = " << t << std::endl;
    std::cout << "Linf  = " << linf_norm << std::endl;
    std::cout << "L1    = " << l1_norm << std::endl;
    std::cout << "L2    = " << l2_norm << std::endl;
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::output(StateVector &state_vector,
                                                  const std::string &name,
                                                  const Number t,
                                                  const unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeLoop<dim, Number>::output(t = " << t << ")" << std::endl;
#endif

    const bool do_full_output =
        (cycle % timer_output_full_multiplier_ == 0) && enable_output_full_;
    const bool do_levelsets =
        (cycle % timer_output_levelsets_multiplier_ == 0) &&
        enable_output_levelsets_;

    /* There is nothing to do: */
    if (!(do_full_output || do_levelsets))
      return;

    /* Data output: */

    Scope scope(computing_timer_, "time step [X]   - perform vtu output");
    print_info("scheduling output");

    postprocessor_.compute(state_vector);
    /*
     * Workaround: Manually reset bounds during the first output cycle
     * (which is often just a uniform flow field) to obtain a better
     * normailization:
     */
    if (cycle == 0)
      postprocessor_.reset_bounds();

    /* Make sure we have a valid vector of smoothness indicators. */
    mesh_adaptor_.compute_smoothness_indicators(state_vector);

    vtu_output_.schedule_output(
        state_vector, name, t, cycle, do_full_output, do_levelsets);
  }


  /*
   * ---------------------------------------------------------------------------
   * Output and logging related functions:
   * ---------------------------------------------------------------------------
   */


  template <typename Description, int dim, typename Number>
  void
  TimeLoop<Description, dim, Number>::print_parameters(std::ostream &stream)
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    /* Output commit and library information: */

    print_revision_and_version(stream);

    /* Print run time parameters: */

    stream << std::endl << "Run time parameters:" << std::endl << std::endl;
    ParameterAcceptor::prm.print_parameters(
        stream, ParameterHandler::OutputStyle::ShortPRM);
    stream << std::endl;

    /* Also print out parameters to a prm file: */

    std::ofstream output(base_name_ + "-parameters.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::ShortPRM);
  }


  template <typename Description, int dim, typename Number>
  void
  TimeLoop<Description, dim, Number>::print_mpi_partition(std::ostream &stream)
  {
    /*
     * Fixme: this conversion to double is really not elegant. We should
     * improve the Utilities::MPI::min_max_avg function in deal.II to
     * handle different data types
     */

    // NOLINTBEGIN
    std::vector<double> values = {
        (double)offline_data_.n_export_indices(),
        (double)offline_data_.n_locally_internal(),
        (double)offline_data_.n_locally_owned(),
        (double)offline_data_.n_locally_relevant(),
        (double)offline_data_.n_export_indices() /
            (double)offline_data_.n_locally_relevant(),
        (double)offline_data_.n_locally_internal() /
            (double)offline_data_.n_locally_relevant(),
        (double)offline_data_.n_locally_owned() /
            (double)offline_data_.n_locally_relevant()};
    // NOLINTEND

    const auto data =
        Utilities::MPI::min_max_avg(values, mpi_ensemble_.world_communicator());

    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::ostringstream output;

    unsigned int n =
        dealii::Utilities::needed_digits(mpi_ensemble_.n_world_ranks());

    const auto print_snippet = [&output, n](const std::string &name,
                                            const auto &values) {
      output << name << ": ";
      // NOLINTBEGIN
      output << std::setw(9) << (unsigned int)values.min          //
             << " [p" << std::setw(n) << values.min_index << "] " //
             << std::setw(9) << (unsigned int)values.avg << " "   //
             << std::setw(9) << (unsigned int)values.max          //
             << " [p" << std::setw(n) << values.max_index << "]"; //
      // NOLINTEND
    };

    const auto print_percentages = [&output, n](const auto &percentages) {
      output << std::endl << "                  ";
      output << "  (" << std::setw(3) << std::setprecision(2)
             << percentages.min * 100 << "% )"
             << " [p" << std::setw(n) << percentages.min_index << "] "
             << "   (" << std::setw(3) << std::setprecision(2)
             << percentages.avg * 100 << "% )"
             << " "
             << "   (" << std::setw(3) << std::setprecision(2)
             << percentages.max * 100 << "% )"
             << " [p" << std::setw(n) << percentages.max_index << "]";
    };

    output << std::endl << std::endl << "Partition:   ";
    print_snippet("exp", data[0]);
    print_percentages(data[4]);

    output << std::endl << "             ";
    print_snippet("int", data[1]);
    print_percentages(data[5]);

    output << std::endl << "             ";
    print_snippet("own", data[2]);
    print_percentages(data[6]);

    output << std::endl << "             ";
    print_snippet("rel", data[3]);

    stream << output.str() << std::endl;
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::print_info(const std::string &header)
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::cout << "[INFO] " << header << std::endl;
  }


  template <typename Description, int dim, typename Number>
  void
  TimeLoop<Description, dim, Number>::print_head(const std::string &header,
                                                 const std::string &secondary,
                                                 std::ostream &stream)
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    const int header_size = header.size();
    const auto padded_header =
        std::string(std::max(0, 34 - header_size) / 2, ' ') + header +
        std::string(std::max(0, 35 - header_size) / 2, ' ');

    const int secondary_size = secondary.size();
    const auto padded_secondary =
        std::string(std::max(0, 34 - secondary_size) / 2, ' ') + secondary +
        std::string(std::max(0, 35 - secondary_size) / 2, ' ');

    /* clang-format off */
    stream << "\n";
    stream << "    ####################################################\n";
    stream << "    #########"     <<  padded_header   <<     "#########\n";
    stream << "    #########"     << padded_secondary <<     "#########\n";
    stream << "    ####################################################\n";
    stream << std::endl;
    /* clang-format on */
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::print_information(
      unsigned int timer_cycle,
      Number last_checkpoint,
      std::ostream &stream,
      bool final_time)
  {
    static const std::string vectorization_name = [] {
      constexpr auto width = VectorizedArray<Number>::size();

      std::string result;
      if (width == 1)
        result = "scalar ";
      else
        result = std::to_string(width * 8 * sizeof(Number)) + "bit packed ";

      if constexpr (std::is_same_v<Number, double>)
        return result + "double";
      else if constexpr (std::is_same_v<Number, float>)
        return result + "float";
      else
        __builtin_trap();
    }();

    stream << "Information: (HYP) " << hyperbolic_system_.get().problem_name;
    if constexpr (!ParabolicSystem::is_identity) {
      stream << "\n             (PAR) " << parabolic_system_.get().problem_name;
    }
    stream << "\n             [" << base_name_ << "] ";
    if (mpi_ensemble_.n_ensembles() > 1) {
      stream << mpi_ensemble_.n_ensembles() << " ensembles ";
    }
    stream << "with "                                      //
           << n_global_dofs_ << " Qdofs on "               //
           << mpi_ensemble_.n_world_ranks() << " ranks / " //
#ifdef WITH_OPENMP
           << omp_get_max_threads() << " omp"
#else
           << "[openmp disabled]"
#endif
#ifdef WITH_DEAL_II_THREADS
           << " (" << MultithreadInfo::n_threads() << " dealii)"
#endif
           << " threads <" << vectorization_name << ">\n";

    stream << "             Last output cycle "                    //
           << timer_cycle - 1                                      //
           << " at t = " << timer_granularity_ * (timer_cycle - 1) //
           << "  [ log ";

    if (enable_output_full_)
      stream << "full ";
    if (enable_output_levelsets_)
      stream << "levelsets ";
    if (enable_compute_quantities_)
      stream << "quantities ";

    stream << "]\n";

    if (checkpoint_update_interval_ != Number(0.)) {
      const auto wall_time =
          Utilities::MPI::min_max_avg(computing_timer_["time loop"].wall_time(),
                                      mpi_ensemble_.world_communicator());

      if (final_time) {
        stream << "             Last checkpoint at FINAL TIME\n";
      } else {
        stream << "             Last checkpoint at wall time "          //
               << std::setprecision(2) << std::fixed << last_checkpoint //
               << "s  (" << std::setprecision(0)
               << std::max(0., wall_time.max - last_checkpoint)
               << "s ago, interval " << checkpoint_update_interval_ << "s)\n";
      }
    }
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::print_memory_statistics(
      std::ostream &stream)
  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);

    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(
        stats.VmRSS / 1024., mpi_ensemble_.world_communicator());

    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::ostringstream output;

    unsigned int n =
        dealii::Utilities::needed_digits(mpi_ensemble_.n_world_ranks());

    output << "\nMemory:      [MiB]"                          //
           << std::setw(8) << data.min                        //
           << " [p" << std::setw(n) << data.min_index << "] " //
           << std::setw(8) << data.avg << " "                 //
           << std::setw(8) << data.max                        //
           << " [p" << std::setw(n) << data.max_index << "]"; //

    stream << output.str() << std::endl;
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::print_timers(std::ostream &stream)
  {
    std::vector<std::ostringstream> output(computing_timer_.size());

    const auto equalize = [&]() {
      const auto ptr =
          std::max_element(output.begin(),
                           output.end(),
                           [](const auto &left, const auto &right) {
                             return left.str().length() < right.str().length();
                           });
      const auto length = ptr->str().length();
      for (auto &it : output)
        it << std::string(length - it.str().length() + 1, ' ');
    };

    const auto print_wall_time = [&](auto &timer, auto &stream) {
      const auto wall_time = Utilities::MPI::min_max_avg(
          timer.wall_time(), mpi_ensemble_.world_communicator());

      constexpr auto eps = std::numeric_limits<double>::epsilon();
      /*
       * Cut off at 99.9% to avoid silly percentages cluttering up the
       * output.
       */
      const auto skew_negative = std::max(
          100. * (wall_time.min - wall_time.avg) / wall_time.avg - eps, -99.9);
      const auto skew_positive = std::min(
          100. * (wall_time.max - wall_time.avg) / wall_time.avg + eps, 99.9);

      stream << std::setprecision(2) << std::fixed << std::setw(9)
             << wall_time.avg << "s [sk: " << std::setprecision(1)
             << std::setw(5) << std::fixed << skew_negative << "%/"
             << std::setw(4) << std::fixed << skew_positive << "%]";
      unsigned int n =
          dealii::Utilities::needed_digits(mpi_ensemble_.n_world_ranks());
      stream << " [p" << std::setw(n) << wall_time.min_index << "/"
             << wall_time.max_index << "]";
    };

    const auto cpu_time_statistics =
        Utilities::MPI::min_max_avg(computing_timer_["time loop"].cpu_time(),
                                    mpi_ensemble_.world_communicator());
    const double total_cpu_time = cpu_time_statistics.sum;

    const auto print_cpu_time =
        [&](auto &timer, auto &stream, bool percentage) {
          const auto cpu_time = Utilities::MPI::min_max_avg(
              timer.cpu_time(), mpi_ensemble_.world_communicator());

          stream << std::setprecision(2) << std::fixed << std::setw(12)
                 << cpu_time.sum << "s ";

          if (percentage)
            stream << "(" << std::setprecision(1) << std::setw(4)
                   << 100. * cpu_time.sum / total_cpu_time << "%)";
        };

    auto jt = output.begin();
    for (auto &it : computing_timer_)
      *jt++ << "  " << it.first;
    equalize();

    jt = output.begin();
    for (auto &it : computing_timer_)
      print_wall_time(it.second, *jt++);
    equalize();

    jt = output.begin();
    bool compute_percentages = false;
    for (auto &it : computing_timer_) {
      print_cpu_time(it.second, *jt++, compute_percentages);
      if (it.first.starts_with("time loop"))
        compute_percentages = true;
    }
    equalize();

    if (mpi_ensemble_.world_rank() != 0)
      return;

    stream << std::endl << "Timer statistics:\n";
    for (auto &it : output)
      stream << it.str() << std::endl;
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::print_throughput(
      unsigned int cycle, Number t, std::ostream &stream, bool final_time)
  {
    /*
     * Fixme: The global state kept in this function should be refactored
     * into its own class object.
     */
    static struct Data {
      unsigned int cycle = 0;
      double t = 0.;
      double cpu_time_sum = 0.;
      double cpu_time_avg = 0.;
      double cpu_time_min = 0.;
      double cpu_time_max = 0.;
      double wall_time = 0.;
    } previous, current;

    static double time_per_second_exp = 0.;

    /* Update statistics: */

    {
      previous = current;

      current.cycle = cycle;
      current.t = t;

      const auto wall_time_statistics =
          Utilities::MPI::min_max_avg(computing_timer_["time loop"].wall_time(),
                                      mpi_ensemble_.world_communicator());
      current.wall_time = wall_time_statistics.max;

      const auto cpu_time_statistics =
          Utilities::MPI::min_max_avg(computing_timer_["time loop"].cpu_time(),
                                      mpi_ensemble_.world_communicator());
      current.cpu_time_sum = cpu_time_statistics.sum;
      current.cpu_time_avg = cpu_time_statistics.avg;
      current.cpu_time_min = cpu_time_statistics.min;
      current.cpu_time_max = cpu_time_statistics.max;
    }

    if (final_time)
      previous = Data();

    /* Take averages: */

    double delta_cycles = current.cycle - previous.cycle;
    const double cycles_per_second =
        delta_cycles / (current.wall_time - previous.wall_time);

    const auto efficiency = time_integrator_.efficiency();
    const auto n_dofs = static_cast<double>(n_global_dofs_);

    double wall_m_dofs_per_sec = delta_cycles * n_dofs * efficiency / 1.e6 /
                                 (current.wall_time - previous.wall_time);

    double cpu_m_dofs_per_sec = delta_cycles * n_dofs * efficiency / 1.e6 /
                                (current.cpu_time_sum - previous.cpu_time_sum);

    /* Determine whether we fudge the CPU timings: */
    const bool fudge_cpu_timings = terminal_correct_for_hypertreadhing_ &&
#ifdef WITH_OPENMP
                                   (omp_get_max_threads() == 2);
#else
                                   false;
#endif

    if (fudge_cpu_timings)
      cpu_m_dofs_per_sec *= 2.;

    double cpu_time_skew = (current.cpu_time_max - current.cpu_time_min - //
                            previous.cpu_time_max + previous.cpu_time_min) /
                           delta_cycles;
    /* avoid printing small negative numbers: */
    cpu_time_skew = std::max(0., cpu_time_skew);

    const double cpu_time_skew_percentage =
        cpu_time_skew * delta_cycles /
        (current.cpu_time_avg - previous.cpu_time_avg);

    const double delta_time =
        (current.t - previous.t) / (current.cycle - previous.cycle);
    const double time_per_second =
        (current.t - previous.t) / (current.wall_time - previous.wall_time);

    /* Print Jean-Luc and Martin metrics: */

    std::ostringstream output;

    /* clang-format off */
    output << std::endl;

    output << "Throughput:\n  "
           << (fudge_cpu_timings ? "CPU*: " : "CPU : ")
           << std::setprecision(4) << std::fixed << cpu_m_dofs_per_sec
           << " MQ/s  ("
           << std::scientific << 1. / cpu_m_dofs_per_sec * 1.e-6
           << " s/Qdof/substep)" << std::endl;

    output << "        [cpu time skew: "
           << std::setprecision(2) << std::scientific << cpu_time_skew
           << "s/cycle ("
           << std::setprecision(1) << std::setw(4) << std::setfill(' ') << std::fixed
           << 100. * cpu_time_skew_percentage
           << "%)]" << std::endl;

    output << "  WALL: "
           << std::setprecision(4) << std::fixed << wall_m_dofs_per_sec
           << " MQ/s  ("
           << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6
           << " s/Qdof/substep)  ("
           << std::setprecision(2) << std::fixed << cycles_per_second
           << " cycles/s)" << std::endl;

    const auto &scheme = time_integrator_.time_stepping_scheme();
    output << "        [ "
           << Patterns::Tools::Convert<TimeSteppingScheme>::to_string(scheme)
           << " with CFL = "
           << std::setprecision(2) << std::fixed << hyperbolic_module_.cfl()
           << " ("
           << std::setprecision(0) << std::fixed << hyperbolic_module_.n_restarts()
           << "/"
           << std::setprecision(0) << std::fixed << parabolic_module_.n_restarts()
           << " rsts) ("
           << std::setprecision(0) << std::fixed << hyperbolic_module_.n_warnings()
           << "/"
           << std::setprecision(0) << std::fixed << parabolic_module_.n_warnings()
           << " warn) ("
           << std::setprecision(0) << std::fixed << hyperbolic_module_.n_corrections()
           << "/"
           << std::setprecision(0) << std::fixed << parabolic_module_.n_corrections()
           << " corr) ]" << std::endl;

    if constexpr (!ParabolicSystem::is_identity)
      parabolic_module_.print_solver_statistics(output);

    output << "        [ dt = "
           << std::scientific << std::setprecision(2) << delta_time
           << " ( "
           << time_per_second
           << " dt/s) ]" << std::endl;
    /* clang-format on */

    /* And print an ETA: */

    time_per_second_exp = 0.8 * time_per_second_exp + 0.2 * time_per_second;
    auto eta = static_cast<unsigned int>(std::max(t_final_ - t, Number(0.)) /
                                         time_per_second_exp);

    output << "\n  ETA : ";

    const unsigned int days = eta / (24 * 3600);
    if (days > 0) {
      output << days << " d  ";
      eta %= 24 * 3600;
    }

    const unsigned int hours = eta / 3600;
    if (hours > 0) {
      output << hours << " h  ";
      eta %= 3600;
    }

    const unsigned int minutes = eta / 60;
    output << minutes << " min";

    output << "   (terminal update every " //
           << std::setprecision(2) << std::fixed << terminal_update_interval_
           << "s)";

    if (mpi_ensemble_.world_rank() != 0)
      return;

    stream << output.str() << std::endl;
  }


  template <typename Description, int dim, typename Number>
  void TimeLoop<Description, dim, Number>::print_cycle_statistics(
      unsigned int cycle,
      Number t,
      unsigned int timer_cycle,
      Number last_checkpoint,
      bool write_to_logfile,
      bool final_time)
  {
    std::ostringstream output;

    /* Print header: */

    std::ostringstream primary;
    if (final_time) {
      primary << "FINAL  (cycle " << Utilities::int_to_string(cycle, 6) << ")";
    } else {
      primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
              << "  (" << std::fixed << std::setprecision(1)     //
              << t / t_final_ * 100 << "%)";
    }

    std::ostringstream secondary;
    secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

    print_head(primary.str(), secondary.str(), output);

    /* Print information and statistics: */

    print_information(timer_cycle, last_checkpoint, output, final_time);
    print_memory_statistics(output);
    print_timers(output);
    print_throughput(cycle, t, output, final_time);

    /* Only output on rank 0: */
    if (mpi_ensemble_.world_rank() != 0)
      return;

#ifndef DEBUG_OUTPUT
    std::cout << "\033[2J\033[H";
#endif
    std::cout << output.str() << std::flush;

    if (write_to_logfile) {
      logfile_ << "\n" << output.str() << std::flush;
    }
  }
} // namespace ryujin
