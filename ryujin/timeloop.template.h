#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "timeloop.h"

#include <helper.h>
#include <indicator.h>
#include <limiter.h>
#include <riemann_solver.h>
#include <scope.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/core/demangle.hpp>

#ifdef CALLGRIND
#include <valgrind/callgrind.h>
#endif

#include <filesystem>
#include <fstream>
#include <iomanip>

using namespace dealii;
using namespace grendel;


namespace ryujin
{

  template <int dim, typename Number>
  TimeLoop<dim, Number>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("A - TimeLoop")
      , mpi_communicator(mpi_comm)
      , discretization(mpi_communicator, "B - Discretization")
      , offline_data(mpi_communicator, discretization, "C - OfflineData")
      , initial_values("D - InitialValues")
      , time_step(mpi_communicator,
                  computing_timer,
                  offline_data,
                  initial_values,
                  "E - TimeStep")
      , postprocessor(
            mpi_communicator, offline_data, "F - SchlierenPostprocessor")
      , mpi_rank(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , n_mpi_processes(
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator))
      , output_thread_active(0)
  {
    base_name = "cylinder";
    add_parameter("basename", base_name, "Base name for all output files");

    t_final = Number(5.);
    add_parameter("final time", t_final, "Final time");

    output_granularity = Number(0.01);
    add_parameter(
        "output granularity", output_granularity, "time interval for output");

    update_granularity = 10;
    add_parameter(
        "update granularity",
        update_granularity,
        "number of cycles after which output statistics are recomputed");

    enable_checkpointing = true;
    add_parameter("enable checkpointing",
                  enable_checkpointing,
                  "Write out checkpoints to resume an interrupted computation "
                  "at output granularity intervals");

    resume = false;
    add_parameter("resume", resume, "Resume an interrupted computation");

    write_output_files = true;
    add_parameter("write output files",
                  write_output_files,
                  "Write out postprocessed output files in vtu/pvtu format");

    enable_compute_error = false;
    add_parameter("enable compute error",
                  enable_compute_error,
                  "Flag to control whether we compute the Linfty Linf_norm of "
                  "the difference to an analytic solution. Implemented only "
                  "for certain initial state configurations.");
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::run()
  {
    initialize();

    print_parameters();

#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::run()" << std::endl;
#endif

    Number t = 0.;
    unsigned int output_cycle = 0;
    vector_type U;

    /* Prepare data structures: */

    print_head("initialize data structures");

    {
      Scope scope(computing_timer, "initialize data structures");

      discretization.prepare();
      offline_data.prepare();
      time_step.prepare();
      postprocessor.prepare();

      print_mpi_partition();

      const auto &partitioner = offline_data.partitioner();
      for (auto &it : U)
        it.reinit(partitioner);

      if (!resume) {
        print_head("interpolate initial values");
        U = interpolate_initial_values();

      } else {

        print_head("resume interrupted computation");

        const auto &triangulation = discretization.triangulation();
        const unsigned int i = triangulation.locally_owned_subdomain();
        std::string name = base_name + "-checkpoint-" +
                           dealii::Utilities::int_to_string(i, 4) + ".archive";
        std::ifstream file(name, std::ios::binary);

        boost::archive::binary_iarchive ia(file);
        ia >> t >> output_cycle;

        for (auto &it1 : U) {
          for (auto &it2 : it1)
            ia >> it2;
          it1.update_ghost_values();
        }
      }
    }

    if (write_output_files) {
      output(U, base_name + "-solution", t, output_cycle);

      if (enable_compute_error) {
        const auto analytic = interpolate_initial_values(t);
        output(analytic, base_name + "-analytic_solution", t, output_cycle);
      }
    }
    ++output_cycle;

    print_head("enter main loop");
    computing_timer["time loop"].start();

    /* Loop: */

    unsigned int cycle = 1;
    for (; t < t_final; ++cycle) {

#ifdef DEBUG_OUTPUT
      print_cycle(cycle, t);
#endif

      /* Do a time step: */

      const auto tau = time_step.step(U, t);
      t += tau;

      if (t > output_cycle * output_granularity) {
        if (write_output_files) {
          output(U,
                 base_name + "-solution",
                 t,
                 output_cycle,
                 /*checkpoint*/ enable_checkpointing);

          if (enable_compute_error) {
            const auto analytic = interpolate_initial_values(t);
            output(analytic, base_name + "-analytic_solution", t, output_cycle);
          }
        }
        ++output_cycle;

        print_cycle_statistics(cycle, t, output_cycle);
      }

#ifndef DEBUG_OUTPUT
      if (cycle % update_granularity == 0)
        print_cycle_statistics(cycle, t, output_cycle);
#endif
    } /* end of loop */

    --cycle; /* We have actually performed one cycle less. */

#ifdef CALLGRIND
    CALLGRIND_DUMP_STATS;
#endif

    /* Wait for output thread: */
    if (output_thread.joinable())
      output_thread.join();

    computing_timer["time loop"].stop();

    if (enable_compute_error) {
      /* Output final error: */
      compute_error(U, t);
    }

    print_timers();
    print_throughput(cycle, t);

#ifdef DEBUG_OUTPUT
    /* Detach deallog: */
    if (mpi_rank == 0) {
      deallog.detach();
    }
#endif
  }


  /**
   * Set up deallog output, read in parameters and initialize all objects.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::initialize()
  {
    /* Read in parameters and initialize all objects: */

    if (mpi_rank == 0) {

      std::cout << "[Init] initiating flux capacitor" << std::endl;
      std::cout << "[Init] bringing warp core online" << std::endl;

      std::cout << "[Init] read parameters and allocate objects" << std::endl;

      ParameterAcceptor::initialize("ryujin.prm");

    } else {

      ParameterAcceptor::initialize("ryujin.prm");
      return;
    }

    /* Print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);

    /* Prepare and attach logfile: */

    filestream.reset(new std::ofstream(base_name + "-deallog.log"));

#ifdef DEBUG_OUTPUT
    deallog.pop();
    deallog.attach(*filestream);
    deallog.depth_console(4);
    deallog.depth_file(4);
#endif

#ifdef DEBUG
    deallog.push("DEBUG");
#endif
  }


  template <int dim, typename Number>
  typename TimeLoop<dim, Number>::vector_type
  TimeLoop<dim, Number>::interpolate_initial_values(Number t)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::interpolate_initial_values(t = " << t
            << ")" << std::endl;
#endif

    vector_type U;

    const auto &partitioner = offline_data.partitioner();
    for (auto &it : U)
      it.reinit(partitioner);

    constexpr auto problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    const auto callable = [&](const auto &p) {
      return initial_values.initial_state(p, t);
    };

    for (unsigned int i = 0; i < problem_dimension; ++i)
      VectorTools::interpolate(offline_data.dof_handler(),
                               to_function<dim, Number>(callable, i),
                               U[i]);

    for (auto &it : U)
      it.update_ghost_values();

    return U;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::compute_error(
      const typename TimeLoop<dim, Number>::vector_type &U, const Number t)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::compute_error()" << std::endl;
#endif

    constexpr auto problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    /* Compute L_inf norm: */

    Vector<float> difference_per_cell(
        discretization.triangulation().n_active_cells());

    Number linf_norm = 0.;
    Number l1_norm = 0;
    Number l2_norm = 0;

    auto analytic = interpolate_initial_values(t);

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      auto &error = analytic[i];

      /* Compute norms of analytic solution: */

      const Number linf_norm_analytic =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_analytic = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      /* Compute norms of error: */

      error -= U[i];

      const Number linf_norm_error =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_error = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      linf_norm += linf_norm_error / linf_norm_analytic;
      l1_norm += l1_norm_error / l1_norm_analytic;
      l2_norm += l2_norm_error / l2_norm_analytic;
    }

    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    auto &stream = *filestream;
#endif

    print_head("compute error");

    stream << "Normalized consolidated Linf, L1, and L2 errors at "
           << "final time" << std::endl;
    stream << "#dofs = " << offline_data.dof_handler().n_dofs() << std::endl;
    stream << "t     = " << t << std::endl;
    stream << "Linf  = " << linf_norm << std::endl;
    stream << "L1    = " << l1_norm << std::endl;
    stream << "L2    = " << l2_norm << std::endl;

#ifndef DEBUG_OUTPUT
    if (mpi_rank == 0) {
      std::cout << "Normalized consolidated Linf, L1, and L2 errors at "
                << "final time" << std::endl;
      std::cout << "#dofs = " << offline_data.dof_handler().n_dofs()
                << std::endl;
      std::cout << "t     = " << t << std::endl;
      std::cout << "Linf  = " << linf_norm << std::endl;
      std::cout << "L1    = " << l1_norm << std::endl;
      std::cout << "L2    = " << l2_norm << std::endl;
    }
#endif
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::output(
      const typename TimeLoop<dim, Number>::vector_type &U,
      const std::string &name,
      Number t,
      unsigned int cycle,
      bool checkpoint)
  {
#ifdef DEBUG_OUTPUT
    deallog << "TimeLoop<dim, Number>::output(t = " << t
            << ", checkpoint = " << checkpoint << ")" << std::endl;
#endif

    /*
     * Offload output to a worker thread.
     *
     * We wait for a previous thread to finish before we schedule a new
     * one. This logic also serves as a mutex for the postprocessor class.
     */

    if (output_thread.joinable()) {
      Scope scope(computing_timer, "output stall");
      output_thread.join();
    }

    {
      Scope scope(computing_timer, "postprocessor");
      postprocessor.compute(U, time_step.alpha());
    }

    /* capture name, t, cycle, and checkpoint by value */
    const auto output_worker = [this, name, t, cycle, checkpoint]() {
      /* Flag thread as active: */
      output_thread_active = 1;
      Scope scope(computing_timer, "output write out");

      /* Checkpointing: */

      if (checkpoint) {
#ifdef DEBUG_OUTPUT
        deallog << "        Checkpointing" << std::endl;
#endif

        const auto &triangulation = discretization.triangulation();
        const unsigned int i = triangulation.locally_owned_subdomain();

        std::string name = base_name + "-checkpoint-" +
                           dealii::Utilities::int_to_string(i, 4) + ".archive";

        if (std::filesystem::exists(name))
          std::filesystem::rename(name, name + "~");

        std::ofstream file(name, std::ios::binary | std::ios::trunc);

        boost::archive::binary_oarchive oa(file);
        oa << t << cycle;
        for (const auto &it1 : postprocessor.U())
          for (const auto &it2 : it1)
            oa << it2;
      }

      /* Data output: */

      postprocessor.write_out_vtu(
          name + "-" + Utilities::int_to_string(cycle, 6), t, cycle);

#ifdef DEBUG_OUTPUT
      deallog << "        Commit output (cycle = " << cycle << ")" << std::endl;
#endif

      /* Flag thread as inactive: */
      output_thread_active = 0;
    };

    /*
     * And spawn the thread:
     */

#ifdef DEBUG_OUTPUT
    deallog << "        Schedule output (cycle = " << cycle << ")" << std::endl;
#endif

    output_thread = std::move(std::thread(output_worker));
  }

  /*
   * Output and logging related functions:
   */


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_parameters()
  {
    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    auto &stream = *filestream;
#endif

    /* Output commit and library informations: */

    /* clang-format off */
    stream << std::endl;
    stream << "###" << std::endl;
    stream << "#" << std::endl;
    stream << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    stream << "# ryujin  version " << std::setw(8) << RYUJIN_VERSION
            << "  -  " << RYUJIN_GIT_REVISION << std::endl;
    stream << "#" << std::endl;
    stream << "###" << std::endl;
    stream << std::endl;

    /* Print compile time parameters: */

    stream << "Compile time parameters:" << std::endl;

    stream << "DIM == " << dim << std::endl;
    stream << "NUMBER == " << typeid(Number).name() << std::endl;

#ifdef USE_SIMD
    stream << "SIMD width == " << VectorizedArray<Number>::n_array_elements << std::endl;
#else
    stream << "SIMD width == " << "(( disabled ))" << std::endl;
#endif

#ifdef USE_CUSTOM_POW
    stream << "serial pow == broadcasted pow(Vec4f)/pow(Vec2d)" << std::endl;
#else
    stream << "serial pow == std::pow"<< std::endl;
#endif

    stream << "Indicator<dim, Number>::indicators_ == ";
    switch (Indicator<dim, Number>::indicator_) {
    case Indicator<dim, Number>::Indicators::zero:
      stream << "Indicator<dim, Number>::Indicators::zero" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::one:
      stream << "Indicator<dim, Number>::Indicators::one" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::entropy_viscosity_commutator:
      stream << "Indicator<dim, Number>::Indicators::entropy_viscosity_commutator" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::smoothness_indicator:
      stream << "Indicator<dim, Number>::Indicators::smoothness_indicator" << std::endl;
    }

    stream << "Indicator<dim, Number>::smoothness_indicator_ == ";
    switch (Indicator<dim, Number>::smoothness_indicator_) {
    case Indicator<dim, Number>::SmoothnessIndicators::rho:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::rho" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::internal_energy:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::internal_energy" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::pressure:
      stream << "Indicator<dim, Number>::SmoothnessIndicators::pressure" << std::endl;
    }

    stream << "Indicator<dim, Number>::smoothness_indicator_alpha_0_ == "
            << Indicator<dim, Number>::smoothness_indicator_alpha_0_ << std::endl;

    stream << "Indicator<dim, Number>::smoothness_indicator_power_ == "
            << Indicator<dim, Number>::smoothness_indicator_power_ << std::endl;

    stream << "Indicator<dim, Number>::compute_second_variations_ == "
            << Indicator<dim, Number>::compute_second_variations_ << std::endl;

    stream << "Limiter<dim, Number>::limiter_ == ";
    switch (Limiter<dim, Number>::limiter_) {
    case Limiter<dim, Number>::Limiters::none:
      stream << "Limiter<dim, Number>::Limiters::none" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::rho:
      stream << "Limiter<dim, Number>::Limiters::rho" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::specific_entropy:
      stream << "Limiter<dim, Number>::Limiters::specific_entropy" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::entropy_inequality:
      stream << "Limiter<dim, Number>::Limiters::entropy_inequality" << std::endl;
      break;
    }

    stream << "grendel::newton_max_iter == "
            << grendel::newton_max_iter << std::endl;

    stream << "Limiter<dim, Number>::relax_bounds_ == "
            << Limiter<dim, Number>::relax_bounds_ << std::endl;

    stream << "Limiter<dim, Number>::relaxation_order_ == "
            << Limiter<dim, Number>::relaxation_order_ << std::endl;

    stream << "RiemannSolver<dim, Number>::newton_max_iter_ == "
            <<  RiemannSolver<dim, Number>::newton_max_iter_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_dij_ == "
            <<  RiemannSolver<dim, Number>::greedy_dij_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_threshold_ == "
            <<  RiemannSolver<dim, Number>::greedy_threshold_ << std::endl;

    stream << "RiemannSolver<dim, Number>::greedy_relax_bounds_ == "
            <<  RiemannSolver<dim, Number>::greedy_relax_bounds_ << std::endl;


    stream << "TimeStep<dim, Number>::order_ == ";
    switch (TimeStep<dim, Number>::order_) {
    case TimeStep<dim, Number>::Order::first_order:
      stream << "TimeStep<dim, Number>::Order::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::Order::second_order:
      stream << "TimeStep<dim, Number>::Order::second_order" << std::endl;
    }

    stream << "TimeStep<dim, Number>::time_step_order_ == ";
    switch (TimeStep<dim, Number>::time_step_order_) {
    case TimeStep<dim, Number>::TimeStepOrder::first_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::second_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::second_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::third_order:
      stream << "TimeStep<dim, Number>::TimeStepOrder::third_order" << std::endl;
      break;
    }

    stream << "TimeStep<dim, Number>::limiter_iter_ == "
            <<  TimeStep<dim, Number>::limiter_iter_ << std::endl;

    /* clang-format on */

#ifndef DEBUG_OUTPUT
    stream << std::endl;
    stream << "Run time parameters:" << std::endl;
    ParameterAcceptor::prm.print_parameters(
        stream, ParameterHandler::OutputStyle::ShortText);
#endif
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_mpi_partition()
  {
    unsigned int dofs[2];
    dofs[0] = offline_data.n_locally_owned();
    dofs[1] = offline_data.n_locally_internal();

    if (mpi_rank > 0) {
      MPI_Send(&dofs, 2, MPI_UNSIGNED, 0, 0, mpi_communicator);

    } else {

#ifdef DEBUG_OUTPUT
      auto &stream = deallog;
#else
      auto &stream = *filestream;
#endif

      stream << "Number of MPI ranks: " << n_mpi_processes << std::endl;
      stream << "Number of threads:   " << MultithreadInfo::n_threads()
             << std::endl;

      /* Print out the DoF distribution: */

      const auto n_dofs = offline_data.dof_handler().n_dofs();

      stream << "Qdofs: " << n_dofs
             << " global DoFs, local DoF distribution:" << std::endl;


      for (unsigned int p = 0; p < n_mpi_processes; ++p) {
        stream << "    Rank " << p << std::flush;

        if (p != 0)
          MPI_Recv(&dofs,
                   2,
                   MPI_UNSIGNED,
                   p,
                   0,
                   mpi_communicator,
                   MPI_STATUS_IGNORE);

        stream << ":\tlocal: " << dofs[0] << std::flush;
        stream << "\tinternal: " << dofs[1] << std::endl;
      } /* p */
    }   /* mpi_rank */
  }


  /**
   * A small function that prints formatted section headings.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_head(const std::string &header,
                                         const std::string &secondary,
                                         bool use_cout)
  {
    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    std::ostream &stream = use_cout ? std::cout : *filestream;
#endif

    const auto header_size = header.size();
    const auto padded_header = std::string((34 - header_size) / 2, ' ') +
                               header +
                               std::string((35 - header_size) / 2, ' ');

    const auto secondary_size = secondary.size();
    const auto padded_secondary = std::string((34 - secondary_size) / 2, ' ') +
                                  secondary +
                                  std::string((35 - secondary_size) / 2, ' ');

    /* clang-format off */
    stream << std::endl;
    stream << "    ####################################################" << std::endl;
    stream << "    #########                                  #########" << std::endl;
    stream << "    #########"     <<  padded_header   <<     "#########" << std::endl;
    stream << "    #########"     << padded_secondary <<     "#########" << std::endl;
    stream << "    #########                                  #########" << std::endl;
    stream << "    ####################################################" << std::endl;
    stream << std::endl;
    /* clang-format on */

    if (secondary == "")
      std::cout << "[Init] " << header << std::endl;
  }


  /**
   * Print a formatted head for a given cycle:
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_cycle(unsigned int cycle,
                                          Number t,
                                          bool use_cout)
  {
    std::ostringstream primary;
    primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
            << "  (" << std::fixed << std::setprecision(1)     //
            << t / t_final * 100 << "%)";

    std::ostringstream secondary;
    secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

    print_head(primary.str(), secondary.str(), use_cout);
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_timers(bool use_cout)
  {
    std::vector<std::ostringstream> output(computing_timer.size());

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
      const auto wall_time =
          Utilities::MPI::min_max_avg(timer.wall_time(), mpi_communicator);

      stream << std::setprecision(2) << std::fixed << std::setw(8)
             << wall_time.avg << "s [sk: " << std::setprecision(1)
             << std::setw(5) << std::fixed
             << 100. * (wall_time.max - wall_time.avg) / wall_time.avg << "%]";
      unsigned int n = dealii::Utilities::needed_digits(n_mpi_processes);
      stream << " [p" << std::setw(n) << wall_time.max_index << "]";
    };

    const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
        computing_timer["time loop"].cpu_time(), mpi_communicator);
    const double total_cpu_time = cpu_time_statistics.sum;

    const auto print_cpu_time = [&](auto &timer,
                                    auto &stream,
                                    bool percentage) {
      const auto cpu_time =
          Utilities::MPI::min_max_avg(timer.cpu_time(), mpi_communicator);

      stream << std::setprecision(2) << std::fixed << std::setw(9)
             << cpu_time.sum << "s ";

      if (percentage)
        stream << "(" << std::setprecision(1) << std::setw(4)
               << 100. * cpu_time.sum / total_cpu_time << "%)";
      else
        stream << "       ";

      stream << " [sk: " << std::setprecision(1) << std::setw(5) << std::fixed
             << 100. * (cpu_time.max - cpu_time.avg) / cpu_time.avg << "%]";
    };

    auto jt = output.begin();
    for (auto &it : computing_timer)
      *jt++ << "  " << it.first;
    equalize();

    jt = output.begin();
    for (auto &it : computing_timer)
      print_wall_time(it.second, *jt++);
    equalize();

    jt = output.begin();
    for (auto &it : computing_timer)
      print_cpu_time(it.second, *jt++, it.first.find("time s") == 0);
    equalize();

    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    std::ostream &stream = use_cout ? std::cout : *filestream;
#endif

    stream << std::endl << std::endl << "Timer statistics:" << std::endl;
    for (auto &it : output)
      stream << it.str() << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_throughput(unsigned int cycle,
                                               Number t,
                                               bool use_cout)
  {
    /* Print Jean-Luc and Martin metrics: */

    const auto wall_time_statistics = Utilities::MPI::min_max_avg(
        computing_timer["time loop"].wall_time(), mpi_communicator);
    const double wall_time = wall_time_statistics.max;

    const auto cpu_time_statistics = Utilities::MPI::min_max_avg(
        computing_timer["time loop"].cpu_time(), mpi_communicator);
    const double cpu_time = cpu_time_statistics.sum;

    const double wall_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        wall_time;

    const double cpu_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        cpu_time;

    std::ostringstream head;

    head << std::setprecision(4) << std::endl << std::endl;
    head << "Throughput:  (CPU )  "                                    //
         << std::fixed << cpu_m_dofs_per_sec << " MQ/s  ("             //
         << std::scientific << 1. / cpu_m_dofs_per_sec * 1.e-6         //
         << " s/Qdof/cycle)" << std::endl;                             //
    head << "                     [cpu time skew: "                    //
         << std::setprecision(2) << std::scientific                    //
         << cpu_time_statistics.max - cpu_time_statistics.avg << "s (" //
         << std::setprecision(1) << std::setw(4) << std::setfill(' ')
         << std::fixed
         << 100. * (cpu_time_statistics.max - cpu_time_statistics.avg) /
                cpu_time_statistics.avg
         << "%)]" << std::endl
         << std::endl;

    head << "             (WALL)  "                                      //
         << std::fixed << wall_m_dofs_per_sec << " MQ/s  ("              //
         << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6          //
         << " s/Qdof/cycle)  ("                                          //
         << std::fixed << ((double)cycle) / wall_time                    //
         << " cycles/s)  (avg dt = "                                     //
         << std::scientific << t / ((double)cycle)                       //
         << ")" << std::endl;                                            //
    head << "                     [ "                                    //
         << std::setprecision(0) << std::fixed << time_step.n_restarts() //
         << " rsts   (" << std::setprecision(2) << std::scientific
         << time_step.n_restarts() / ((double)cycle) << " rsts/cycle) ]"
         << std::endl
         << std::endl;

    head << "ETA:  " << std::fixed << std::setprecision(4)
         << ((t_final - t) / t * wall_time / 3600.) << " h";

    if (mpi_rank != 0)
      return;

#ifdef DEBUG_OUTPUT
    auto &stream = deallog;
#else
    std::ostream &stream = use_cout ? std::cout : *filestream;
#endif

    stream << head.str() << std::endl;
  }


  /**
   * A small function that prints formatted section headings.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_cycle_statistics(unsigned int cycle,
                                                     Number t,
                                                     unsigned int output_cycle)
  {
    unsigned int n_active_writebacks =
        Utilities::MPI::sum(output_thread_active, mpi_communicator);

    if (mpi_rank == 0) {
      std::ostringstream primary;
      primary << "Cycle  " << Utilities::int_to_string(cycle, 6) //
              << "  (" << std::fixed << std::setprecision(1)     //
              << t / t_final * 100 << "%)";

      std::ostringstream secondary;
      secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

#ifndef DEBUG_OUTPUT
      std::cout << "\033[2J\033[H" << std::endl;
#endif

      print_head(primary.str(), secondary.str(), /*use_cout*/ true);

      std::cout << std::endl;
      std::cout << "Information: [" << base_name << "] with "
                << offline_data.dof_handler().n_dofs() << " Qdofs on "
                << n_mpi_processes << " ranks / "
                << MultithreadInfo::n_threads() << " threads" << std::endl;

      std::cout << "             Last output cycle " << output_cycle - 1
                << " at t = " << output_granularity * (output_cycle - 1)
                << std::endl;

      if (n_active_writebacks > 0)
        std::cout << "             !!! " << n_active_writebacks
                  << " ranks performing output !!!" << std::flush;
    }

    print_timers(/*use_cout*/ true);
    print_throughput(cycle, t, /*use_cout*/ true);
  }


} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */
