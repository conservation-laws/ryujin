#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "timeloop.h"

#include <helper.h>
#include <indicator.h>
#include <limiter.h>
#include <riemann_solver.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

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


namespace
{
  /**
   * A helper function to print formatted section headings.
   */

  void print_head(std::string header, std::string secondary = "")
  {
    const auto header_size = header.size();
    const auto padded_header = std::string((34 - header_size) / 2, ' ') +
                               header +
                               std::string((35 - header_size) / 2, ' ');

    const auto secondary_size = secondary.size();
    const auto padded_secondary = std::string((34 - secondary_size) / 2, ' ') +
                                  secondary +
                                  std::string((35 - secondary_size) / 2, ' ');

    /* clang-format off */
    deallog << std::endl;
    deallog << "    ####################################################" << std::endl;
    deallog << "    #########                                  #########" << std::endl;
    deallog << "    #########"     <<  padded_header   <<     "#########" << std::endl;
    deallog << "    #########"     << padded_secondary <<     "#########" << std::endl;
    deallog << "    #########                                  #########" << std::endl;
    deallog << "    ####################################################" << std::endl;
    deallog << std::endl;
    /* clang-format on */
  }
} // namespace


namespace ryujin
{

  template <int dim, typename Number>
  TimeLoop<dim, Number>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("A - TimeLoop")
      , mpi_communicator(mpi_comm)
      , computing_timer(mpi_communicator,
                        timer_output,
                        TimerOutput::never,
                        TimerOutput::cpu_and_wall_times)
      , discretization(mpi_communicator, computing_timer, "B - Discretization")
      , offline_data(mpi_communicator,
                     computing_timer,
                     discretization,
                     "C - OfflineData")
      , initial_values("D - InitialValues")
      , time_step(mpi_communicator,
                  computing_timer,
                  offline_data,
                  initial_values,
                  "E - TimeStep")
      , postprocessor(mpi_communicator,
                      computing_timer,
                      offline_data,
                      "F - SchlierenPostprocessor")
  {
    base_name = "cylinder";
    add_parameter("basename", base_name, "Base name for all output files");

    t_final = Number(5.);
    add_parameter("final time", t_final, "Final time");

    output_granularity = Number(0.01);
    add_parameter(
        "output granularity", output_granularity, "time interval for output");

    enable_detailed_output = false;
    add_parameter("enable detailed output",
                  enable_detailed_output,
                  "Enable detailed terminal output to deallog");

    enable_checkpointing = true;
    add_parameter("enable checkpointing",
                  enable_checkpointing,
                  "Write out checkpoints to resume an interrupted computation "
                  "at output granularity intervals");

    resume = false;
    add_parameter("resume", resume, "Resume an interrupted computation");

    write_mesh = false;
    add_parameter("write mesh",
                  write_mesh,
                  "Write out the (distributed) mesh in inp format");

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
    /* Initialize deallog: */

    initialize();

    deallog << "TimeLoop<dim, Number>::run()" << std::endl;

    /* Create distributed triangulation and output the triangulation: */

    print_head("create triangulation");
    discretization.prepare();

    if (write_mesh) {
      deallog << "        output triangulation" << std::endl;
      std::ofstream output(
          base_name + "-triangulation-p" +
          std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) +
          ".inp");
      GridOut().write_ucd(discretization.triangulation(), output);
    }

    /* Prepare offline data: */

    print_head("compute offline data");
    offline_data.prepare();

    print_head("set up time step");
    time_step.prepare();
    postprocessor.prepare();

    /* Interpolate initial values: */

    print_head("interpolate initial values");

    Number t = 0.;
    unsigned int output_cycle = 0;
    auto U = interpolate_initial_values();

    if (resume) {
      print_head("restore interrupted computation");

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

    if (write_output_files)
      output(U, base_name + "-solution", t, output_cycle);

    if (write_output_files && enable_compute_error) {
      const auto analytic = interpolate_initial_values(t);
      output(analytic, base_name + "-analytic_solution", t, output_cycle);
    }

    ++output_cycle;

    print_head("enter main loop");

    /* Disable deallog output: */

    if (!enable_detailed_output)
      deallog.push("SILENCE!");

    /* Loop: */

    unsigned int cycle = 1;
    for (; t < t_final; ++cycle) {

      std::ostringstream head;
      head << "Cycle  " << Utilities::int_to_string(cycle, 6) << "  ("
           << std::fixed << std::setprecision(1) << t / t_final * 100 << "%)";
      std::ostringstream secondary;
      secondary << "at time t = " << std::setprecision(8) << std::fixed << t;
      print_head(head.str(), secondary.str());

      /* Do a time step: */

      const auto tau = time_step.step(U, t);
      t += tau;

      if (t > output_cycle * output_granularity) {
        if (!enable_detailed_output) {
          deallog.pop();
          print_head(head.str(), secondary.str());
        }

        if (write_output_files)
          output(U,
                 base_name + "-solution",
                 t,
                 output_cycle,
                 /*checkpoint*/ enable_checkpointing);

        if (write_output_files && enable_compute_error) {
          const auto analytic = interpolate_initial_values(t);
          output(analytic, base_name + "-analytic_solution", t, output_cycle);
        }

        ++output_cycle;

        print_throughput(cycle);

        if (!enable_detailed_output)
          deallog.push("SILENCE!");
      }
    } /* end of loop */

#ifdef CALLGRIND
    CALLGRIND_DUMP_STATS;
#endif

    /* Wait for output thread: */

    if (output_thread.joinable())
      output_thread.join();

    /* Reenable deallog output: */

    if (!enable_detailed_output)
      deallog.pop();

    if (enable_compute_error) {
      /* Output final error: */

      const auto &affine_constraints = offline_data.affine_constraints();
      for (auto &it : U)
        affine_constraints.distribute(it);
      compute_error(U, t);
    }

    computing_timer.print_summary();
    deallog << timer_output.str() << std::endl;

    print_throughput(cycle);

    /* Detach deallog: */
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      deallog.pop();
      deallog.detach();
    }
  }


  /**
   * Set up deallog output, read in parameters and initialize all objects.
   */
  template <int dim, typename Number>
  void TimeLoop<dim, Number>::initialize()
  {
    /* Read in parameters and initialize all objects: */

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {

      deallog.pop();

      deallog << "[Init] Initiating Flux Capacitor... [ OK ]" << std::endl;
      deallog << "[Init] Bringing Warp Core online... [ OK ]" << std::endl;

      deallog << "[Init] Reading parameters and allocating objects... "
              << std::flush;

      ParameterAcceptor::initialize("ryujin.prm");

      deallog << "[ OK ]" << std::endl;

    } else {

      ParameterAcceptor::initialize("ryujin.prm");
      return;
    }

    /* Print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);

    /* Prepare and attach logfile: */

    filestream.reset(new std::ofstream(base_name + "-deallog.log"));
    deallog.attach(*filestream);

    /* Output commit and library informations: */

    /* clang-format off */
    deallog.depth_console(4);
    deallog << "###" << std::endl;
    deallog << "#" << std::endl;
    deallog << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    deallog << "# ryujin  version " << std::setw(8) << RYUJIN_VERSION
            << "  -  " << RYUJIN_GIT_REVISION << std::endl;
    deallog << "#" << std::endl;
    deallog << "###" << std::endl;

    /* Print compile time parameters: */

    deallog << "Compile time parameters:" << std::endl;

    deallog << "DIM == " << dim << std::endl;
    deallog << "NUMBER == " << typeid(Number).name() << std::endl;

#ifdef USE_SIMD
    deallog << "SIMD width == " << VectorizedArray<Number>::n_array_elements << std::endl;
#else
    deallog << "SIMD width == " << "(( disabled ))" << std::endl;
#endif

#ifdef USE_CUSTOM_POW
    deallog << "serial pow == broadcasted pow(Vec4f)/pow(Vec2d)" << std::endl;
#else
    deallog << "serial pow == std::pow"<< std::endl;
#endif


    deallog << "Indicator<dim, Number>::indicators_ == ";
    switch (Indicator<dim, Number>::indicator_) {
    case Indicator<dim, Number>::Indicators::zero:
      deallog << "Indicator<dim, Number>::Indicators::zero" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::one:
      deallog << "Indicator<dim, Number>::Indicators::one" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::entropy_viscosity_commutator:
      deallog << "Indicator<dim, Number>::Indicators::entropy_viscosity_commutator" << std::endl;
      break;
    case Indicator<dim, Number>::Indicators::smoothness_indicator:
      deallog << "Indicator<dim, Number>::Indicators::smoothness_indicator" << std::endl;
    }

    deallog << "Indicator<dim, Number>::smoothness_indicator_ == ";
    switch (Indicator<dim, Number>::smoothness_indicator_) {
    case Indicator<dim, Number>::SmoothnessIndicators::rho:
      deallog << "Indicator<dim, Number>::SmoothnessIndicators::rho" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::internal_energy:
      deallog << "Indicator<dim, Number>::SmoothnessIndicators::internal_energy" << std::endl;
      break;
    case Indicator<dim, Number>::SmoothnessIndicators::pressure:
      deallog << "Indicator<dim, Number>::SmoothnessIndicators::pressure" << std::endl;
    }

    deallog << "Indicator<dim, Number>::smoothness_indicator_alpha_0_ == "
            << Indicator<dim, Number>::smoothness_indicator_alpha_0_ << std::endl;

    deallog << "Indicator<dim, Number>::smoothness_indicator_power_ == "
            << Indicator<dim, Number>::smoothness_indicator_power_ << std::endl;

    deallog << "Indicator<dim, Number>::compute_second_variations_ == "
            << Indicator<dim, Number>::compute_second_variations_ << std::endl;

    deallog << "Limiter<dim, Number>::limiter_ == ";
    switch (Limiter<dim, Number>::limiter_) {
    case Limiter<dim, Number>::Limiters::none:
      deallog << "Limiter<dim, Number>::Limiters::none" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::rho:
      deallog << "Limiter<dim, Number>::Limiters::rho" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::internal_energy:
      deallog << "Limiter<dim, Number>::Limiters::internal_energy" << std::endl;
      break;
    case Limiter<dim, Number>::Limiters::specific_entropy:
      deallog << "Limiter<dim, Number>::Limiters::specific_entropy" << std::endl;
    }

    deallog << "Limiter<dim, Number>::relax_bounds_ == "
            << Limiter<dim, Number>::relax_bounds_ << std::endl;

    deallog << "Limiter<dim, Number>::relaxation_order_ == "
            << Limiter<dim, Number>::relaxation_order_ << std::endl;

    deallog << "Limiter<dim, Number>::line_search_eps_ == "
            << Limiter<dim, Number>::line_search_eps_ << std::endl;

    deallog << "Limiter<dim, Number>::line_search_max_iter_ == "
            << Limiter<dim, Number>::line_search_max_iter_ << std::endl;

    deallog << "RiemannSolver<dim, Number>::newton_eps_ == "
            <<  RiemannSolver<dim, Number>::newton_eps_ << std::endl;

    deallog << "RiemannSolver<dim, Number>::newton_max_iter_ == "
            <<  RiemannSolver<dim, Number>::newton_max_iter_ << std::endl;

    deallog << "TimeStep<dim, Number>::order_ == ";
    switch (TimeStep<dim, Number>::order_) {
    case TimeStep<dim, Number>::Order::first_order:
      deallog << "TimeStep<dim, Number>::Order::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::Order::second_order:
      deallog << "TimeStep<dim, Number>::Order::second_order" << std::endl;
    }

    deallog << "TimeStep<dim, Number>::time_step_order_ == ";
    switch (TimeStep<dim, Number>::time_step_order_) {
    case TimeStep<dim, Number>::TimeStepOrder::first_order:
      deallog << "TimeStep<dim, Number>::TimeStepOrder::first_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::second_order:
      deallog << "TimeStep<dim, Number>::TimeStepOrder::second_order" << std::endl;
      break;
    case TimeStep<dim, Number>::TimeStepOrder::third_order:
      deallog << "TimeStep<dim, Number>::TimeStepOrder::third_order" << std::endl;
      break;
    }

    deallog << "TimeStep<dim, Number>::limiter_iter_ == "
            <<  TimeStep<dim, Number>::limiter_iter_ << std::endl;

    /* clang-format on */

    deallog << "Run time parameters:" << std::endl;

    ParameterAcceptor::prm.log_parameters(deallog);

    deallog << "Number of MPI ranks: "
            << Utilities::MPI::n_mpi_processes(mpi_communicator) << std::endl;
    deallog << "Number of threads:   " << MultithreadInfo::n_threads()
            << std::endl;

    deallog.push(DEAL_II_GIT_SHORTREV "+" RYUJIN_GIT_SHORTREV);
    deallog.push(base_name);
#ifdef DEBUG
    deallog.depth_console(3);
    deallog.depth_file(3);
    deallog.push("DEBUG");
#else
    deallog.depth_console(2);
    deallog.depth_file(2);
#endif
  }


  template <int dim, typename Number>
  typename TimeLoop<dim, Number>::vector_type
  TimeLoop<dim, Number>::interpolate_initial_values(Number t)
  {
    deallog << "TimeLoop<dim, Number>::interpolate_initial_values(t = " << t
            << ")" << std::endl;
    TimerOutput::Scope timer(computing_timer,
                             "time_loop - setup scratch space");

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
    deallog << "TimeLoop<dim, Number>::compute_error()" << std::endl;
    TimerOutput::Scope timer(computing_timer, "time_loop - compute error");

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
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
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
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const Number l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const Number l2_norm_error = Number(std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator)));

      linf_norm += linf_norm_error / linf_norm_analytic;
      l1_norm += l1_norm_error / l1_norm_analytic;
      l2_norm += l2_norm_error / l2_norm_analytic;
    }

    deallog << "        Normalized consolidated Linf, L1, and L2 errors at "
            << "final time" << std::endl;
    deallog << "        #dofs = " << offline_data.dof_handler().n_dofs()
            << std::endl;
    deallog << "        t     = " << t << std::endl;
    deallog << "        Linf  = " << linf_norm << std::endl;
    deallog << "        L1    = " << l1_norm << std::endl;
    deallog << "        L2    = " << l2_norm << std::endl;
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::output(
      const typename TimeLoop<dim, Number>::vector_type &U,
      const std::string &name,
      Number t,
      unsigned int cycle,
      bool checkpoint)
  {
    deallog << "TimeLoop<dim, Number>::output(t = " << t
            << ", checkpoint = " << checkpoint << ")" << std::endl;

    /*
     * Offload output to a worker thread.
     *
     * We wait for a previous thread to finish before we schedule a new
     * one. This logic also serves as a mutex for output_vector and
     * postprocessor.
     */

    deallog << "        Schedule output cycle = " << cycle << std::endl;
    if (output_thread.joinable()) {
      TimerOutput::Scope timer(computing_timer, "time_loop - stalled output");
      output_thread.join();
    }

    constexpr auto problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;
    const auto &component_names =
        ProblemDescription<dim, Number>::component_names;
    const auto &affine_constraints = offline_data.affine_constraints();

    /* Copy the current state vector over to output_vector: */

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      output_vector[i] = U[i];
    }

    output_alpha = time_step.alpha();


    /* Distribute hanging nodes and update ghost values: */

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      affine_constraints.distribute(output_vector[i]);
      output_vector[i].update_ghost_values();
    }

    affine_constraints.distribute(output_alpha);
    output_alpha.update_ghost_values();

    postprocessor.compute(output_vector);

    /* Output data in vtu format: */

    /* capture name, t, cycle by value */
    const auto output_worker = [this, name, t, cycle, checkpoint]() {
      constexpr auto problem_dimension =
          ProblemDescription<dim, Number>::problem_dimension;
      const auto &dof_handler = offline_data.dof_handler();
      const auto &triangulation = discretization.triangulation();
      const auto &mapping = discretization.mapping();

      /* Checkpointing: */

      if (checkpoint) {
        deallog << "        Checkpointing" << std::endl;

        const unsigned int i = triangulation.locally_owned_subdomain();
        std::string name = base_name + "-checkpoint-" +
                           dealii::Utilities::int_to_string(i, 4) + ".archive";

        if (std::filesystem::exists(name))
          std::filesystem::rename(name, name + "~");

        std::ofstream file(name, std::ios::binary | std::ios::trunc);

        boost::archive::binary_oarchive oa(file);
        oa << t << cycle;
        for (const auto &it1 : output_vector)
          for (const auto &it2 : it1)
            oa << it2;
      }

      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out.add_data_vector(output_vector[i], component_names[i]);

      data_out.add_data_vector(postprocessor.schlieren(), "schlieren");

      data_out.add_data_vector(output_alpha, "alpha");

      data_out.build_patches(mapping,
                             discretization.finite_element().degree - 1);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out.set_flags(flags);

      const auto filename = [&](const unsigned int i) -> std::string {
        const auto seq = dealii::Utilities::int_to_string(i, 4);
        return name + "-" + Utilities::int_to_string(cycle, 6) + "-" + seq +
               ".vtu";
      };

      /* Write out local vtu: */

      const unsigned int i = triangulation.locally_owned_subdomain();
      std::ofstream output(filename(i));
      data_out.write_vtu(output);

      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        /* Write out pvtu control file: */

        const unsigned int n_mpi_processes =
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < n_mpi_processes; ++i)
          filenames.push_back(filename(i));

        std::ofstream output(name + "-" + Utilities::int_to_string(cycle, 6) +
                             ".pvtu");
        data_out.write_pvtu_record(output, filenames);
      }

      deallog << "        Commit output cycle = " << cycle << std::endl;
    };

    /*
     * And spawn the thread:
     */
    output_thread = std::move(std::thread(output_worker));
  }


  template <int dim, typename Number>
  void TimeLoop<dim, Number>::print_throughput(unsigned int cycle)
  {
    /* Print Jean-Luc and Martin metrics: */

    const auto cpu_summary_data = computing_timer.get_summary_data(
        TimerOutput::OutputData::total_cpu_time);
    const auto wall_summary_data = computing_timer.get_summary_data(
        TimerOutput::OutputData::total_wall_time);

    double cpu_time =
        std::accumulate(cpu_summary_data.begin(),
                        cpu_summary_data.end(),
                        0.,
                        [](auto sum, auto it) { return sum + it.second; });
    cpu_time = Utilities::MPI::sum(cpu_time, mpi_communicator);

    const double wall_time =
        std::accumulate(wall_summary_data.begin(),
                        wall_summary_data.end(),
                        0.,
                        [](auto sum, auto it) { return sum + it.second; });

    const double cpu_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        cpu_time;
    const double wall_m_dofs_per_sec =
        ((double)cycle) * ((double)offline_data.dof_handler().n_dofs()) / 1.e6 /
        wall_time;

    std::ostringstream head;
    head << std::setprecision(4) << std::endl << std::endl;
    head << "Throughput: (CPU )  " << std::fixed << cpu_m_dofs_per_sec
         << " MQ/s  (" << std::scientific << 1. / cpu_m_dofs_per_sec * 1.e-6
         << " s/Qdof/cycle)" << std::endl;
    head << "            (WALL)  " << std::fixed << wall_m_dofs_per_sec
         << " MQ/s  (" << std::scientific << 1. / wall_m_dofs_per_sec * 1.e-6
         << " s/Qdof/cycle)" << std::endl;

    deallog << head.str() << std::endl;
  }


} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */
