#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "timeloop.h"

#include <deal.II/base/work_stream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/grid/grid_out.h>

#include <iomanip>
#include <type_traits>


using namespace dealii;
using namespace grendel;


namespace
{
  /**
   * A helper function to print formatted section headings.
   */
  void print_head(std::string header)
  {
    const auto size = header.size();

    deallog << std::endl;
    deallog << "    ####################################################"
            << std::endl;
    deallog << "    #########                                  #########"
            << std::endl;
    deallog << "    #########"                   //
            << std::string((34 - size) / 2, ' ') //
            << header                            //
            << std::string((35 - size) / 2, ' ') //
            << "#########"                       //
            << std::endl;
    deallog << "    #########                                  #########"
            << std::endl;
    deallog << "    ####################################################"
            << std::endl;
    deallog << std::endl;
  }
} // namespace


namespace ryujin
{
  template <int dim>
  TimeLoop<dim>::TimeLoop(const MPI_Comm &mpi_comm)
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
      , time_step(
            mpi_communicator, computing_timer, offline_data, "D - TimeStep")
  {
    base_name = "test";
    add_parameter("basename", base_name, "base name for all output files");
  }


  template <int dim>
  void TimeLoop<dim>::run()
  {
    initialize();

    discretization.create_triangulation();

    {
      deallog << "        output triangulation" << std::endl;
      std::ofstream output(
          base_name + "-triangulation-p" +
          std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) +
          ".inp");
      GridOut().write_ucd(discretization.triangulation(), output);
    }

    /* Compute offline data: */

    print_head("compute offline data");
    offline_data.prepare();

    //DEBUG

    using rank1_type = typename RiemannSolver<dim>::rank1_type;

#if 0
      rank1_type U_i{{2.21953,1.09817,0,5.09217}};
      rank1_type U_j{{2.21953,1.09817,0,5.09217}};
      Tensor<1, dim> n_ij{{0.948683,-0.316228}};
      // output 1.57222 NEW
#endif

#if 0
      rank1_type U_i{{2.18162, 1.06679, 5.52606e-06, 5.00393}};
      rank1_type U_j{{1.97325, 0.777591, -1.21599e-06, 4.33331}};
      Tensor<1, dim> n_ij{{0.83205, -0.5547}};
      // output 1.47181 NEW
#endif

    rank1_type U_i{{2.21953, 1.09817, 0., 5.09217}};
    rank1_type U_j{{1.4, 0., 0., 2.5}};
    Tensor<1, dim> n_ij{{0.948683, -0.316228}};
    // output 1.33017 NEW

    // benchmarking:
    constexpr unsigned int size = 10000000;
    std::vector<double> scratch(size);
    {
      TimerOutput::Scope t(computing_timer, "benchmark - compute lambda_max");

      const auto on_subranges = [&](auto it1, auto it2) {
        for (auto it = it1; it != it2; ++it) {
          const auto [lambda_max, n_iterations] =
              riemann_solver.lambda_max(U_i, U_j, n_ij);
          *it = lambda_max;
        }
      };
      parallel::apply_to_subranges(
          scratch.begin(), scratch.end(), on_subranges, 4096);
    }

    // Output result:
    {
      const auto [lambda_max, n_iterations] =
          riemann_solver.lambda_max(U_i, U_j, n_ij);
      std::cout << "RESULT: " << lambda_max << " in n=" << n_iterations
                << " iterations <-----" << std::endl;
    }

    // FIXME The loop ...

    computing_timer.print_summary();
    deallog << timer_output.str() << std::endl;

    /* Detach deallog: */

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      deallog.pop();
      deallog.detach();
    }
  }


  /**
   * Set up deallog output, read in parameters and initialize all objects.
   */
  template <int dim>
  void TimeLoop<dim>::initialize()
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

    /* Prepare deallog: */

    deallog.push(DEAL_II_GIT_SHORTREV "+" RYUJIN_GIT_SHORTREV);
#ifdef DEBUG
    deallog.depth_console(5);
    deallog.push("DEBUG");
#else
    deallog.depth_console(4);
#endif
    deallog.push(base_name);

    /* Prepare and attach logfile: */

    filestream.reset(new std::ofstream(base_name + "-deallog.log"));
    deallog.attach(*filestream);

    /* Output commit and library informations: */

    /* clang-format off */
    deallog << "###" << std::endl;
    deallog << "#" << std::endl;
    deallog << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    deallog << "# ryujin  version " << std::setw(8) << RYUJIN_VERSION
            << "  -  " << RYUJIN_GIT_REVISION << std::endl;
    deallog << "#" << std::endl;
    deallog << "###" << std::endl;
    /* clang-format on */

    /* Print out parameters to deallog as well: */

    deallog << "TimeLoop<dim>::run()" << std::endl;
    ParameterAcceptor::prm.log_parameters(deallog);
  }

} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */
