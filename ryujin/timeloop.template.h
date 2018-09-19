#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "timeloop.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>

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
  TimeLoop<dim>::TimeLoop()
      : ParameterAcceptor("A - TimeLoop")
      , discretization("B - Discretization")
      , offline_data(discretization, "C - OfflineData")
  {
    base_name_ = "test";
    add_parameter("basename", base_name_, "base name for all output files");
  }


  template <int dim>
  void TimeLoop<dim>::run()
  {
    initialize_deallog();

    /* Compute offline data: */

    print_head("compute offline data");
    offline_data.prepare();

    // FIXME The loop ...

    /* Detach deallog: */

    deallog.pop();
    deallog.detach();
  }


  /**
   * Set up deallog output, read in parameters and initialize all objects.
   */
  template <int dim>
  void TimeLoop<dim>::initialize_deallog()
  {
    deallog.pop();

    /* Read in parameters and initialize all objects: */

    deallog << "[Init] Initiating Flux Capacitor... [ OK ]" << std::endl;
    deallog << "[Init] Bringing Warp Core online... [ OK ]" << std::endl;

    deallog << "[Init] Reading parameters and allocating objects... "
            << std::flush;

    ParameterAcceptor::initialize("ryujin.prm");

    deallog << "[ OK ]" << std::endl;

    /* Print out parameters to a prm file: */

    std::ofstream output(base_name_ + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);

    /* Prepare deallog: */

    deallog.push(DEAL_II_GIT_SHORTREV "+" RYUJIN_GIT_SHORTREV);
#ifdef DEBUG
    deallog.depth_console(5);
    deallog.push("DEBUG");
#else
    deallog.depth_console(4);
#endif
    deallog.push(base_name_);

    /* Prepare and attach logfile: */

    filestream.reset(new std::ofstream(base_name_ + "-deallog.log"));
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
