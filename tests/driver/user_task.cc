//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/driver/checkpointing.h>
#include <ryujin/driver/logger.h>
#include <ryujin/driver/scheduler.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/stub/parabolic_module.h>
#include <ryujin/typed/equation.template.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

//
// Test a user defined task in a TaskComposition of its own that an Equation
// runs: The task runs in the step phase after the Simulation,
// restricts the time step size with max_tau(), counts its executions, and
// stores the count in checkpoints. We run the computation once uninterrupted
// and once split into a first run that writes a checkpoint and a second run
// that resumes from it. Both have to report the same number of executions.
//

using namespace ryujin;
using namespace dealii;

using HyperbolicDescription = Euler::HyperbolicDescription;
using ParabolicDescription = StubParabolicDescription;

constexpr int dim = 2;
using Number = NUMBER;


class TestTask final : public ParameterAcceptor, public Interface::Task<Number>
{
public:
  TestTask(const MPIEnsemble &mpi_ensemble)
      : ParameterAcceptor("/O - TestTask")
      , Interface::Task<Number>("test task", TaskPhase::step)
      , mpi_ensemble_(mpi_ensemble)
  {
    max_tau_ = Number(0.01);
    add_parameter("max tau", max_tau_, "Largest time step size");
  }

  void initialize(TaskContext<Number> & /*context*/) override
  {
    ++n_initialize_;
  }

  void prepare(TaskContext<Number> & /*context*/) override
  {
    ++n_prepare_;
  }

  Number max_tau(const TaskContext<Number> & /*context*/) const override
  {
    return max_tau_;
  }

  void execute(TaskContext<Number> &context) override
  {
    /* The Simulation has already chosen tau: */
    AssertThrow(context.tau > Number(0.), ExcInternalError());

    ++n_execute_;
    max_tau_seen_ = std::max(max_tau_seen_, context.tau);
  }

  void save(boost::archive::binary_oarchive &archive) const override
  {
    archive << n_execute_;
  }

  void load(boost::archive::binary_iarchive &archive) override
  {
    archive >> n_execute_;
  }

  void finalize(TaskContext<Number> &context) override
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    const bool tau_restricted =
        max_tau_seen_ <= max_tau_ * (Number(1.) + Number(1.e-10));

    std::cout << "test task: initialize " << n_initialize_ << ", prepare "
              << n_prepare_ << ", execute " << n_execute_ << ", tau restricted "
              << (tau_restricted ? "yes" : "no") << ", t = " << std::fixed
              << std::setprecision(6) << context.t << std::endl;
  }

private:
  Number max_tau_;

  const MPIEnsemble &mpi_ensemble_;

  unsigned int n_initialize_ = 0;
  unsigned int n_prepare_ = 0;
  unsigned int n_execute_ = 0;
  Number max_tau_seen_ = Number(0.);
};


/*
 * A TaskComposition with the Logger, the test task, and checkpointing.
 */
template <int dim, typename Number>
struct UserTaskComposition {
  UserTaskComposition(Interface::Simulation<dim, Number> &simulation)
      : simulation(simulation)
      , logger(simulation, "/M - Logger")
      , checkpointing(simulation, "/N - Checkpointing")
      , test_task(simulation.mpi_ensemble())
      , scheduler(simulation.mpi_ensemble().world_communicator(),
                  {&simulation, &logger},
                  {&test_task, &checkpointing})
  {
  }

  void run()
  {
    scheduler.run();
  }

  Interface::Simulation<dim, Number> &simulation;
  Logger<dim, Number> logger;
  Checkpointing<dim, Number> checkpointing;
  TestTask test_task;
  Scheduler<Number> scheduler;
};


/*
 * Run the UserTaskComposition with the given time loop and checkpointing
 * parameters.
 */
void run(const std::string &time_loop_parameters,
         const std::string &checkpointing_parameters)
{
  std::ostringstream parameters;
  parameters << "subsection A - Scheduler\n"
             << time_loop_parameters //
             << "  set tasks = test task, checkpointing\n"
             << "end\n"
             << "subsection B - Equation\n"
             << "  set gamma = 1.4\n"
             << "end\n"
             << "subsection C - Discretization\n"
             << "  set geometry        = rectangular domain\n"
             << "  set mesh refinement = 3\n"
             << "  subsection rectangular domain\n"
             << "    set boundary condition bottom = dirichlet\n"
             << "    set boundary condition left   = dirichlet\n"
             << "    set boundary condition right  = dirichlet\n"
             << "    set boundary condition top    = dirichlet\n"
             << "    set position bottom left      = -5, -5\n"
             << "    set position top right        =  5,  5\n"
             << "  end\n"
             << "end\n"
             << "subsection E - InitialValues\n"
             << "  set configuration = isentropic vortex\n"
             << "  set direction     =  1,  1\n"
             << "  set position      = -1, -1\n"
             << "  subsection isentropic vortex\n"
             << "    set mach number = 1\n"
             << "    set beta        = 5\n"
             << "  end\n"
             << "end\n"
             << "subsection H - TimeIntegrator\n"
             << "  set cfl min               = 0.2\n"
             << "  set cfl max               = 0.2\n"
             << "  set cfl recovery strategy = none\n"
             << "  set time stepping scheme  = erk 33\n"
             << "end\n"
             << "subsection M - Logger\n"
             << "  set terminal update interval = 0\n"
             << "end\n"
             << "subsection N - Checkpointing\n"
             << checkpointing_parameters //
             << "end\n";

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::ofstream("user_task.prm") << parameters.str();
  const int ierr = MPI_Barrier(MPI_COMM_WORLD);
  AssertThrowMPI(ierr);

  const Equation<HyperbolicDescription,
                 ParabolicDescription,
                 Number,
                 UserTaskComposition>
      equation("user equation");
  equation.run(dim, "user_task.prm", MPI_COMM_WORLD);

  ParameterAcceptor::clear();
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* An uninterrupted computation: */
  run("  set basename          = full\n"
      "  set final time        = 0.1\n"
      "  set timer granularity = 0.1\n",
      "");

  /* A computation to half of the final time that writes a checkpoint: */
  run("  set basename          = split\n"
      "  set final time        = 0.05\n"
      "  set timer granularity = 0.05\n",
      "  set trigger interval = 1.e-6\n");

  /* Resume the computation and run to the final time: */
  run("  set basename          = split\n"
      "  set final time        = 0.1\n"
      "  set timer granularity = 0.05\n",
      "  set resume = true\n");

  return 0;
}
