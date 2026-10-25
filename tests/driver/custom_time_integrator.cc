//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/time_integrator.h>
#include <ryujin/driver/environment.h>
#include <ryujin/driver/scheduler.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/interface/simulation.h>
#include <ryujin/interface/time_integrator.h>
#include <ryujin/stub/parabolic_module.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

//
// Test a Simulation with a custom time integrator: ForwardEuler performs
// explicit Euler steps directly with HyperbolicModule::forward_euler_step().
// We run the computation once with the TimeIntegrator of ryujin and the
// scheme "erk 11", and once with ForwardEuler. Both have to produce
// identical results.
//

using namespace ryujin;
using namespace dealii;

using HyperbolicDescription = Euler::HyperbolicDescription;
using ParabolicDescription = StubParabolicDescription;

constexpr int dim = 2;
using Number = NUMBER;


template <int dim, typename Number>
class ForwardEuler final : public Interface::TimeIntegrator<dim, Number>
{
public:
  ForwardEuler(
      const MPIEnsemble & /*mpi_ensemble*/,
      const Interface::HyperbolicSystem & /*hyperbolic_system*/,
      const Interface::ParabolicSystem & /*parabolic_system*/,
      const Interface::HyperbolicModule<dim, Number> &hyperbolic_module,
      const Interface::ParabolicModule<dim, Number> &parabolic_module,
      const std::string &subsection)
      : Interface::TimeIntegrator<dim, Number>(subsection)
      , hyperbolic_module_(hyperbolic_module)
      , parabolic_module_(parabolic_module)
  {
    this->time_stepping_scheme_name_ = "forward euler";
  }

  void prepare() override
  {
    hyperbolic_module_.reinit_state_vector(temp_);
    parabolic_module_.reinit_state_vector(temp_);
    hyperbolic_module_.set_cfl(Number(0.2));
  }

  void prepare_state_vector(StateVector<Number> &state_vector,
                            Number t) const override
  {
    hyperbolic_module_.prepare_state_vector(state_vector, t);
  }

  Number step(StateVector<Number> &state_vector,
              Number t,
              Number tau,
              Number tau_max) override
  {
    tau = hyperbolic_module_.forward_euler_step(
        state_vector, {}, {}, temp_, t, tau, tau_max);
    std::swap(state_vector, temp_);
    return tau;
  }

private:
  const Interface::HyperbolicModule<dim, Number> &hyperbolic_module_;
  const Interface::ParabolicModule<dim, Number> &parabolic_module_;
  StateVector<Number> temp_;
};


/*
 * Run a Simulation with the time integrator TI, print the norms of the
 * final state, and return them.
 */
template <template <int, typename> class TI>
std::pair<double, double> run(const std::string &name)
{
  std::stringstream parameters;
  parameters << "subsection A - Scheduler\n"
             << "  set basename   = " << name << "\n"
             << "  set final time = 0.1\n"
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
             << "end\n";

  /* The parameters of the TimeIntegrator of ryujin: */
  if constexpr (std::is_same_v<TI<dim, Number>, TimeIntegrator<dim, Number>>)
    parameters << "subsection H - TimeIntegrator\n"
               << "  set cfl min               = 0.2\n"
               << "  set cfl max               = 0.2\n"
               << "  set cfl recovery strategy = none\n"
               << "  set time stepping scheme  = erk 11\n"
               << "end\n";

  std::pair<double, double> norms;

  {
    Simulation<HyperbolicDescription, ParabolicDescription, dim, Number, TI>
        simulation(MPI_COMM_WORLD);
    Scheduler<Number> scheduler(
        MPI_COMM_WORLD, {&simulation}, {}, [](const std::string &) {});

    ParameterAcceptor::initialize(parameters);
    scheduler.run();

    const auto &final_state = simulation.state_vector().state;
    final_state.template copy_to_memory_space<dealii::MemorySpace::Host>();

    const auto &state = final_state.deal_ii_vector();
    norms = {state.l1_norm(), state.l2_norm()};

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << name << ": "
                << simulation.time_integrator().time_stepping_scheme_name()
                << ", l1 = " << std::scientific << std::setprecision(8)
                << norms.first << ", l2 = " << norms.second << std::endl;
  }

  ParameterAcceptor::clear();
  return norms;
}


int main(int argc, char *argv[])
{
  Environment environment(argc, argv);

  const auto stock = run<TimeIntegrator>("stock");
  const auto custom = run<ForwardEuler>("custom");

  /* Both computations perform the same arithmetic: */
  AssertThrow(stock == custom, ExcMessage("The results differ."));

  return 0;
}
