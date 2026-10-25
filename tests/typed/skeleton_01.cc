//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include "skeleton.h"

#include <ryujin/concept/hyperbolic_description.h>
#include <ryujin/concept/parabolic_description.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/stub/parabolic_module.h>
#include <ryujin/typed/hyperbolic_module.template.h>
#include <ryujin/typed/initial_values.template.h>
#include <ryujin/typed/postprocessor.h>
#include <ryujin/typed/solution_transfer.template.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iostream>
#include <sstream>

//
// Test the skeleton hyperbolic description: it satisfies all concepts,
// all typed classes instantiate with it, and a Simulation can be
// constructed and initialized from a parameter file.
//

using namespace ryujin;

using HyperbolicSystem = Skeleton::HyperbolicSystem<2>;
using HD = Skeleton::HyperbolicDescription<HyperbolicSystem>;

namespace ryujin
{
  template <>
  auto InitialStateLibrary<HyperbolicSystem, 1, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return Skeleton::initial_states<HyperbolicSystem, 1, NUMBER>();
  }

  template <>
  auto InitialStateLibrary<HyperbolicSystem, 2, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return Skeleton::initial_states<HyperbolicSystem, 2, NUMBER>();
  }

  template <>
  auto InitialStateLibrary<HyperbolicSystem, 3, NUMBER>::defaults()
      -> std::vector<Creator>
  {
    return Skeleton::initial_states<HyperbolicSystem, 3, NUMBER>();
  }

  /* instantiations */

  template class InitialValues<HyperbolicSystem, 1, NUMBER>;
  template class InitialValues<HyperbolicSystem, 2, NUMBER>;
  template class InitialValues<HyperbolicSystem, 3, NUMBER>;

  template class HyperbolicModule<HD, 1, NUMBER>;
  template class HyperbolicModule<HD, 2, NUMBER>;
  template class HyperbolicModule<HD, 3, NUMBER>;

  template class SolutionTransfer<HD, 1, NUMBER>;
  template class SolutionTransfer<HD, 2, NUMBER>;
  template class SolutionTransfer<HD, 3, NUMBER>;

  template class Postprocessor<HD, 1, NUMBER>;
  template class Postprocessor<HD, 2, NUMBER>;
  template class Postprocessor<HD, 3, NUMBER>;
} // namespace ryujin


template <int dim>
void test()
{
  using PD = StubParabolicDescription;

  constexpr bool hyperbolic = Concept::HyperbolicDescription<HD, dim, NUMBER>;
  constexpr bool kernels =
      Concept::HyperbolicModuleDescription<HD, dim, NUMBER>;
  constexpr bool parabolic = Concept::ParabolicDescription<PD, HD, dim, NUMBER>;

  std::cout << "dim = " << dim << std::endl;
  std::cout << "  HyperbolicDescription:       " << hyperbolic << std::endl;
  std::cout << "  HyperbolicModuleDescription: " << kernels << std::endl;
  std::cout << "  ParabolicDescription:        " << parabolic << std::endl;

  {
    Simulation<HD, PD, dim, NUMBER> simulation(MPI_COMM_SELF);

    std::stringstream parameters;
    parameters << "subsection B - Equation\n"
               << "  set skeleton parameter = 2.5\n"
               << "end\n";
    dealii::ParameterAcceptor::initialize(parameters);

    const auto &hyperbolic_system =
        dynamic_cast<const HyperbolicSystem &>(simulation.hyperbolic_system());
    const auto &hyperbolic_module = simulation.hyperbolic_module();

    std::cout << "  problem name:          " << hyperbolic_system.problem_name()
              << std::endl;
    std::cout << "  skeleton parameter:    " << hyperbolic_system.parameter()
              << std::endl;
    std::cout << "  problem dimension:     "
              << hyperbolic_module.problem_dimension() << std::endl;
    std::cout << "  n precomputed values:  "
              << hyperbolic_module.n_precomputed_values() << std::endl;
  }

  dealii::ParameterAcceptor::clear();
}


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::boolalpha;

  test<1>();
  test<2>();
  test<3>();

  return 0;
}
