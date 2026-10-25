//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include "skeleton.h"

#include <ryujin/base/convenience_macros.h>
#include <ryujin/concept/hyperbolic_description.h>
#include <ryujin/concept/parabolic_description.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/stub/parabolic_module.h>
#include <ryujin/typed/hyperbolic_module.template.h>
#include <ryujin/typed/initial_values.template.h>
#include <ryujin/typed/postprocessor.h>
#include <ryujin/typed/solution_transfer.template.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <concepts>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

//
// Test the layout of an additive split of two hyperbolic systems:
//
//  - A SplitSystem derives from the main hyperbolic system (and thus
//    forwards its view, names, and parameters) and owns the subordinate
//    hyperbolic system. Both are constructed with the same subsection.
//
//  - The first (main) and the second (subordinate) hyperbolic description
//    each name their own compute kernels but share the InitialValues of
//    the SplitSystem.
//
//  - The description that the Simulation uses names the SplitSystem,
//    the shared InitialValues, the limiter of the main system (for the
//    SolutionTransfer), and an AdditiveModule, but no compute kernels.
//
// SplitSystem and AdditiveModule are skeletons: the test checks that the
// layout satisfies all concepts and that a Simulation can be constructed
// and initialized.
//

using namespace ryujin;


template <typename S1, typename S2, int dim>
constexpr bool same_layout =
    std::is_same_v<typename S1::template View<dim, NUMBER>::state_type,
                   typename S2::template View<dim, NUMBER>::state_type> &&
    std::is_same_v<
        typename S1::template View<dim, NUMBER>::initial_precomputed_type,
        typename S2::template View<dim, NUMBER>::initial_precomputed_type>;


/*
 * A main hyperbolic system together with a subordinate hyperbolic system
 * operating on the same state layout.
 */
template <typename Main, typename S>
class SplitSystem : public Main
{
public:
  using Subordinate = S;

  static_assert(same_layout<Main, Subordinate, 1> &&
                    same_layout<Main, Subordinate, 2> &&
                    same_layout<Main, Subordinate, 3>,
                "The main and the subordinate hyperbolic system must have "
                "the same state type and initial precomputed type.");

  SplitSystem(const std::string &subsection = "/HyperbolicSystem")
      : Main(subsection)
      , subordinate_(subsection)
  {
  }

  ACCESSOR_READ_ONLY(subordinate)

private:
  Subordinate subordinate_;
};


/*
 * An additive split of a first (main) and a second (subordinate)
 * hyperbolic module.
 */
template <typename HD1, typename HD2, int dim, typename Number>
  requires std::same_as<typename HD1::HyperbolicSystem::Subordinate,
                        typename HD2::HyperbolicSystem>
class AdditiveModule final : public Interface::HyperbolicModule<dim, Number>
{
public:
  using HyperbolicSystem = typename HD1::HyperbolicSystem;

  using InitialValues = typename HD1::template InitialValues<dim, Number>;

  using FirstModule = HyperbolicModule<HD1, dim, Number>;

  using SecondModule = HyperbolicModule<HD2, dim, Number>;

  using StageVectors =
      typename Interface::HyperbolicModule<dim, Number>::StageVectors;

  using StageWeights =
      typename Interface::HyperbolicModule<dim, Number>::StageWeights;

  AdditiveModule(const MPIEnsemble &mpi_ensemble,
                 const OfflineData<dim, Number> &offline_data,
                 const HyperbolicSystem &hyperbolic_system,
                 const InitialValues &initial_values,
                 const std::string &subsection = "/HyperbolicModule")
      : Interface::HyperbolicModule<dim, Number>(
            FirstModule::problem_dimension,
            FirstModule::n_precomputed_values,
            subsection)
      , first_(mpi_ensemble,
               offline_data,
               hyperbolic_system,
               initial_values,
               subsection + "/first stage")
      , second_(mpi_ensemble,
                offline_data,
                hyperbolic_system.subordinate(),
                initial_values,
                subsection + "/second stage")
  {
  }

  void prepare() override
  {
    first_.prepare();
    second_.prepare();
  }

  void reinit_state_vector(StateVector<Number> &state_vector) const override
  {
    first_.reinit_state_vector(state_vector);
  }

  void prepare_state_vector(StateVector<Number> &state_vector,
                            Number t) const override
  {
    first_.prepare_state_vector(state_vector, t);
  }

  void fill_precomputed_values(StateVector<Number> &state_vector) const override
  {
    first_.fill_precomputed_values(state_vector);
  }

  Number forward_euler_step(const StateVector<Number> & /*old_state_vector*/,
                            StageVectors /*stage_state_vectors*/,
                            StageWeights /*stage_weights*/,
                            StateVector<Number> & /*new_state_vector*/,
                            Number /*t*/,
                            Number /*tau*/,
                            Number /*tau_max*/) const override
  {
    AssertThrow(false, dealii::ExcNotImplemented());
    return Number(0.);
  }

  void print_solver_statistics(std::ostream & /*output*/) const override {}

  const MultiComponentVector<Number> &initial_precomputed() const
  {
    return first_.initial_precomputed();
  }

  ACCESSOR_READ_ONLY(first)

  ACCESSOR_READ_ONLY(second)

private:
  FirstModule first_;
  SecondModule second_;
};


using MainSystem = Skeleton::HyperbolicSystem<2>;
using SubordinateSystem = Skeleton::HyperbolicSystem<3>;


struct FirstDescription {
  using HyperbolicSystem = SplitSystem<MainSystem, SubordinateSystem>;

  template <int dim, typename Number>
  using InitialValues = ryujin::InitialValues<HyperbolicSystem, dim, Number>;

  template <typename ScalarNumber = double>
  using Indicator = Skeleton::Indicator<HyperbolicSystem, ScalarNumber>;

  template <typename ScalarNumber = double>
  using Limiter = Skeleton::Limiter<HyperbolicSystem, ScalarNumber>;

  template <typename ScalarNumber = double>
  using WaveSpeedEstimator =
      Skeleton::WaveSpeedEstimator<HyperbolicSystem, ScalarNumber>;
};


struct SecondDescription {
  using HyperbolicSystem = SubordinateSystem;

  template <int dim, typename Number>
  using InitialValues = FirstDescription::InitialValues<dim, Number>;

  template <typename ScalarNumber = double>
  using Indicator = Skeleton::Indicator<HyperbolicSystem, ScalarNumber>;

  template <typename ScalarNumber = double>
  using Limiter = Skeleton::Limiter<HyperbolicSystem, ScalarNumber>;

  template <typename ScalarNumber = double>
  using WaveSpeedEstimator =
      Skeleton::WaveSpeedEstimator<HyperbolicSystem, ScalarNumber>;
};


struct Description {
  using HyperbolicSystem = FirstDescription::HyperbolicSystem;

  template <int dim, typename Number>
  using InitialValues = FirstDescription::InitialValues<dim, Number>;

  template <typename ScalarNumber = double>
  using Limiter = FirstDescription::Limiter<ScalarNumber>;

  template <int dim, typename Number>
  using HyperbolicModule =
      AdditiveModule<FirstDescription, SecondDescription, dim, Number>;
};


using HyperbolicSystem = Description::HyperbolicSystem;

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

  template class HyperbolicModule<FirstDescription, 1, NUMBER>;
  template class HyperbolicModule<FirstDescription, 2, NUMBER>;
  template class HyperbolicModule<FirstDescription, 3, NUMBER>;

  template class HyperbolicModule<SecondDescription, 1, NUMBER>;
  template class HyperbolicModule<SecondDescription, 2, NUMBER>;
  template class HyperbolicModule<SecondDescription, 3, NUMBER>;

  template class SolutionTransfer<Description, 1, NUMBER>;
  template class SolutionTransfer<Description, 2, NUMBER>;
  template class SolutionTransfer<Description, 3, NUMBER>;

  template class Postprocessor<Description, 1, NUMBER>;
  template class Postprocessor<Description, 2, NUMBER>;
  template class Postprocessor<Description, 3, NUMBER>;
} // namespace ryujin


template <typename HD, int dim>
constexpr bool has_kernels =
    Concept::HyperbolicModuleDescription<HD, dim, NUMBER>;


template <int dim>
void test()
{
  using PD = StubParabolicDescription;

  constexpr bool hyperbolic =
      Concept::HyperbolicDescription<Description, dim, NUMBER>;
  constexpr bool parabolic =
      Concept::ParabolicDescription<PD, Description, dim, NUMBER>;

  std::cout << "dim = " << dim << std::endl;
  std::cout << "  Description:" << std::endl;
  std::cout << "    HyperbolicDescription:       " << hyperbolic << std::endl;
  std::cout << "    ParabolicDescription:        " << parabolic << std::endl;
  std::cout << "    HyperbolicModuleDescription: "
            << has_kernels<Description, dim> << std::endl;
  std::cout << "  FirstDescription:" << std::endl;
  std::cout << "    HyperbolicModuleDescription: "
            << has_kernels<FirstDescription, dim> << std::endl;
  std::cout << "  SecondDescription:" << std::endl;
  std::cout << "    HyperbolicModuleDescription: "
            << has_kernels<SecondDescription, dim> << std::endl;

  {
    Simulation<Description, PD, dim, NUMBER> simulation(MPI_COMM_SELF);

    /* A single entry sets the parameter of both hyperbolic systems: */
    std::stringstream parameters;
    parameters << "subsection B - Equation\n"
               << "  set skeleton parameter = 2.5\n"
               << "end\n";
    dealii::ParameterAcceptor::initialize(parameters);

    const auto &system = simulation.hyperbolic_system();
    const auto &split_system = dynamic_cast<const HyperbolicSystem &>(system);
    const auto &module = simulation.hyperbolic_module();

    using Module = Description::HyperbolicModule<dim, NUMBER>;
    constexpr auto n_first = Module::FirstModule::n_precomputed_values;
    constexpr auto n_second = Module::SecondModule::n_precomputed_values;

    std::cout << "  Simulation:" << std::endl;
    std::cout << "    main parameter:              " << split_system.parameter()
              << std::endl;
    std::cout << "    subordinate parameter:       "
              << split_system.subordinate().parameter() << std::endl;
    std::cout << "    number of state names:       "
              << system.state_names(dim).size() << std::endl;
    std::cout << "    module precomputed values:   "
              << module.n_precomputed_values() << std::endl;
    std::cout << "    first stage precomputed:     " << n_first << std::endl;
    std::cout << "    second stage precomputed:    " << n_second << std::endl;
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
