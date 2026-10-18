#include <hyperbolic_description.h>

#include <discretization.h>
#include <mpi_ensemble.h>
#include <offline_data.h>
#include <stub_parabolic_system.h>

#include <ryujin/typed/postprocessor.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iostream>
#include <sstream>

//
// Test the postprocessed quantities of the Postprocessor: We populate a
// state vector with a linear density and a linear, rotating velocity
// field. For linear fields the discrete gradient
//   1/m_i sum_j c_ij u_j
// is exact. Thus, "schlieren_rho" has to be equal to |grad rho| and
// "vorticity_v_1" equal to the curl of the velocity at every degree of
// freedom. In addition, we check that the gather variant of extract()
// rejects postprocessed quantities.
//

using namespace ryujin;
using namespace dealii;

using HyperbolicDescription = Euler::HyperbolicDescription;

constexpr int dim = 2;
using Number = NUMBER;

using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;
using ParabolicSystem = StubParabolicSystem;
using SystemView = typename HyperbolicSystem::template View<dim, Number>;

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* Set up a small rectangular domain and the corresponding offline data: */

  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);

  HyperbolicSystem hyperbolic_system;
  ParabolicSystem parabolic_system;
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, Number> offline_data(mpi_ensemble, discretization);

  std::stringstream parameters;
  parameters << "subsection Discretization\n"
             << "  set geometry        = rectangular domain\n"
             << "  set mesh refinement = 2\n"
             << "end\n";
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(SystemView::problem_dimension,
                       SystemView::n_precomputed_values);

  const auto &scalar_partitioner = offline_data.scalar_partitioner();
  const unsigned int n_owned = offline_data.n_locally_owned();

  /*
   * Populate the state vector at the support points:
   *   rho = 4 + 2 x + 3 y,   v = (-y, x),   E = 100.
   */

  StateVector<Number> state_vector;
  auto &[U, Q, U_dynamic, Q_dynamic] = state_vector;
  U.reinit_with_scalar_partitioner(scalar_partitioner,
                                   SystemView::problem_dimension,
                                   TransferPolicy::implicit_transfers);
  Q.reinit_with_scalar_partitioner(scalar_partitioner,
                                   SystemView::n_precomputed_values,
                                   TransferPolicy::implicit_transfers);

  MultiComponentVector<Number> initial_precomputed;
  initial_precomputed.reinit_with_scalar_partitioner(
      scalar_partitioner,
      SystemView::n_initial_precomputed_values,
      TransferPolicy::implicit_transfers);

  {
    const auto U_view = U.view<SystemView::problem_dimension>();

    const auto &dof_handler = offline_data.dof_handler();
    const auto &support_points = dof_handler.get_fe().get_unit_support_points();
    std::vector<dealii::types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_active_or_mg_dof_indices(local_dof_indices);

      const auto &mapping = discretization.mapping()[cell->active_fe_index()];

      for (unsigned int j = 0; j < local_dof_indices.size(); ++j) {
        const auto i =
            scalar_partitioner->global_to_local(local_dof_indices[j]);
        if (i >= n_owned)
          continue;

        const auto x =
            mapping.transform_unit_to_real_cell(cell, support_points[j]);

        const Number rho = 4. + 2. * x[0] + 3. * x[1];

        typename SystemView::state_type U_i;
        U_i[0] = rho;
        U_i[1] = -rho * x[1];
        U_i[2] = rho * x[0];
        U_i[dim + 1] = 100.;
        U_view.write_tensor(U_i, i);
      }
    }

    U_view.update_ghost_values();
  }

  /* Set up the postprocessor: */

  Postprocessor<HyperbolicDescription, dim, Number> postprocessor(
      offline_data, hyperbolic_system, parabolic_system, initial_precomputed);

  const std::vector<std::string> selected{
      "rho", "schlieren_rho", "vorticity_v_1"};
  const auto selection = postprocessor.prepare(selected);

  const auto components = postprocessor.extract(selection, state_vector);

  /* Compare against the exact values: */

  const std::vector<Number> exact{
      Number(0.), Number(std::sqrt(13.)), Number(2.)};

  for (unsigned int k = 1; k < selected.size(); ++k) {
    Number difference = 0.;
    for (unsigned int i = 0; i < n_owned; ++i)
      difference = std::max(
          difference, std::abs(components[k].local_element(i) - exact[k]));
    difference =
        Utilities::MPI::max(difference, mpi_ensemble.ensemble_communicator());

    std::cout << selected[k] << ": exact up to round-off: "
              << (difference < 1.e-5 ? "yes" : "no") << std::endl;
  }

  /* The gather variant of extract() rejects postprocessed quantities: */

  try {
    unsigned int index = 0;
    std::vector<Number> values(selected.size());
    postprocessor.extract(selection, state_vector, &index, 1, values.data());
    std::cout << "gather: no exception" << std::endl;
  } catch (const dealii::ExceptionBase &) {
    std::cout << "gather: exception" << std::endl;
  }

  /* A vorticity needs dim consecutive components: */

  try {
    postprocessor.prepare({"vorticity_p"});
    std::cout << "vorticity_p: no exception" << std::endl;
  } catch (const dealii::ExceptionBase &) {
    std::cout << "vorticity_p: exception" << std::endl;
  }
}
