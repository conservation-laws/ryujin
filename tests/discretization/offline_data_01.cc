//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

//
// Regression test for everything that OfflineData computes. All printed
// quantities are independent of the numbering of degrees of freedom and of
// the summation order (up to round-off) so that the test can be used to
// verify a change of the setup algorithms. Bilinear forms such as
// sum_ij g(x_i) m_ij h(x_j) with two different functions g and h detect
// misplaced entries in a matrix.
//

using namespace ryujin;
using namespace dealii;

namespace
{
  /* Suppress round-off when a quantity vanishes. */
  void print(const std::string &name, const double value)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "  " << name << ": "
                << (std::abs(value) < 1.e-10 ? 0. : value) << std::endl;
  }

  double sum(const double value)
  {
    return Utilities::MPI::sum(value, MPI_COMM_WORLD);
  }

  template <int dim>
  double g(const Point<dim> &x)
  {
    double result = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      result += (d + 1) * x[d];
    return result;
  }

  template <int dim>
  double h(const Point<dim> &x)
  {
    return 1. + x.square();
  }

  /* Print the number of distinct values of a distributed index map. */
  void print_image_size(const std::string &name, const auto &index_map)
  {
    const std::vector<types::global_dof_index> local(index_map.begin(),
                                                     index_map.end());
    std::set<types::global_dof_index> image;
    for (const auto &part : Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
      image.insert(part.begin(), part.end());
    print(name, image.size());
  }
} // namespace


template <int dim>
void test(const std::string &name, const std::string &parameter_string)
{
  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);
  Discretization<dim> discretization(mpi_ensemble);
  OfflineData<dim, dim, NUMBER> offline_data(mpi_ensemble, discretization);

  std::stringstream parameters(parameter_string);
  ParameterAcceptor::initialize(parameters);

  discretization.prepare("test");
  offline_data.prepare(/*problem dimension*/ 1, /*precomputed values*/ 1);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << name << std::endl;

  const unsigned int n_owned = offline_data.n_locally_owned();
  const auto &support_points = offline_data.support_points();

  /* Exchange the support points so that we can evaluate them on ghosts: */

  std::vector<double> g_values(offline_data.n_locally_relevant());
  std::vector<double> h_values(offline_data.n_locally_relevant());
  {
    LinearAlgebra::distributed::Vector<double> g_vector, h_vector;
    g_vector.reinit(offline_data.scalar_partitioner());
    h_vector.reinit(offline_data.scalar_partitioner());
    for (unsigned int i = 0; i < n_owned; ++i) {
      g_vector.local_element(i) = g(support_points[i]);
      h_vector.local_element(i) = h(support_points[i]);
    }
    g_vector.update_ghost_values();
    h_vector.update_ghost_values();
    for (unsigned int i = 0; i < g_values.size(); ++i) {
      g_values[i] = g_vector.local_element(i);
      h_values[i] = h_vector.local_element(i);
    }
  }

  /* Sizes: */

  const auto sparsity = offline_data.sparsity_pattern_simd().view();

  double n_entries = 0.;
  double n_constrained = 0.;
  for (unsigned int i = 0; i < n_owned; ++i) {
    n_entries += sparsity.row_length(i);
    n_constrained += sparsity.row_length(i) == 1 ? 1. : 0.;
  }

  print("n_dofs", sum(n_owned));
  print("n_entries", sum(n_entries));
  print("n_constrained", sum(n_constrained));
  print("constraints", sum(offline_data.affine_constraints().n_constraints()));

  /* Support points and lumped mass: */

  const auto lumped_mass = offline_data.lumped_mass_matrix().template view<1>();
  const auto lumped_mass_inverse =
      offline_data.lumped_mass_matrix_inverse().template view<1>();

  double m_sum = 0., m_g = 0., m_inverse_h = 0.;
  for (unsigned int i = 0; i < n_owned; ++i) {
    if (sparsity.row_length(i) == 1)
      continue;
    const double m_i = lumped_mass.read_entry(i);
    m_sum += m_i;
    m_g += m_i * g_values[i];
    m_inverse_h += m_i * lumped_mass_inverse.read_entry(i) * h_values[i];
  }

  print("measure_of_omega", offline_data.measure_of_omega());
  print("lumped_mass", sum(m_sum));
  print("lumped_mass_g", sum(m_g));
  print("lumped_mass_inverse_h", sum(m_inverse_h));

  /* Matrices: */

  const auto print_matrix = [&](const std::string &matrix_name,
                                const auto &matrix) {
    const auto view = matrix.view();
    double norm = 0.;
    double form = 0.;
    for (unsigned int i = 0; i < n_owned; ++i) {
      const unsigned int stride = sparsity.stride_of_row(i);
      const unsigned int *js = sparsity.columns(i);
      for (unsigned int c = 0; c < sparsity.row_length(i); ++c) {
        const double entry = view.read_entry(i, c);
        norm += entry * entry;
        form += g_values[i] * entry * h_values[js[c * stride]];
      }
    }
    print(matrix_name + "_norm", sum(norm));
    print(matrix_name + "_form", sum(form));
  };

  print_matrix("mass", offline_data.mass_matrix());
  print_matrix("betaij", offline_data.betaij_matrix());
  if (offline_data.have_discontinuous_ansatz()) {
    print_matrix("mass_inverse", offline_data.mass_matrix_inverse());
    print_matrix("incidence", offline_data.incidence_matrix());
  }

  {
    const auto view = offline_data.cij_matrix().view();
    double norm = 0.;
    Tensor<1, dim> form;
    for (unsigned int i = 0; i < n_owned; ++i) {
      const unsigned int stride = sparsity.stride_of_row(i);
      const unsigned int *js = sparsity.columns(i);
      for (unsigned int c = 0; c < sparsity.row_length(i); ++c) {
        const Tensor<1, dim> entry = view.read_tensor(i, c);
        norm += entry.norm_square();
        form += g_values[i] * entry * h_values[js[c * stride]];
      }
    }
    print("cij_norm", sum(norm));
    for (unsigned int d = 0; d < dim; ++d)
      print("cij_form_" + std::to_string(d), sum(form[d]));
  }

  /* Boundary map and coupling pairs: */

  double boundary_mass_sum = 0., normal_mass_sum = 0., boundary_g = 0.;
  std::array<double, 8> entries_by_id{};
  Tensor<1, dim> closure;
  for (const auto &[i, normal, normal_mass, boundary_mass, id, position] :
       offline_data.boundary_map()) {
    boundary_mass_sum += boundary_mass;
    normal_mass_sum += normal_mass;
    boundary_g += boundary_mass * g(position);
    closure += normal_mass * Tensor<1, dim>(normal);
    entries_by_id.at(id) += 1.;
    (void)i;
  }

  print("boundary_map", sum(offline_data.boundary_map().size()));
  for (unsigned int id = 0; id < entries_by_id.size(); ++id)
    if (const double n = sum(entries_by_id[id]); n != 0.)
      print("boundary_map_id_" + std::to_string(id), n);
  print("boundary_indices", sum(offline_data.boundary_indices().size()));
  print("boundary_mass", sum(boundary_mass_sum));
  print("boundary_mass_g", sum(boundary_g));
  print("normal_mass", sum(normal_mass_sum));
  for (unsigned int d = 0; d < dim; ++d)
    print("closure_" + std::to_string(d), sum(closure[d]));

  double coupling_form = 0.;
  const auto &pairs = offline_data.coupling_boundary_pairs();
  const auto *pair = pairs.view();
  for (std::size_t k = 0; k < pairs.size(); ++k)
    coupling_form += g_values[pair[k].i] * h_values[pair[k].j];
  print("coupling_pairs", sum(pairs.size()));
  print("coupling_form", sum(coupling_form));

  /* Index maps: */

  print_image_size("dg_to_cg", offline_data.dg_to_cg_index_map());
  print_image_size("cg_to_dg", offline_data.cg_to_dg_index_map());

  /* Multigrid data: the lumped mass is stored in single precision. */

  const auto &level_mass = offline_data.level_lumped_mass_matrix();
  const auto &level_map = offline_data.level_boundary_map();
  for (unsigned int level = 0; level < level_mass.size(); ++level) {
    double level_boundary_mass = 0.;
    for (const auto &entry : level_map[level])
      level_boundary_mass += std::get<3>(entry);

    const auto label = "level_" + std::to_string(level);
    print(label + "_size", level_mass[level].size());
    print(label + "_mass",
          std::round(1.e4 * level_mass[level].l1_norm()) / 1.e4);
    print(label + "_boundary_map", sum(level_map[level].size()));
    print(label + "_boundary_mass", sum(level_boundary_mass));
  }

  ParameterAcceptor::clear();
}


namespace
{
  std::string domain_2d(const std::string &ansatz,
                        const unsigned int refinement,
                        const std::string &extra = "",
                        const std::string &domain_extra = "",
                        const std::string &bc_x = "slip",
                        const std::string &bc_y = "dirichlet")
  {
    return "subsection Discretization\n"
           "  set finite element ansatz = " +
           ansatz + "\n" +
           "  set geometry = rectangular domain\n"
           "  set mesh refinement = " +
           std::to_string(refinement) + "\n" + extra +
           "  subsection rectangular domain\n"
           "    set position bottom left = 0, 0\n"
           "    set position top right = 1, 2\n"
           "    set subdivisions y = 2\n" +
           domain_extra + "    set boundary condition left = " + bc_x +
           "\n    set boundary condition right = " + bc_x +
           "\n    set boundary condition bottom = " + bc_y +
           "\n    set boundary condition top = " + bc_y +
           "\n  end\n"
           "end\n";
  }

  std::string domain_3d(const std::string &ansatz,
                        const unsigned int refinement,
                        const std::string &extra = "",
                        const std::string &domain_extra = "")
  {
    return "subsection Discretization\n"
           "  set finite element ansatz = " +
           ansatz + "\n" +
           "  set geometry = rectangular domain\n"
           "  set mesh refinement = " +
           std::to_string(refinement) + "\n" + extra +
           "  subsection rectangular domain\n"
           "    set position bottom left = 0, 0, 0\n"
           "    set position top right = 1, 1, 2\n"
           "    set subdivisions x = 2\n" +
           domain_extra +
           "    set boundary condition left = slip\n"
           "    set boundary condition right = slip\n"
           "    set boundary condition bottom = dirichlet\n"
           "    set boundary condition top = dirichlet\n"
           "    set boundary condition back = no slip\n"
           "    set boundary condition front = no slip\n"
           "  end\n"
           "end\n";
  }
} // namespace


int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  std::cout << std::scientific << std::setprecision(10);

  const std::string hanging = "  set mesh local refinement = 1\n"
                              "  set mesh local refinement region = x - 0.5\n";
  const std::string transformed =
      "    set transformation = x + 0.1*sin(y); y + 0.1*x*x\n";

  test<2>("cG Q1", domain_2d("cG Q1", 3));
  test<2>("cG Q2", domain_2d("cG Q2", 2));
  test<2>("cG Q3", domain_2d("cG Q3", 2));
  test<2>("cG Q1 periodic", domain_2d("cG Q1", 3, "", "", "slip", "periodic"));
  test<2>("cG Q1 hanging", domain_2d("cG Q1", 2, hanging));
  test<2>("cG Q2 hanging", domain_2d("cG Q2", 2, hanging));
  test<2>("cG Q1 hanging periodic",
          domain_2d("cG Q1", 2, hanging, "", "slip", "periodic"));
  test<2>("cG Q2 transformed", domain_2d("cG Q2", 3, "", transformed));
  test<2>("dG Q1", domain_2d("dG Q1", 3));
  test<2>("dG Q2", domain_2d("dG Q2", 2));
  test<2>("dG Q1 periodic", domain_2d("dG Q1", 3, "", "", "slip", "periodic"));
  test<3>("cG Q1 3D", domain_3d("cG Q1", 2));
  test<3>("cG Q2 3D hanging", domain_3d("cG Q2", 0, hanging));
  test<3>(
      "cG Q1 3D transformed",
      domain_3d(
          "cG Q1", 2, "", "    set transformation = x; y; z + 0.25*z*x*x\n"));

  /* Enough cells for code paths that split up cell loops into chunks: */
  test<2>("cG Q1 large", domain_2d("cG Q1", 7));
  test<2>("dG Q1 large", domain_2d("dG Q1", 7));
}
