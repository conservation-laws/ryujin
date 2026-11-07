//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/discretization/offline_data.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

//
// Test OfflineData::update_geometry(): After moving the mesh, the update
// has to keep the numbering, sparsity pattern and partitioners and has to
// reproduce all geometric data of a prepare() on the moved mesh, entry by
// entry. Moving the mesh back has to reproduce the initial data. The
// "merging corners" configuration changes the length of the boundary map.
//

using namespace ryujin;
using namespace dealii;

namespace
{
  using Data = std::map<std::string, std::vector<double>>;

  struct Snapshot {
    Data topology;
    Data geometry;
    double measure_of_omega;
    double n_boundary_map_entries;
  };

  template <int dim>
  using Transformation = std::function<Point<dim>(
      const typename Triangulation<dim>::cell_iterator &, const Point<dim> &)>;

  template <int dim, typename Number>
  void
  read_sparsity_and_matrices(const OfflineData<dim, dim, Number> &offline_data,
                             Snapshot &snapshot)
  {
    const unsigned int n_owned = offline_data.n_locally_owned();
    const auto sparsity = offline_data.sparsity_pattern_simd().view();

    auto &columns = snapshot.topology["sparsity pattern"];
    for (unsigned int i = 0; i < n_owned; ++i) {
      const unsigned int stride = sparsity.stride_of_row(i);
      const unsigned int *js = sparsity.columns(i);
      for (unsigned int c = 0; c < sparsity.row_length(i); ++c)
        columns.push_back(js[c * stride]);
      columns.push_back(-1.);
    }

    const auto read = [&](const std::string &name,
                          const auto &matrix,
                          const bool with_transposed) {
      const auto view = matrix.view();
      auto &entries = snapshot.geometry[name];
      for (unsigned int i = 0; i < n_owned; ++i)
        for (unsigned int c = 0; c < sparsity.row_length(i); ++c) {
          entries.push_back(view.read_entry(i, c));
          if (with_transposed)
            entries.push_back(view.read_transposed_entry(i, c));
        }
    };

    read("mass", offline_data.mass_matrix(), true);
    read("betaij", offline_data.betaij_matrix(), true);
    if (offline_data.have_discontinuous_ansatz()) {
      read("mass_inverse", offline_data.mass_matrix_inverse(), true);
      read("incidence", offline_data.incidence_matrix(), false);
    }

    const auto cij = offline_data.cij_matrix().view();
    auto &cij_entries = snapshot.geometry["cij"];
    for (unsigned int i = 0; i < n_owned; ++i)
      for (unsigned int c = 0; c < sparsity.row_length(i); ++c) {
        const Tensor<1, dim> entry = cij.read_tensor(i, c);
        const Tensor<1, dim> transposed = cij.read_transposed_tensor(i, c);
        for (unsigned int d = 0; d < dim; ++d) {
          cij_entries.push_back(entry[d]);
          cij_entries.push_back(transposed[d]);
        }
      }
  }


  template <int dim, typename Number>
  Snapshot take_snapshot(const OfflineData<dim, dim, Number> &offline_data)
  {
    Snapshot snapshot;
    read_sparsity_and_matrices(offline_data, snapshot);

    auto &topology = snapshot.topology;
    auto &geometry = snapshot.geometry;

    topology["index ranges"] = {double(offline_data.n_locally_owned()),
                                double(offline_data.n_locally_internal()),
                                double(offline_data.n_locally_relevant()),
                                double(offline_data.n_export_indices())};

    const auto copy_indices = [](auto &target, const auto &map) {
      target.assign(map.begin(), map.end());
    };
    copy_indices(topology["dg_to_cg"], offline_data.dg_to_cg_index_map());
    copy_indices(topology["cg_to_dg"], offline_data.cg_to_dg_index_map());

    const auto &boundary_indices = offline_data.boundary_indices();
    const auto *boundary_index = boundary_indices.view();
    topology["boundary indices"].assign(
        boundary_index, boundary_index + boundary_indices.size());

    const auto &pairs = offline_data.coupling_boundary_pairs();
    const auto *pair = pairs.view();
    auto &coupling = topology["coupling boundary pairs"];
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      coupling.push_back(pair[k].i);
      coupling.push_back(pair[k].col_idx);
      coupling.push_back(pair[k].j);
    }

    for (const auto &point : offline_data.support_points())
      for (unsigned int d = 0; d < dim; ++d)
        geometry["support points"].push_back(point[d]);

    const auto lumped = offline_data.lumped_mass_matrix().template view<1>();
    const auto inverse =
        offline_data.lumped_mass_matrix_inverse().template view<1>();
    for (unsigned int i = 0; i < offline_data.n_locally_relevant(); ++i) {
      geometry["lumped mass"].push_back(lumped.read_entry(i));
      geometry["lumped mass inverse"].push_back(inverse.read_entry(i));
    }

    const auto read_boundary_map = [&](auto &target, const auto &map) {
      for (const auto &[i, normal, normal_mass, boundary_mass, id, x] : map) {
        target.push_back(i);
        target.push_back(normal_mass);
        target.push_back(boundary_mass);
        target.push_back(id);
        for (unsigned int d = 0; d < dim; ++d) {
          target.push_back(normal[d]);
          target.push_back(x[d]);
        }
      }
    };

    read_boundary_map(geometry["boundary map"], offline_data.boundary_map());
    const auto &slots = offline_data.boundary_slots();
    geometry["boundary slots"].assign(slots.begin(), slots.end());

    const auto &level_mass = offline_data.level_lumped_mass_matrix();
    const auto &level_map = offline_data.level_boundary_map();
    for (unsigned int level = 0; level < level_mass.size(); ++level) {
      const auto &mass = level_mass[level];
      auto &target = geometry["level mass"];
      target.insert(target.end(), mass.begin(), mass.end());
      read_boundary_map(geometry["level boundary map"], level_map[level]);
    }

    snapshot.measure_of_omega = offline_data.measure_of_omega();
    snapshot.n_boundary_map_entries = Utilities::MPI::sum(
        double(offline_data.boundary_map().size()), MPI_COMM_WORLD);
    return snapshot;
  }


  /* The level mass is stored in single precision. */
  double tolerance(const std::string &name)
  {
    return name == "level mass" ? 1.e-4 : 1.e-10;
  }


  void print_comparison(const std::string &label,
                        const Data &left,
                        const Data &right)
  {
    /* The sets of keys coincide on all ranks. */
    std::string result;
    for (const auto &[name, left_values] : left) {
      const auto &right_values = right.at(name);
      double difference = 0.;
      if (left_values.size() != right_values.size())
        difference = std::numeric_limits<double>::infinity();
      else
        for (std::size_t k = 0; k < left_values.size(); ++k)
          difference =
              std::max(difference, std::abs(left_values[k] - right_values[k]));

      if (Utilities::MPI::max(difference, MPI_COMM_WORLD) > tolerance(name))
        result += " [" + name + "]";
    }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "  " << label << ":" << (result.empty() ? " ok" : result)
                << std::endl;
  }
} // namespace


template <int dim>
void test(const std::string &name,
          const std::string &parameter_string,
          const Transformation<dim> &transformation)
{
  MPIEnsemble mpi_ensemble(MPI_COMM_WORLD);

  Snapshot initial, updated, restored, fresh;
  bool partitioner_kept = false;

  {
    Discretization<dim> discretization(mpi_ensemble);
    OfflineData<dim, dim, NUMBER> offline_data(mpi_ensemble, discretization);
    std::stringstream parameters(parameter_string);
    ParameterAcceptor::initialize(parameters);

    discretization.prepare("test");
    offline_data.prepare(/*problem dimension*/ 1, /*precomputed values*/ 1);
    initial = take_snapshot(offline_data);
    const auto partitioner = offline_data.scalar_partitioner();

    discretization.transformation() = transformation;
    discretization.update_mapping();
    offline_data.update_geometry();
    updated = take_snapshot(offline_data);

    partitioner_kept = partitioner == offline_data.scalar_partitioner();

    discretization.transformation() = nullptr;
    discretization.update_mapping();
    offline_data.update_geometry();
    restored = take_snapshot(offline_data);

    ParameterAcceptor::clear();
  }

  {
    Discretization<dim> discretization(mpi_ensemble);
    OfflineData<dim, dim, NUMBER> offline_data(mpi_ensemble, discretization);
    std::stringstream parameters(parameter_string);
    ParameterAcceptor::initialize(parameters);

    discretization.prepare("test");
    discretization.transformation() = transformation;
    discretization.update_mapping();
    offline_data.prepare(/*problem dimension*/ 1, /*precomputed values*/ 1);
    fresh = take_snapshot(offline_data);

    ParameterAcceptor::clear();
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::cout << name << std::endl;
    std::cout << "  measure of omega: " << initial.measure_of_omega << " -> "
              << updated.measure_of_omega << std::endl;
    std::cout << "  boundary map entries: " << initial.n_boundary_map_entries
              << " -> " << updated.n_boundary_map_entries << std::endl;
    std::cout << "  partitioner kept: " << partitioner_kept << std::endl;
  }

  print_comparison("topology unchanged", initial.topology, updated.topology);
  print_comparison(
      "topology matches prepare", fresh.topology, updated.topology);
  print_comparison(
      "geometry matches prepare", fresh.geometry, updated.geometry);
  print_comparison("geometry restored", initial.geometry, restored.geometry);
}


namespace
{
  std::string domain(const std::string &ansatz,
                     const unsigned int refinement,
                     const std::string &bc_x = "slip",
                     const std::string &bc_y = "dirichlet",
                     const std::string &extra = "")
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
           "    set subdivisions y = 2\n"
           "    set boundary condition left = " +
           bc_x + "\n    set boundary condition right = " + bc_x +
           "\n    set boundary condition bottom = " + bc_y +
           "\n    set boundary condition top = " + bc_y +
           "\n  end\n"
           "end\n";
  }
} // namespace


int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* A transformation that is periodic in y direction: */
  const Transformation<2> deformed = [](const auto &,
                                        const Point<2> &x) -> Point<2> {
    return {x[0] + 0.1 * std::sin(numbers::PI * x[1]),
            x[1] + 0.1 * x[0] * x[0]};
  };

  const Transformation<2> sheared = [](const auto &,
                                       const Point<2> &x) -> Point<2> {
    return {x[0], x[1] + 2. * x[0]};
  };

  const std::string hanging = "  set mesh local refinement = 1\n"
                              "  set mesh local refinement region = x - 0.5\n";

  test<2>("cG Q1", domain("cG Q1", 3), deformed);
  test<2>("cG Q2", domain("cG Q2", 2), deformed);
  test<2>("cG Q1 periodic", domain("cG Q1", 3, "slip", "periodic"), deformed);
  test<2>("cG Q2 hanging",
          domain("cG Q2", 2, "slip", "dirichlet", hanging),
          deformed);
  test<2>("cG Q1 merging corners", domain("cG Q1", 3, "slip", "slip"), sheared);
  test<2>("dG Q1", domain("dG Q1", 3), deformed);
  test<2>("dG Q2 periodic", domain("dG Q2", 2, "slip", "periodic"), deformed);

  test<3>("cG Q1 3D",
          "subsection Discretization\n"
          "  set finite element ansatz = cG Q1\n"
          "  set geometry = rectangular domain\n"
          "  set mesh refinement = 2\n"
          "  subsection rectangular domain\n"
          "    set position bottom left = 0, 0, 0\n"
          "    set position top right = 1, 1, 2\n"
          "    set boundary condition left = slip\n"
          "    set boundary condition right = slip\n"
          "    set boundary condition bottom = dirichlet\n"
          "    set boundary condition top = dirichlet\n"
          "    set boundary condition back = no slip\n"
          "    set boundary condition front = no slip\n"
          "  end\n"
          "end\n",
          [](const auto &, const Point<3> &x) -> Point<3> {
            return {x[0], x[1], x[2] + 0.25 * x[2] * x[0] * x[0]};
          });
}
