#ifndef POSTPROCESSOR_TEMPLATE_H
#define POSTPROCESSOR_TEMPLATE_H

#include "helper.h"
#include "postprocessor.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <atomic>
#include <fstream>

namespace grendel
{
  using namespace dealii;

  template <>
  const std::array<std::string, 2> Postprocessor<1, double>::component_names{
      "schlieren", "alpha"};

  template <>
  const std::array<std::string, 3> Postprocessor<2, double>::component_names{
      "schlieren", "vorticity", "alpha"};

  template <>
  const std::array<std::string, 5> Postprocessor<3, double>::component_names{
      "schlieren", "vorticity_1", "vorticity_2", "vorticity_3", "alpha"};

  template <>
  const std::array<std::string, 2> Postprocessor<1, float>::component_names{
      "schlieren", "alpha"};

  template <>
  const std::array<std::string, 3> Postprocessor<2, float>::component_names{
      "schlieren", "vorticity", "alpha"};

  template <>
  const std::array<std::string, 5> Postprocessor<3, float>::component_names{
      "schlieren", "vorticity_1", "vorticity_2", "vorticity_3", "alpha"};


  template <int dim, typename Number>
  Postprocessor<dim, Number>::Postprocessor(
      const MPI_Comm &mpi_communicator,
      const grendel::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Postprocessor"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , offline_data_(&offline_data)
  {
    schlieren_beta_ = 10.;
    add_parameter("schlieren beta",
                  schlieren_beta_,
                  "Beta factor used in the Schlieren postprocessor");

    coarsening_level_ = 0;
    add_parameter(
        "coarsening level",
        coarsening_level_,
        "Number of coarsening steps applied before writing pvtu/vtu output");
  }


  template <int dim, typename Number>
  void Postprocessor<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    deallog << "Postprocessor<dim, Number>::prepare()" << std::endl;
#endif

    const auto &partitioner = offline_data_->partitioner();

    for (auto &it : U_)
      it.reinit(partitioner);

    for (auto &it : quantities_)
      it.reinit(partitioner);

    /* Prepare triangulation and dof_handler for coarsened output: */

    AssertThrow(coarsening_level_ == 0, dealii::ExcNotImplemented());
  }


  template <int dim, typename Number>
  void Postprocessor<dim, Number>::compute(const vector_type &U,
                                           const scalar_type &alpha)
  {
#ifdef DEBUG_OUTPUT
    deallog << "Postprocessor<dim, Number>::compute()" << std::endl;
#endif

    const auto &affine_constraints = offline_data_->affine_constraints();
    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_normal_map = offline_data_->boundary_normal_map();

    const auto n_locally_owned = offline_data_->n_locally_owned();

    /*
     * Step 1: Copy the current state vector over to output_vector:
     */

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      U_[i] = U[i];
    }

    /*
     * Step 2: Compute r_i and r_i_max, r_i_min:
     */

    std::atomic<Number> r_i_max{0.};
    std::atomic<Number> r_i_min{std::numeric_limits<Number>::infinity()};

    {
      GRENDEL_PARALLEL_REGION_BEGIN

      Number r_i_max_on_subrange = 0.;
      Number r_i_min_on_subrange = std::numeric_limits<Number>::infinity();

      GRENDEL_OMP_FOR
      for (unsigned int i = 0; i < n_locally_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom */
        if (row_length == 1)
          continue;

        Tensor<1, dim, Number> r_i;
        curl_type vorticity;

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const unsigned int j = js[col_idx];

          const auto U_j = gather(U, j);
          const auto m_j = ProblemDescription<dim, Number>::momentum(U_j);

          const auto c_ij = cij_matrix.get_tensor(i, col_idx);

          r_i += c_ij * U_j[0];

          if constexpr (dim == 2) {
            vorticity[0] += cross_product_2d(c_ij) * m_j;
          } else if constexpr (dim == 3) {
            vorticity += cross_product_3d(c_ij, m_j);
          }
        }

        /* Fix up boundaries: */

        const auto bnm_it = boundary_normal_map.find(i);
        if (bnm_it != boundary_normal_map.end()) {
          const auto [normal, id, _] = bnm_it->second;
          if (id == Boundary::slip) {
            r_i -= 1. * (r_i * normal) * normal;
          } else {
            /* FIXME: This is not particularly elegant. On all other
             * boundary types, we simply set r_i to zero. */
            r_i = 0.;
          }
        }

        /* Populate quantities: */

        const Number rho_i = U[0].local_element(i);
        const Number m_i = lumped_mass_matrix.local_element(i);

        Tensor<1, n_quantities, Number> quantities;

        quantities[0] = r_i.norm() / m_i;

        vorticity /= (m_i * rho_i);
        if constexpr (dim == 2) {
          quantities[1] = vorticity[0];
        } else if constexpr (dim == 3) {
          quantities[1] = vorticity[0];
          quantities[2] = vorticity[1];
        }

        quantities[n_quantities - 1] = alpha.local_element(i);

        r_i_max_on_subrange = std::max(r_i_max_on_subrange, quantities[0]);
        r_i_min_on_subrange = std::min(r_i_min_on_subrange, quantities[0]);

        scatter(quantities_, quantities, i);
      }

      /* Synchronize over all threads: */

      Number current_r_i_max = r_i_max.load();
      while (
          current_r_i_max < r_i_max_on_subrange &&
          !r_i_max.compare_exchange_weak(current_r_i_max, r_i_max_on_subrange))
        ;

      Number current_r_i_min = r_i_min.load();
      while (
          current_r_i_min > r_i_min_on_subrange &&
          !r_i_min.compare_exchange_weak(current_r_i_min, r_i_min_on_subrange))
        ;

      GRENDEL_PARALLEL_REGION_END
    }

    /* And synchronize over all processors: */

    r_i_max.store(Utilities::MPI::max(r_i_max.load(), mpi_communicator_));
    r_i_min.store(Utilities::MPI::min(r_i_min.load(), mpi_communicator_));

    /*
     * Step 3: Normalize schlieren:
     */

    {
      GRENDEL_PARALLEL_REGION_BEGIN

      GRENDEL_OMP_FOR
      for (unsigned int i = 0; i < n_locally_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom */
        if (row_length == 1)
          continue;

        const auto r_i = quantities_[0].local_element(i);
        quantities_[0].local_element(i) =
            Number(1.) -
            std::exp(-schlieren_beta_ * (r_i - r_i_min) / (r_i_max - r_i_min));
      }

      GRENDEL_PARALLEL_REGION_END
    }

    /*
     * Step 4: Fix up constraints and distribute:
     */

    for (auto &it : U_) {
      affine_constraints.distribute(it);
      it.update_ghost_values();
    }

    for (auto &it : quantities_) {
      affine_constraints.distribute(it);
      it.update_ghost_values();
    }
  }


  namespace
  {
    template <int dim, typename Number>
    void interpolate_to_coarser_mesh(
        const dealii::InterGridMap<dealii::DoFHandler<dim>> &intergridmap,
        const dealii::LinearAlgebra::distributed::Vector<Number> &u_1,
        dealii::Vector<Number> &u_2)
    {
      const auto &dof1 = intergridmap.get_source_grid();
      auto cell_1 = dof1.begin();
      const auto endc1 = dof1.end();

      Vector<Number> cache;
      cache.reinit(cell_1->get_fe().dofs_per_cell);

      for (; cell_1 != endc1; ++cell_1) {
        const auto cell_2 = intergridmap[cell_1];

        if (cell_1->level() != cell_2->level())
          continue;

        if (!cell_1->active() && !cell_2->active())
          continue;

        /*
         * We have to skip artificial cells on the first (distributed)
         * triangulation:
         */
        auto cell = cell_1;
        for (; !cell->active(); cell = cell->child(0))
          ;
        if (!cell->is_locally_owned())
          continue;

        cell_1->get_interpolated_dof_values(
            u_1, cache, cell_2->active_fe_index());
        cell_2->set_dof_values_by_interpolation(
            cache, u_2, cell_2->active_fe_index());
      }
    }

  } /* namespace */


  template <int dim, typename Number>
  void Postprocessor<dim, Number>::write_out_vtu(std::string name,
                                                 Number t,
                                                 unsigned int cycle)
  {
    constexpr auto problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;
    constexpr auto n_quantities = Postprocessor<dim, Number>::n_quantities;

    const auto &discretization = offline_data_->discretization();
    const auto &mapping = discretization.mapping();
    const auto &triangulation = discretization.triangulation();

    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(offline_data_->dof_handler());

    for (unsigned int i = 0; i < problem_dimension; ++i)
      data_out.add_data_vector(
          U_[i], ProblemDescription<dim, Number>::component_names[i]);
    for (unsigned int i = 0; i < n_quantities; ++i)
      data_out.add_data_vector(quantities_[i], component_names[i]);

    data_out.build_patches(mapping, discretization.finite_element().degree - 1);

    DataOutBase::VtkFlags flags(
        t, cycle, true, DataOutBase::VtkFlags::best_speed);
    data_out.set_flags(flags);

    const auto filename = [&](const unsigned int i) -> std::string {
      const auto seq = dealii::Utilities::int_to_string(i, 4);
      return name + "-" + seq + ".vtu";
    };

    /* Write out local vtu: */

    const unsigned int i = triangulation.locally_owned_subdomain();
    std::ofstream output(filename(i));
    data_out.write_vtu(output);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {
      /* Write out pvtu control file: */

      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator_);
           ++i)
        filenames.push_back(filename(i));

      std::ofstream output(name + ".pvtu");
      data_out.write_pvtu_record(output, filenames);
    }
  }

} /* namespace grendel */

#endif /* POSTPROCESSOR_TEMPLATE_H */
