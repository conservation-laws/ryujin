#ifndef OFFLINE_DATA_TEMPLATE_H
#define OFFLINE_DATA_TEMPLATE_H

#include "offline_data.h"
#include "scratch_data.h"

#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <boost/range/iterator_range.hpp>

namespace grendel
{
  using namespace dealii;


  template <int dim>
  OfflineData<dim>::OfflineData(
      const MPI_Comm &mpi_communicator,
      dealii::TimerOutput &computing_timer,
      const grendel::Discretization<dim> &discretization,
      const std::string &subsection /*= "OfflineData"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , discretization_(&discretization)
  {
  }


  template <int dim>
  void OfflineData<dim>::setup()
  {
    deallog << "OfflineData<dim>::setup()" << std::endl;

    dof_handler_.initialize(discretization_->triangulation(),
                            discretization_->finite_element());

    {
      TimerOutput::Scope t(computing_timer_, "offline_data - distribute dofs");
      DoFRenumbering::Cuthill_McKee(dof_handler_);

      locally_owned_ = dof_handler_.locally_owned_dofs();
      locally_relevant_.clear();
      DoFTools::extract_locally_relevant_dofs(dof_handler_, locally_relevant_);

      /*
       * Print out the DoF distribution
       */

      deallog << "        " << dof_handler_.n_dofs()
              << " global DoFs, local DoF distribution:" << std::endl;

      const auto this_mpi_process =
          Utilities::MPI::this_mpi_process(mpi_communicator_);
      const auto n_mpi_processes =
          Utilities::MPI::n_mpi_processes(mpi_communicator_);
      unsigned int n_locally_owned_dofs = dof_handler_.n_locally_owned_dofs();

      if (this_mpi_process > 0) {
        MPI_Send(
            &n_locally_owned_dofs, 1, MPI_UNSIGNED, 0, 0, mpi_communicator_);

      } else {

        deallog << "        ( " << n_locally_owned_dofs << std::flush;
        for (unsigned int p = 1; p < n_mpi_processes; ++p) {
          MPI_Recv(&n_locally_owned_dofs,
                   1,
                   MPI_UNSIGNED,
                   p,
                   0,
                   mpi_communicator_,
                   MPI_STATUS_IGNORE);
          deallog << " + " << n_locally_owned_dofs << std::flush;
        }
        deallog << " )" << std::endl;
      }
    }

    /*
     * We are not doing anything with constraints (yet).
     */

    affine_constraints_.clear();
    affine_constraints_.close();

    /*
     * We need a local view of a couple of matrices. Because they are never
     * used in a matrix-vector product, and in order to avoid unnecessary
     * overhead, we simply assemble these parts into a local
     * dealii::SparseMatrix<dim>.
     */

    {
      TimerOutput::Scope t(computing_timer_,
                           "offline_data - create sparsity pattern");
      const auto n_dofs = locally_relevant_.size();
      const auto dofs_per_cell =
          discretization_->finite_element().dofs_per_cell;

      DynamicSparsityPattern dsp(n_dofs, n_dofs);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      for (auto cell : dof_handler_.active_cell_iterators()) {
        /* iterate over locally owned cells and the ghost layer */
        if (cell->is_artificial())
          continue;

        cell->get_dof_indices(dof_indices);
        affine_constraints_.add_entries_local_to_global(dof_indices, dsp, true);
      }

      sparsity_pattern_.copy_from(dsp);
    }

    {
      TimerOutput::Scope t(computing_timer_,
                           "offline_data - setup matrices");
      mass_matrix_.reinit(sparsity_pattern_);
      lumped_mass_matrix_.reinit(sparsity_pattern_);
      norm_matrix_.reinit(sparsity_pattern_);
      for (auto &matrix : cij_matrix_)
        matrix.reinit(sparsity_pattern_);
      for (auto &matrix : nij_matrix_)
        matrix.reinit(sparsity_pattern_);
    }
  }


  template <int dim>
  void OfflineData<dim>::assemble()
  {
    deallog << "OfflineData<dim>::assemble()" << std::endl;

    mass_matrix_ = 0.;
    lumped_mass_matrix_ = 0.;
    norm_matrix_ = 0.;
    for (auto &matrix : cij_matrix_)
      matrix = 0.;
    for (auto &matrix : nij_matrix_)
      matrix = 0.;

    const unsigned int dofs_per_cell =
        discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = discretization_->quadrature().size();

    /*
     * First part: Assemble mass matrices and cij matrices:
     */

    /* The local, per-cell assembly routine: */

    const auto local_assemble_system =
        [&](const auto &cell, auto &scratch, auto &copy) {
          /* iterate over locally owned cells and the ghost layer */

          auto &is_artificial = copy.is_artificial_;
          auto &local_dof_indices = copy.local_dof_indices_;
          auto &cell_mass_matrix = copy.cell_mass_matrix_;
          auto &cell_lumped_mass_matrix = copy.cell_lumped_mass_matrix_;
          auto &cell_cij_matrix = copy.cell_cij_matrix_;

          auto &fe_values = scratch.fe_values_;

          is_artificial = cell->is_artificial();
          if (is_artificial)
            return;

          cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_lumped_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
          for (auto &matrix : cell_cij_matrix)
            matrix.reinit(dofs_per_cell, dofs_per_cell);

          fe_values.reinit(cell);
          local_dof_indices.resize(dofs_per_cell);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const auto JxW = fe_values.JxW(q_point);

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {

              const auto value_JxW = fe_values.shape_value(j, q_point) * JxW;
              const auto grad_JxW = fe_values.shape_grad(j, q_point) * JxW;

              cell_lumped_mass_matrix(j, j) += value_JxW;

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {

                const auto value = fe_values.shape_value(i, q_point);

                cell_mass_matrix(i, j) += value * value_JxW;

                for (unsigned int d = 0; d < dim; ++d)
                  cell_cij_matrix[d](i, j) += (value * grad_JxW)[d];

              } /* for i */
            }   /* for j */
          }     /* for q */
        };

    const auto copy_local_to_global = [&](const auto &copy) {
      const auto &is_artificial = copy.is_artificial_;
      const auto &local_dof_indices = copy.local_dof_indices_;
      const auto &cell_mass_matrix = copy.cell_mass_matrix_;
      const auto &cell_lumped_mass_matrix = copy.cell_lumped_mass_matrix_;
      const auto &cell_cij_matrix = copy.cell_cij_matrix_;

      if(is_artificial)
        return;

      affine_constraints_.distribute_local_to_global(
          cell_mass_matrix, local_dof_indices, mass_matrix_);

      affine_constraints_.distribute_local_to_global(
          cell_lumped_mass_matrix, local_dof_indices, lumped_mass_matrix_);

      for (int k = 0; k < dim; ++k) {
        affine_constraints_.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, cij_matrix_[k]);
        affine_constraints_.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, nij_matrix_[k]);
      }
    };

    /*
     * And run a workstream to assemble the matrix.
     *
     * We need a graph coloring for the cells exactly once (the TimeStep
     * iterates over degrees of freedom without conflicts). Thus, construct
     * a graph coloring locally instead of caching it.
     */

#if 0
    // FIXME: The graph coloring is slow
    deallog << "        construct graph" << std::endl;

    const auto get_conflict_indices = [&](auto &cell) {
      if (cell->is_artificial())
        return std::vector<types::global_dof_index>();

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      return local_dof_indices;
    };

    const auto graph = GraphColoring::make_graph_coloring(
        dof_handler_.begin_active(), dof_handler_.end(), get_conflict_indices);
#endif

    {
      deallog << "        assemble mass matrices and c_ijs" << std::endl;
      TimerOutput::Scope t(computing_timer_, "offline_data - assemble c_ij");

      WorkStream::run(dof_handler_.begin_active(),
                      dof_handler_.end(),
                      local_assemble_system,
                      copy_local_to_global,
                      AssemblyScratchData<dim>(*discretization_),
                      AssemblyCopyData<dim>());
    }

    /*
     * Second part: Compute norms and n_ijs
     */

    const auto local_compute_norms = [&](const auto &it, auto &, auto &) {
      const auto row_index = *it;

      std::for_each(sparsity_pattern_.begin(row_index),
                    sparsity_pattern_.end(row_index),
                    [&](const auto &it) {
                      const auto col_index = it.column();
                      double norm = 0;
                      for (const auto &matrix : cij_matrix_) {
                        const auto value = matrix(row_index, col_index);
                        norm += value * value;
                      }
                      norm_matrix_(row_index, col_index) = std::sqrt(norm);
                    });

      for (auto &matrix : nij_matrix_) {
        auto nij_entry = matrix.begin(row_index);
        std::for_each(norm_matrix_.begin(row_index),
                      norm_matrix_.end(row_index),
                      [&](const auto &it) {
                        const auto norm = it.value();
                        nij_entry->value() /= norm;
                        ++nij_entry;
                      });
      }
    };

    {
      deallog << "        compute |c_ij|s and n_ijs" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "offline_data - compute |c_ij| and n_ij");

      WorkStream::run(locally_relevant_.begin(),
                      locally_relevant_.end(),
                      local_compute_norms,
                      [](const auto &) {},
                      double(),
                      double());
    }

  }


  template <int dim>
  void OfflineData<dim>::clear()
  {
    dof_handler_.clear();
    sparsity_pattern_.reinit(0, 0, 0);
    affine_constraints_.clear();

    mass_matrix_.clear();
    lumped_mass_matrix_.clear();

    for (auto &matrix : cij_matrix_)
      matrix.clear();
  }

} /* namespace grendel */

#endif /* OFFLINE_DATA_TEMPLATE_H */