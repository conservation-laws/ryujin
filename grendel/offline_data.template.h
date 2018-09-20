#ifndef OFFLINE_DATA_TEMPLATE_H
#define OFFLINE_DATA_TEMPLATE_H

#include "offline_data.h"
#include "scratch_data.h"

#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace grendel
{
  using namespace dealii;


  template <int dim>
  OfflineData<dim>::OfflineData(
      const grendel::Discretization<dim> &discretization,
      const std::string &subsection /*= "OfflineData"*/)
      : ParameterAcceptor(subsection)
      , discretization_(&discretization)
  {
  }


  template <int dim>
  void OfflineData<dim>::setup()
  {
    deallog << "OfflineData<dim>::setup()" << std::endl;

    dof_handler_.initialize(discretization_->triangulation(),
                            discretization_->finite_element());

    deallog << "        " << dof_handler_.n_dofs() << " DoFs" << std::endl;

    DoFRenumbering::Cuthill_McKee(dof_handler_);

    affine_constraints_.clear();
    affine_constraints_.close();

    DynamicSparsityPattern c_sparsity(dof_handler_.n_dofs(),
                                      dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(
        dof_handler_, c_sparsity, affine_constraints_, false);
    sparsity_pattern_.copy_from(c_sparsity);

    mass_matrix_.reinit(sparsity_pattern_);
    lumped_mass_matrix_.reinit(sparsity_pattern_);
    for (auto &matrix : cij_matrix_)
      matrix.reinit(sparsity_pattern_);
  }


  template <int dim>
  void OfflineData<dim>::assemble()
  {
    deallog << "OfflineData<dim>::assemble()" << std::endl;

    mass_matrix_ = 0.;
    lumped_mass_matrix_ = 0.;
    for (auto &matrix : cij_matrix_)
      matrix = 0.;

    const unsigned int dofs_per_cell =
        discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = discretization_->quadrature().size();

    /* The local, per-cell assembly routine: */

    auto local_assemble_system =
        [&](const auto &cell, auto &scratch, auto &copy) {
          auto &local_dof_indices = copy.local_dof_indices_;
          auto &cell_mass_matrix = copy.cell_mass_matrix_;
          auto &cell_lumped_mass_matrix = copy.cell_lumped_mass_matrix_;
          auto &cell_cij_matrix = copy.cell_cij_matrix_;

          auto &fe_values = scratch.fe_values_;

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

    /* The local, per-cell assembly routine: */

    auto copy_local_to_global = [&](const auto &copy) {
      const auto &local_dof_indices = copy.local_dof_indices_;
      const auto &cell_mass_matrix = copy.cell_mass_matrix_;
      const auto &cell_lumped_mass_matrix = copy.cell_lumped_mass_matrix_;
      const auto &cell_cij_matrix = copy.cell_cij_matrix_;

      affine_constraints_.distribute_local_to_global(
          cell_mass_matrix, local_dof_indices, mass_matrix_);

      affine_constraints_.distribute_local_to_global(
          cell_lumped_mass_matrix, local_dof_indices, lumped_mass_matrix_);

      for (int k = 0; k < dim; ++k)
        affine_constraints_.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, cij_matrix_[k]);
    };

    /*
     * And run a workstream to assemble the matrix.
     *
     * We need a graph coloring for the cells exactly once (the TimeStep
     * iterates over degrees of freedom without conflicts). Thus, construct
     * a graph coloring locally instead of caching it.
     */

    const auto get_conflict_indices = [&](auto &cell) {
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      return local_dof_indices;
    };

    const auto graph = GraphColoring::make_graph_coloring(
        dof_handler_.begin_active(), dof_handler_.end(), get_conflict_indices);

    WorkStream::run(graph,
                    local_assemble_system,
                    copy_local_to_global,
                    AssemblyScratchData<dim>(*discretization_),
                    AssemblyCopyData<dim>());
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
