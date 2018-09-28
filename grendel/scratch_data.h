#ifndef SCRATCH_DATA_H
#define SCRATCH_DATA_H

#include "discretization.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

namespace grendel
{

  /*
   * internal scratch data for parallelized assembly, time-stepping and
   * updates
   */

  template <int dim>
  class AssemblyScratchData
  {
  public:
    AssemblyScratchData(const AssemblyScratchData<dim> &assembly_scratch_data)
        : AssemblyScratchData(assembly_scratch_data.discretization_)
    {
    }


    AssemblyScratchData(const grendel::Discretization<dim> &discretization)
        : discretization_(discretization)
        , fe_values_(discretization_.mapping(),
                     discretization_.finite_element(),
                     discretization_.quadrature(),
                     dealii::update_values | dealii::update_gradients |
                         dealii::update_quadrature_points |
                         dealii::update_JxW_values)
    {
    }

    const grendel::Discretization<dim> &discretization_;
    dealii::FEValues<dim> fe_values_;
  };

  template <int dim>
  class AssemblyCopyData
  {
  public:
    bool is_artificial_;
    std::vector<dealii::types::global_dof_index> local_dof_indices_;
    std::vector<dealii::types::global_dof_index> local_boundary_dof_indices_;
    dealii::FullMatrix<double> cell_mass_matrix_;
    dealii::FullMatrix<double> cell_lumped_mass_matrix_;
    std::array<dealii::FullMatrix<double>, dim> cell_cij_matrix_;
  };

} // namespace grendel

#endif /* SCRATCH_DATA_H */
