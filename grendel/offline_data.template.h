#ifndef OFFLINE_DATA_TEMPLATE_H
#define OFFLINE_DATA_TEMPLATE_H

#include "offline_data.h"

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
    deallog << "OfflineData<dim>::setup_system()" << std::endl;
  }


  template <int dim>
  void OfflineData<dim>::assemble()
  {
    deallog << "OfflineData<dim>::assemble()" << std::endl;
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

    data_out_.clear();
  }

} /* namespace grendel */

#endif /* OFFLINE_DATA_TEMPLATE_H */
