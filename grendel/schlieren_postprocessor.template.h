#ifndef SCHLIEREN_POSTPROCESSOR_TEMPLATE_H
#define SCHLIEREN_POSTPROCESSOR_TEMPLATE_H

#include "helper.h"
#include "schlieren_postprocessor.h"

namespace grendel
{
  using namespace dealii;


  template <int dim>
  SchlierenPostprocessor<dim>::SchlierenPostprocessor(
      const MPI_Comm &mpi_communicator,
      dealii::TimerOutput &computing_timer,
      const grendel::OfflineData<dim> &offline_data,
      const std::string &subsection /*= "SchlierenPostprocessor"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
  {
    schlieren_beta_ = 10.;
    add_parameter("schlieren beta",
                  schlieren_beta_,
                  "Beta factor used in Schlieren-type postprocessor");

    schlieren_index_ = 0;
    add_parameter("schlieren index",
                  schlieren_index_,
                  "Use the corresponding component of the state vector for the "
                  "schlieren plot");
  }


  template <int dim>
  void SchlierenPostprocessor<dim>::prepare()
  {
    deallog << "SchlierenPostprocessor<dim>::prepare()" << std::endl;
    TimerOutput::Scope t(computing_timer_,
                         "schlieren_postprocessor - prepare scratch space");

    const auto &locally_owned = offline_data_->locally_owned();
    const auto &locally_relevant = offline_data_->locally_relevant();
    const auto &sparsity_pattern = offline_data_->sparsity_pattern();

    schlieren_.reinit(locally_owned, locally_relevant, mpi_communicator_);
  }


  template <int dim>
  void SchlierenPostprocessor<dim>::compute_schlieren(const vector_type &U)
  {
    deallog << "SchlierenPostprocessor<dim>::compute_schlieren()" << std::endl;

    TimerOutput::Scope t(computing_timer_,
                         "schlieren_postprocessor - compute schlieren plot");

    const auto &locally_relevant = offline_data_->locally_relevant();
    const auto &locally_owned = offline_data_->locally_owned();
    const auto &sparsity = offline_data_->sparsity_pattern();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &norm_matrix = offline_data_->norm_matrix();
    const auto &nij_matrix = offline_data_->nij_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_normal_map = offline_data_->boundary_normal_map();
  }

} /* namespace grendel */

#endif /* SCHLIEREN_POSTPROCESSOR_TEMPLATE_H */
