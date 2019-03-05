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

    r_i_.resize(locally_relevant.n_elements());
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
    const auto &cij_matrix = offline_data_->cij_matrix();

    /*
     * Step 1: Compute r_i
     */

    {
      const auto on_subranges = [&](const auto it1, const auto it2) {
        /* [it1, it2) is an iterator range over r_i_ */

        /* Create an iterator for the index set: */
        const unsigned int pos = std::distance(r_i_.begin(), it1);
        auto set_iterator =
            locally_relevant.at(locally_relevant.nth_index_in_set(pos));

        for (auto it = it1; it != it2; ++it, ++set_iterator) {
          const auto i = *set_iterator;

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          double r_i = 0.;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            const auto U_js = U[schlieren_index_][j];
            const auto c_ij = gather_get_entry(cij_matrix, jt);

            r_i += c_ij.norm_square() * U_js * U_js;
          }

          const double m_i = lumped_mass_matrix.diag_element(i);
          r_i = sqrt(r_i) / m_i;

          *it = r_i;
        }
      };
      parallel::apply_to_subranges(
          r_i_.begin(), r_i_.end(), on_subranges, 4096);
    }

    /*
     * Step 2: Compute schlieren:
     */

    {
      const auto on_subranges = [&](const auto it1, const auto it2) {
        /* [it1, it2) is an iterator range over r_i_ */

        /* Create an iterator for the index set: */
        const unsigned int pos = std::distance(r_i_.begin(), it1);
        auto set_iterator =
            locally_relevant.at(locally_relevant.nth_index_in_set(pos));

        for (auto it = it1; it != it2; ++it, ++set_iterator) {
          const auto i = *set_iterator;

          const auto r_i = *it;

          /* Only iterate over locally owned subset */
          if (!locally_owned.is_element(i))
            continue;

          double max_r_i = 0.;
          double min_r_i = std::numeric_limits<double>::max();

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            const unsigned int pos_j = locally_relevant.index_within_set(j);
            const auto r_j = r_i_[pos_j];

            max_r_i = std::max(max_r_i, r_j);
            min_r_i = std::min(min_r_i, r_j);
          }

          schlieren_[i] =
              std::exp(-schlieren_beta_ * (r_i - min_r_i) / (max_r_i - min_r_i));
        }
      };

      parallel::apply_to_subranges(
          r_i_.begin(), r_i_.end(), on_subranges, 4096);
    }
  }

} /* namespace grendel */

#endif /* SCHLIEREN_POSTPROCESSOR_TEMPLATE_H */
