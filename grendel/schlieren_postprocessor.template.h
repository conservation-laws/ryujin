#ifndef SCHLIEREN_POSTPROCESSOR_TEMPLATE_H
#define SCHLIEREN_POSTPROCESSOR_TEMPLATE_H

#include "helper.h"
#include "schlieren_postprocessor.h"

#include <boost/range/irange.hpp>

#include <atomic>

namespace grendel
{
  using namespace dealii;


  template <int dim, typename Number>
  SchlierenPostprocessor<dim, Number>::SchlierenPostprocessor(
      const MPI_Comm &mpi_communicator,
      dealii::TimerOutput &computing_timer,
      const grendel::OfflineData<dim, Number> &offline_data,
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


  template <int dim, typename Number>
  void SchlierenPostprocessor<dim, Number>::prepare()
  {
    deallog << "SchlierenPostprocessor<dim, Number>::prepare()" << std::endl;
    TimerOutput::Scope t(computing_timer_,
                         "schlieren_postprocessor - prepare scratch space");

    const auto &n_locally_relevant = offline_data_->n_locally_relevant();
    const auto &partitioner = offline_data_->partitioner();

    r_i_.reinit(n_locally_relevant);
    schlieren_.reinit(partitioner);
  }


  template <int dim, typename Number>
  void
  SchlierenPostprocessor<dim, Number>::compute_schlieren(const vector_type &U)
  {
    deallog << "SchlierenPostprocessor<dim, Number>::compute_schlieren()"
            << std::endl;

    TimerOutput::Scope t(computing_timer_,
                         "schlieren_postprocessor - compute schlieren plot");


    const auto &affine_constraints = offline_data_->affine_constraints();
    const auto &sparsity = offline_data_->sparsity_pattern();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_normal_map = offline_data_->boundary_normal_map();

    const auto &n_locally_owned = offline_data_->n_locally_owned();
    const auto indices = boost::irange<unsigned int>(0, n_locally_owned);

    /*
     * Step 1: Compute r_i and r_i_max, r_i_min:
     */

    std::atomic<Number> r_i_max{0.};
    std::atomic<Number> r_i_min{std::numeric_limits<Number>::infinity()};

    {
      const auto on_subranges = [&](auto i1, const auto i2) {
        Number r_i_max_on_subrange = 0.;
        Number r_i_min_on_subrange = std::numeric_limits<Number>::infinity();

        for (; i1 < i2; ++i1) {
          const auto i = *i1;

          /* Only iterate over locally owned subset */
          Assert(i < n_locally_owned, ExcInternalError());

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          Tensor<1, dim, Number> r_i;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            if (i == j)
              continue;

            const auto U_js = U[schlieren_index_].local_element(j);
            const auto c_ij = gather_get_entry(cij_matrix, jt);

            r_i += c_ij * U_js;
          }

          /* Fix up boundaries: */
          const auto bnm_it = boundary_normal_map.find(i);
          if (bnm_it != boundary_normal_map.end()) {
            const auto [normal, id, _] = bnm_it->second;
            if (id == Boundary::slip) {
              r_i -= 1. * (r_i * normal) * normal;
            } else {
              /*
               * FIXME: This is not particularly elegant. On all other
               * boundary types, we simply set r_i to zero.
               */
              r_i = 0.;
            }
          }

          const Number m_i = lumped_mass_matrix.diag_element(i);
          r_i_[i] = r_i.norm() / m_i;

          r_i_max_on_subrange = std::max(r_i_max_on_subrange, r_i_[i]);
          r_i_min_on_subrange = std::min(r_i_min_on_subrange, r_i_[i]);
        }

        /* Synchronize over all threads: */

        Number current_r_i_max = r_i_max.load();
        while (current_r_i_max < r_i_max_on_subrange &&
               !r_i_max.compare_exchange_weak(current_r_i_max,
                                              r_i_max_on_subrange))
          ;

        Number current_r_i_min = r_i_min.load();
        while (current_r_i_min > r_i_min_on_subrange &&
               !r_i_min.compare_exchange_weak(current_r_i_min,
                                              r_i_min_on_subrange))
          ;
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);
    }

    /* And synchronize over all processors: */

    r_i_max.store(Utilities::MPI::max(r_i_max.load(), mpi_communicator_));
    r_i_min.store(Utilities::MPI::min(r_i_min.load(), mpi_communicator_));

    /*
     * Step 2: Compute schlieren:
     */

    {
      const auto on_subranges = [&](auto i1, const auto i2) {
        for (; i1 < i2; ++i1) {
          const auto i = *i1;

          /* Only iterate over locally owned subset */
          Assert(i < n_locally_owned, ExcInternalError());

          /* Skip constrained degrees of freedom */
          if (++sparsity.begin(i) == sparsity.end(i))
            continue;

          const auto r_i = r_i_[i]; /* SIC */
          schlieren_.local_element(i) =
              Number(1.) - std::exp(-schlieren_beta_ * (r_i - r_i_min) /
                                    (r_i_max - r_i_min));
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);

      /* Fix up hanging nodes: */
      affine_constraints.distribute(schlieren_);
    }

    schlieren_.update_ghost_values();
  }

} /* namespace grendel */

#endif /* SCHLIEREN_POSTPROCESSOR_TEMPLATE_H */
