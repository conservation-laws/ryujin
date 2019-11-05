#ifndef POSTPROCESSOR_TEMPLATE_H
#define POSTPROCESSOR_TEMPLATE_H

#include "helper.h"
#include "postprocessor.h"

#include <boost/range/irange.hpp>

#include <atomic>

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
      dealii::TimerOutput &computing_timer,
      const grendel::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Postprocessor"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
  {
    schlieren_beta_ = 10.;
    add_parameter("schlieren beta",
                  schlieren_beta_,
                  "Beta factor used in the Schlieren postprocessor");
  }


  template <int dim, typename Number>
  void Postprocessor<dim, Number>::prepare()
  {
    deallog << "Postprocessor<dim, Number>::prepare()" << std::endl;
    TimerOutput::Scope t(computing_timer_,
                         "postprocessor - prepare scratch space");

    const auto &partitioner = offline_data_->partitioner();

    for (auto &it : U_)
      it.reinit(partitioner);

    for (auto &it : quantities_)
      it.reinit(partitioner);
  }


  template <int dim, typename Number>
  void Postprocessor<dim, Number>::compute(const vector_type &U,
                                           const scalar_type &alpha)
  {
    deallog << "Postprocessor<dim, Number>::compute()" << std::endl;

    TimerOutput::Scope t(computing_timer_, "postprocessor - compute");


    const auto &affine_constraints = offline_data_->affine_constraints();
    const auto &sparsity = offline_data_->sparsity_pattern();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &boundary_normal_map = offline_data_->boundary_normal_map();

    const auto &n_locally_owned = offline_data_->n_locally_owned();
    const auto indices = boost::irange<unsigned int>(0, n_locally_owned);

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
          curl_type vorticity;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto j = jt->column();

            if (i == j)
              continue;

            const auto U_j = gather(U, j);
            const auto m_j = ProblemDescription<dim, Number>::momentum(U_j);

            const auto c_ij = cij_matrix.get_tensor(i, jt - sparsity.begin(i));

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
          const Number m_i = lumped_mass_matrix.diag_element(i);

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
     * Step 3: Normalize schlieren:
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

          const auto r_i = quantities_[0].local_element(i);
          quantities_[0].local_element(i) =
              Number(1.) - std::exp(-schlieren_beta_ * (r_i - r_i_min) /
                                    (r_i_max - r_i_min));
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);
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

} /* namespace grendel */

#endif /* POSTPROCESSOR_TEMPLATE_H */
