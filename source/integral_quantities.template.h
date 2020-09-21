//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef INTEGRAL_QUANTITIES_TEMPLATE_H
#define INTEGRAL_QUANTITIES_TEMPLATE_H

#include "integral_quantities.h"
#include "simd.h"

#include <fstream>

namespace ryujin
{
  using namespace dealii;

  template <int dim, typename Number>
  IntegralQuantities<dim, Number>::IntegralQuantities(
      const MPI_Comm &mpi_communicator,
      const ProblemDescription &problem_description,
      const OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "IntegralQuantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , mpi_rank(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , problem_description(problem_description)
      , offline_data_(&offline_data)
  {
    record_conserved_quantities_ = true;
    add_parameter(
        "record conserved quantities",
        record_conserved_quantities_,
        "Compute integrals of conserved quantities and write results to the "
        "\"[basename]-integral_quantities.log\" at specified intervals");
  }


  template <int dim, typename Number>
  void IntegralQuantities<dim, Number>::prepare(const std::string &name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "IntegralQuantities<dim, Number>::prepare()" << std::endl;
#endif
    if (mpi_rank != 0)
      return;

    output.open(name);

    output << "time";
    if (record_conserved_quantities_)
      output << "\ttotal mass\ttotal momentum\ttotal energy\ts_min\te_min";
    output << std::endl;
  }


  template <int dim, typename Number>
  void IntegralQuantities<dim, Number>::compute(const vector_type &U, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::compute()" << std::endl;
#endif
    if (mpi_rank == 0)
      output << std::scientific << std::setprecision(14) << t;

    if (record_conserved_quantities_) {
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
      const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

      rank1_type summed_quantities;
      Number s_min = std::numeric_limits<Number>::max();
      Number e_min = std::numeric_limits<Number>::max();

      RYUJIN_PARALLEL_REGION_BEGIN

      rank1_type summed_quantities_thread_local;
      Number s_min_thread_local = std::numeric_limits<Number>::max();
      Number e_min_thread_local = std::numeric_limits<Number>::max();

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_owned; ++i) {

        /* Skip constrained degrees of freedom (periodic constraints) */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        const auto m_i = lumped_mass_matrix.local_element(i);
        const auto U_i = U.get_tensor(i);
        summed_quantities_thread_local += m_i * U_i;

        const auto s_i = problem_description.specific_entropy(U_i);
        const auto rho_i = problem_description.density(U_i);
        const auto e_i = problem_description.internal_energy(U_i) / rho_i;
        s_min_thread_local = std::min(s_min_thread_local, s_i);
        e_min_thread_local = std::min(e_min_thread_local, e_i);
      }

      RYUJIN_OMP_CRITICAL
      {
        summed_quantities += summed_quantities_thread_local;
        s_min = std::min(s_min, s_min_thread_local);
        e_min = std::min(e_min, e_min_thread_local);
      }

      RYUJIN_PARALLEL_REGION_END

      for (unsigned int k = 0; k < problem_dimension; ++k) {
        summed_quantities[k] =
            Utilities::MPI::sum(summed_quantities[k], mpi_communicator_);
        if (mpi_rank == 0)
          output << "\t" << summed_quantities[k];
      }

      s_min = Utilities::MPI::min(s_min, mpi_communicator_);
      e_min = Utilities::MPI::min(e_min, mpi_communicator_);
      if (mpi_rank == 0)
        output << "\t" << s_min << "\t" << e_min;
    }

    if (mpi_rank == 0)
      output << std::endl;
  }

} /* namespace ryujin */

#endif /* INTEGRAL_QUANTITIES_TEMPLATE_H */
