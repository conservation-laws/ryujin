//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#include "multicomponent_vector.h"

namespace ryujin
{
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
  create_vector_partitioner(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &scalar_partitioner,
      const unsigned int n_components)
  {
    dealii::IndexSet vector_owned_set(n_components *
                                      scalar_partitioner->size());
    for (auto it = scalar_partitioner->locally_owned_range().begin_intervals();
         it != scalar_partitioner->locally_owned_range().end_intervals();
         ++it)
      vector_owned_set.add_range(*it->begin() * n_components,
                                 (it->last() + 1) * n_components);
    vector_owned_set.compress();
    dealii::IndexSet vector_ghost_set(n_components *
                                      scalar_partitioner->size());
    for (auto it = scalar_partitioner->ghost_indices().begin_intervals();
         it != scalar_partitioner->ghost_indices().end_intervals();
         ++it)
      vector_ghost_set.add_range(*it->begin() * n_components,
                                 (it->last() + 1) * n_components);
    vector_ghost_set.compress();
    const auto vector_partitioner =
        std::make_shared<const dealii::Utilities::MPI::Partitioner>(
            vector_owned_set,
            vector_ghost_set,
            scalar_partitioner->get_mpi_communicator());

    return vector_partitioner;
  }
} // namespace ryujin
