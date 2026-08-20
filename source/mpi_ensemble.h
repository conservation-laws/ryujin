//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

namespace ryujin
{
  /**
   * A class responsible for subdividing a given global MPI communicator
   * into a set of "ensembles" with a corresponding ensemble communicator.
   * This allows us to run similar, related hyperbolic systems in parallel
   * on subranges of the set of (global) MPI processes.
   *
   * After @p prepare() is called, all getter functions return valid
   * references.
   *
   * @ingroup Miscellaneous
   */
  class MPIEnsemble final
  {
  public:
    /**
     * Prepare the MPI ensemble and split the global MPI communicator into
     * @p n_ensembles different subranges of comparable size. The boolean
     * @p global_synchronization indicates whether (global) world
     * synchronization of time step size and other synchronization state is
     * performed in the HyperbolicModule and ParabolicModule, or whether
     * such synchronization remains local to the ensemble.
     *
     * @pre The total number of mpi ranks must be an integer multiple of
     * n_ensembles.
     */
    MPIEnsemble(const MPI_Comm &mpi_communicator,
                const int n_ensembles = 1,
                const bool global_synchronization = true);

    ~MPIEnsemble();

    /**
     * Return the world communicator.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(world_communicator);

    /**
     * If true, then ensembles run in lockstep with a synchronized tau_max.
     */
    ACCESSOR_READ_ONLY(global_synchronization);

    /**
     * The (global) world rank of the current MPI process.
     */
    ACCESSOR_READ_ONLY(world_rank);

    /**
     * The total number of (global) MPI processes.
     */
    ACCESSOR_READ_ONLY(n_world_ranks);

    /**
     * Return the ensemble in the interval [0, n_ensembles) that the given
     * MPI process belongs to.
     */
    ACCESSOR_READ_ONLY(ensemble);

    /**
     * Return the total number of ensembles.
     */
    ACCESSOR_READ_ONLY(n_ensembles);

    /**
     * The (local) ensemble rank of the current MPI process.
     */
    ACCESSOR_READ_ONLY(ensemble_rank);

    /**
     * The total number of (local) MPI processes belonging to the ensemble.
     */
    ACCESSOR_READ_ONLY(n_ensemble_ranks);

    /**
     * The corresponding subrange communicator of the ensemble. The
     * communicator is collective over all ranks participating in the
     * ensemble. I.e., it allows for ensemble-local communication.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(ensemble_communicator);

    /**
     * Return a communicator for synchronization. The method either returns
     * the (global) world communicator if global synchronization is
     * enabled, or the (local) ensemble communicator.
     */
    DEAL_II_ALWAYS_INLINE inline const MPI_Comm &
    synchronization_communicator() const
    {
      if (global_synchronization_)
        return world_communicator_;
      else
        return ensemble_communicator_;
    }

    /**
     * A communicator spanning over all ensemble leaders that have ensemble
     * rank 0. This communicator is collective over all ensemble leaders
     * and invalid for all other ranks.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(ensemble_leader_communicator);

    /**
     * A "peer communicator" that groups all kth ranks of each ensemble
     * together. (Suppose the ensemble_communicator() groups rows, then the
     * peer_communicator() groups columns of the ensemble partition). The
     * peer communicator is collective over all world ranks.
     */
    ACCESSOR_READ_ONLY_NO_DEREFERENCE(peer_communicator);

  private:
    const MPI_Comm &world_communicator_;

    bool global_synchronization_;

    int world_rank_;
    int n_world_ranks_;
    int ensemble_;
    int n_ensembles_;
    int ensemble_rank_;
    int n_ensemble_ranks_;

    MPI_Group world_group_;
    std::vector<MPI_Group> ensemble_groups_;
    MPI_Group ensemble_leader_group_;

    MPI_Comm ensemble_communicator_;
    MPI_Comm ensemble_leader_communicator_;
    MPI_Comm peer_communicator_;
  };
} /* namespace ryujin */
