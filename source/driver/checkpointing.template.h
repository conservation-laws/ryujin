//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/computing_timer.h>
#include <ryujin/driver/checkpointing.h>

#include <deal.II/base/mpi.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace ryujin
{
  template <int dim, typename Number>
  Checkpointing<dim, Number>::Checkpointing(
      Interface::Simulation<dim, Number> &simulation,
      const std::string &subsection /*= "/Checkpointing"*/)
      : ParameterAcceptor(subsection)
      , Interface::Task<Number>("checkpointing", TaskPhase::post_step)
      , simulation_(simulation)
  {
    this->trigger_ = std::make_unique<Trigger<Number>>(
        TriggerMode::wall_clock, /*multiplier*/ 1, /*interval*/ 3600);
    this->trigger_->add_parameters(*this);

    resume_ = false;
    add_parameter("resume", resume_, "Resume an interrupted computation");

    resume_at_time_zero_ = false;
    add_parameter("resume at time zero",
                  resume_at_time_zero_,
                  "Resume from the latest checkpoint but set the time to t=0.");
  }


  template <int dim, typename Number>
  void Checkpointing<dim, Number>::start(TaskContext<Number> &context)
  {
    if (!resume_)
      return;

#ifdef DEBUG_OUTPUT
    std::cout << "Checkpointing<dim, Number>::start()" << std::endl;
#endif

    context.info("resume: reading mesh and loading state vector");

    const auto &mpi_ensemble = simulation_.mpi_ensemble();
    const auto base_name = mpi_ensemble.base_name(context.base_name);
    const std::string name = base_name + "-checkpoint";

    /*
     * Read in and broadcast metadata:
     */

    Number t = 0.;
    unsigned int timer_cycle = 0;
    unsigned int transfer_handle = 0;
    std::map<std::string, std::string> task_data;

    if (mpi_ensemble.ensemble_rank() == 0) {
      std::ifstream file(name + ".metadata", std::ios::binary);
      boost::archive::binary_iarchive ia(file);
      ia >> t >> timer_cycle >> transfer_handle >> task_data;
    }

    const auto &communicator = mpi_ensemble.ensemble_communicator();
    t = dealii::Utilities::MPI::broadcast(communicator, t);
    timer_cycle = dealii::Utilities::MPI::broadcast(communicator, timer_cycle);
    transfer_handle =
        dealii::Utilities::MPI::broadcast(communicator, transfer_handle);
    task_data = dealii::Utilities::MPI::broadcast(communicator, task_data);

    /*
     * Read in the mesh and attach the state vector:
     */

    auto &discretization = simulation_.discretization();
    discretization.refinement() = 0; /* do not refine */
    discretization.prepare(base_name);
    discretization.triangulation().load(name + ".mesh");

    simulation_.solution_transfer().set_handle(transfer_handle);

    /*
     * Restore the internal state of all active tasks:
     */

    for (auto *task : context.tasks) {
      const auto it = task_data.find(task->name());
      if (it == task_data.end())
        continue;
      std::istringstream stream(it->second);
      boost::archive::binary_iarchive ia(stream);
      task->load(ia);
    }

    if (resume_at_time_zero_) {
      /* Reset the current time t and the output cycle count to zero: */
      t = 0.;
      timer_cycle = 0;
    }

    context.t = t;
    context.timer_cycle = timer_cycle;
  }


  template <int dim, typename Number>
  void Checkpointing<dim, Number>::execute(TaskContext<Number> &context)
  {
    if (!this->trigger_->fire(context))
      return;

    write_checkpoint(context);
  }


  template <int dim, typename Number>
  void Checkpointing<dim, Number>::finalize(TaskContext<Number> &context)
  {
    write_checkpoint(context);
  }


  template <int dim, typename Number>
  void Checkpointing<dim, Number>::write_checkpoint(
      const TaskContext<Number> &context)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Checkpointing<dim, Number>::write_checkpoint()" << std::endl;
#endif

    ComputingTimer::Scope scope("time step [X]   - perform checkpointing");
    context.info("scheduling checkpointing");

    const auto &mpi_ensemble = simulation_.mpi_ensemble();
    const std::string name =
        mpi_ensemble.base_name(context.base_name) + "-checkpoint";

    /* Attach the state vector to the triangulation: */

    auto &solution_transfer = simulation_.solution_transfer();
    solution_transfer.prepare_projection(simulation_.state_vector());
    const auto transfer_handle = solution_transfer.get_handle();
    solution_transfer.reset_handle();

    if (mpi_ensemble.ensemble_rank() == 0) {
      for (const std::string suffix :
           {".mesh", ".mesh_fixed.data", ".mesh.info", ".metadata"})
        if (std::filesystem::exists(name + suffix))
          std::filesystem::rename(name + suffix, name + suffix + "~");
    }

    simulation_.discretization().triangulation().save(name + ".mesh");

    /* Serialize the internal state of all active tasks (on all ranks): */

    std::map<std::string, std::string> task_data;
    for (const auto *task : context.tasks) {
      std::ostringstream stream;
      {
        boost::archive::binary_oarchive oa(stream);
        task->save(oa);
      }
      task_data[task->name()] = stream.str();
    }

    /* Write out metadata on rank 0: */

    if (mpi_ensemble.ensemble_rank() == 0) {
      std::ofstream file(name + ".metadata",
                         std::ios::binary | std::ios::trunc);
      boost::archive::binary_oarchive oa(file);
      oa << context.t << context.timer_cycle << transfer_handle << task_data;
    }

    const int ierr = MPI_Barrier(mpi_ensemble.ensemble_communicator());
    AssertThrowMPI(ierr);
  }
} // namespace ryujin
