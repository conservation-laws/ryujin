//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef CHECKPOINTING_H
#define CHECKPOINTING_H

#include <deal.II/base/utilities.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/core/demangle.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace ryujin
{
  /**
   * Performs a resume operation. Given a @p base_name the function tries
   * to locate correponding checkpoint files and will read in the saved
   * state @p U at saved time @p t with saved output cycle @p output_cycle.
   *
   * @todo Some day, we should refactor this into a class and do something
   * smarter...
   *
   * @ingroup Miscellaneous
   */
  template<typename T1, typename T2, typename T3>
  void do_resume(const std::string &base_name,
                 unsigned int id,
                 T1 &U,
                 T2 &t,
                 T3 &output_cycle)
  {
    std::string name = base_name + "-checkpoint-" +
                       dealii::Utilities::int_to_string(id, 4) + ".archive";
    std::ifstream file(name, std::ios::binary);

    boost::archive::binary_iarchive ia(file);
    ia >> t >> output_cycle;

    for (auto &it : U) {
      ia >> it;
    }
    U.update_ghost_values();
  }


  /**
   * Writes out a checkpoint to disk. Given a @p base_name and a current
   * state @p U at time @p t and output cycle @p output_cycle the function
   * writes out the state to disk using boost::archive for serialization.
   *
   * @todo Some day, we should refactor this into a class and do something
   * smarter...
   *
   * @ingroup Miscellaneous
   */
  template<typename T1, typename T2, typename T3>
  void do_checkpoint(const std::string &base_name,
                     unsigned int id,
                     const T1 &U,
                     const T2 t,
                     const T3 output_cycle)
  {
    std::string name = base_name + "-checkpoint-" +
                       dealii::Utilities::int_to_string(id, 4) + ".archive";

    if (std::filesystem::exists(name))
      std::filesystem::rename(name, name + "~");

    std::ofstream file(name, std::ios::binary | std::ios::trunc);

    boost::archive::binary_oarchive oa(file);
    oa << t << output_cycle;
    for (const auto &it : U)
      oa << it;
  }
} // namespace ryujin

#endif /* CHECKPOINTING_H */
