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
  /*
   * Checkpoint-restore helper functions:
   *
   * TODO: Some day, we should refactor this into a class and do something
   * smarter...
   */
  const auto do_resume = [](const std::string &base_name,
                            unsigned int id,
                            auto &U,
                            auto &t,
                            auto &output_cycle) {
    std::string name = base_name + "-checkpoint-" +
                       dealii::Utilities::int_to_string(id, 4) + ".archive";
    std::ifstream file(name, std::ios::binary);

    boost::archive::binary_iarchive ia(file);
    ia >> t >> output_cycle;

    for (auto &it : U) {
      ia >> it;
    }
    U.update_ghost_values();
  };

  const auto do_checkpoint = [](const std::string &base_name,
                                unsigned int id,
                                const auto &U,
                                const auto t,
                                const auto output_cycle) {
    std::string name = base_name + "-checkpoint-" +
                       dealii::Utilities::int_to_string(id, 4) + ".archive";

    if (std::filesystem::exists(name))
      std::filesystem::rename(name, name + "~");

    std::ofstream file(name, std::ios::binary | std::ios::trunc);

    boost::archive::binary_oarchive oa(file);
    oa << t << output_cycle;
    for (const auto &it : U)
      oa << it;
  };
} // namespace ryujin

#endif /* CHECKPOINTING_H */
