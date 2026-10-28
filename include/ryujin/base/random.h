//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <random>

namespace ryujin
{
  /**
   * Return a std::mt19937_64 Mersenne Twister whose entire internal state
   * is seeded from std::random_device.
   */
  inline std::mt19937_64 seeded_mersenne_twister()
  {
    /* The state consists of state_size 64 bit words: */
    constexpr auto n_words = 2 * std::mt19937_64::state_size;
    std::array<std::seed_seq::result_type, n_words> seed_data;

    std::random_device random_device;
    std::generate(seed_data.begin(), seed_data.end(), std::ref(random_device));
    std::seed_seq seed_sequence(seed_data.begin(), seed_data.end());

    return std::mt19937_64(seed_sequence);
  }
} // namespace ryujin
