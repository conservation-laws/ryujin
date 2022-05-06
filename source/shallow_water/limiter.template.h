//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "limiter.h"

namespace ryujin
{
  template <int dim, typename Number>
  std::tuple<Number, bool>
  Limiter<dim, Number>::limit(const HyperbolicSystem &hyperbolic_system,
                              const Bounds &bounds,
                              const state_type &U,
                              const state_type &P,
                              const ScalarNumber newton_tolerance,
                              const unsigned int newton_max_iter,
                              const Number t_min /* = Number(0.) */,
                              const Number t_max /* = Number(1.) */)
  {
    bool success = true;
    Number t_r = t_max;

    return {t_r, success};
  }

} // namespace ryujin
