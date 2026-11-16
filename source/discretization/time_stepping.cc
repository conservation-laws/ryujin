//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include "time_stepping.template.h"

namespace ryujin
{
  /* instantiations */

#define INSTANTIATE(dim)                                                       \
  template std::size_t n_temporary_vectors(                                    \
      const ButcherTableau<NUMBER> &,                                          \
      const Interface::HyperbolicModule<dim, NUMBER> &);                       \
                                                                               \
  template NUMBER step_runge_kutta(                                            \
      const ButcherTableau<NUMBER> &,                                          \
      const Interface::HyperbolicModule<dim, NUMBER> &,                        \
      const Interface::ParabolicModule<dim, NUMBER> &,                         \
      std::span<StateVector<NUMBER>>,                                          \
      const StateVector<NUMBER> &,                                             \
      StateVector<NUMBER> &,                                                   \
      NUMBER,                                                                  \
      NUMBER,                                                                  \
      NUMBER);                                                                 \
                                                                               \
  template NUMBER step_ssprk_22(                                               \
      const Interface::HyperbolicModule<dim, NUMBER> &,                        \
      StateVector<NUMBER> &,                                                   \
      const StateVector<NUMBER> &,                                             \
      StateVector<NUMBER> &,                                                   \
      NUMBER,                                                                  \
      NUMBER,                                                                  \
      NUMBER);                                                                 \
                                                                               \
  template NUMBER step_ssprk_33(                                               \
      const Interface::HyperbolicModule<dim, NUMBER> &,                        \
      StateVector<NUMBER> &,                                                   \
      const StateVector<NUMBER> &,                                             \
      StateVector<NUMBER> &,                                                   \
      NUMBER,                                                                  \
      NUMBER,                                                                  \
      NUMBER);                                                                 \
                                                                               \
  template NUMBER step_strang(                                                 \
      const Interface::HyperbolicModule<dim, NUMBER> &,                        \
      const StepFunction<NUMBER> &,                                            \
      const StepFunction<NUMBER> &,                                            \
      StateVector<NUMBER> &,                                                   \
      const StateVector<NUMBER> &,                                             \
      StateVector<NUMBER> &,                                                   \
      NUMBER,                                                                  \
      NUMBER,                                                                  \
      NUMBER);

  INSTANTIATE(1)
  INSTANTIATE(2)
  INSTANTIATE(3)

#undef INSTANTIATE
} // namespace ryujin
