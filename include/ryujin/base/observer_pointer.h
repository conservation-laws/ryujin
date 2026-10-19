//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <deal.II/base/config.h>

#if DEAL_II_VERSION_GTE(9, 7, 0)
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/observer_pointer.h>

#else

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>

DEAL_II_NAMESPACE_OPEN

template <typename T, typename P = void>
using ObserverPointer = SmartPointer<T, P>;

using EnableObserverPointer = Subscriptor;
DEAL_II_NAMESPACE_CLOSE

#endif
