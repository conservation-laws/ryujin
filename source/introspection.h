//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "openmp.h"

/**
 * @name Various macros and include for instrumentation via valgrind,
 * likwid, and clang lsan.
 */
//@{

/**
 * A set of macros that start and stop callgrind instrumentation (if the
 * executable is run with valgrind). We currently wrap the hot paths in the
 * Euler and Navier-Stokes modules in the HyperbolicModule::step() and
 * DissipationModule::step() functions. Usage:
 *
 * @code
 * CALLGRIND_START_INSTRUMENTATION
 * // critical compute kernel section
 * CALLGRIND_STOP_INSTRUMENTATION
 * @endcode
 */
#define CALLGRIND_START_INSTRUMENTATION

/**
 * @copydoc CALLGRIND_START_INSTRUMENTATION
 */
#define CALLGRIND_STOP_INSTRUMENTATION

#ifdef WITH_VALGRIND
#undef CALLGRIND_START_INSTRUMENTATION
#undef CALLGRIND_STOP_INSTRUMENTATION
#include <valgrind/callgrind.h>
#endif


/**
 * Wrapper macro initializing likwid introspection. Used in main().
 */
#define LIKWID_INIT

/**
 * Wrapper macro finalizing likwid introspection. Used in main().
 */
#define LIKWID_CLOSE

/**
 * A set of macros that start and stop likwid instrumentation (if support
 * for likwid is enabled). We currently wrap the hot paths in the
 * Euler and Navier-Stokes modules in the HyperbolicModule::step() and
 * DissipationModule::step() functions. Usage:
 *
 * @code
 * LIKWID_MARKER_START("string identifier")
 * // critical compute kernel section
 * LIKWID_MARKER_STOP("string identifier")
 * @endcode
 */
#define LIKWID_MARKER_START(opt)

/**
 * @copydoc LIKWID_MARKER_START
 */
#define LIKWID_MARKER_STOP(opt)

#ifdef WITH_LIKWID
#undef LIKWID_INIT
#undef LIKWID_CLOSE
#undef LIKWID_MARKER_START
#undef LIKWID_MARKER_STOP
#include <likwid.h>

#define LIKWID_INIT                                                            \
  LIKWID_MARKER_INIT;                                                          \
  RYUJIN_PARALLEL_REGION_BEGIN                                                 \
  LIKWID_MARKER_THREADINIT;                                                    \
  RYUJIN_PARALLEL_REGION_END

#define LIKWID_CLOSE LIKWID_MARKER_CLOSE;
#endif


/**
 * Explicitly disable/enable the LLVM/Clang LeakSanitiver
 *
 * @code
 * LSAN_DISABLE
 * // Calling some external code path that is leaky and that we cannot
 * // control...
 * LSAN_ENABLE
 * @endcode
 */
#define LSAN_DISABLE

/**
 * @copydoc LSAN_DISABLE
 */
#define LSAN_ENABLE

#if defined(__clang__) && defined(DEBUG)
#if __has_feature(address_sanitizer)
#include <sanitizer/lsan_interface.h>
#undef LSAN_DISABLE
#define LSAN_DISABLE __lsan_disable();
#undef LSAN_ENABLE
#define LSAN_ENABLE __lsan_enable();
#endif
#endif

//@}
