//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef INTROSPECTION_H
#define INTROSPECTION_H

/*
 * Valgrind
 */

#ifdef VALGRIND_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_START_INSTRUMENTATION
#define CALLGRIND_STOP_INSTRUMENTATION
#endif

/*
 * Likwid
 */

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_START(opt)
#define LIKWID_MARKER_STOP(opt)
#endif

/*
 * Clang address sanitizer
 */

#define LSAN_DISABLE
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

#endif /* INTROSPECTION_H */
