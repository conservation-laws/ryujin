##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 by the ryujin authors
##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  )

add_executable(ryujin
  discretization.cc
  dissipation_module.cc
  euler_module.cc
  initial_values.cc
  limiter.cc
  main.cc
  offline_data.cc
  postprocessor.cc
  problem_description.cc
  quantities.cc
  riemann_solver.cc
  simd.cc
  sparse_matrix_simd.cc
  time_loop.cc
  )

deal_ii_setup_target(ryujin)

if(false)
if(NOT CMAKE_VERSION VERSION_LESS 3.16)
  target_precompile_headers(ryujin
    PRIVATE
    discretization.h
    geometry.h
    initial_values.h
    multicomponent_vector.h
    offline_data.h
    postprocessor.h
    problem_description.h
    riemann_solver.h
    scope.h
    scratch_data.h
    sparse_matrix_simd.h
    <array>
    <atomic>
    <chrono>
    <filesystem>
    <fstream>
    <functional>
    <future>
    <iomanip>
    <map>
    <memory>
    <omp.h>
    <random>
    <set>
    <sstream>
    <string>
    <boost/archive/binary_iarchive.hpp>
    <boost/archive/binary_oarchive.hpp>
    <boost/core/demangle.hpp>
    <boost/range/irange.hpp>
    <boost/range/iterator_range.hpp>
    <deal.II/base/aligned_vector.h>
    <deal.II/base/config.h>
    <deal.II/base/function.h>
    <deal.II/base/graph_coloring.h>
    <deal.II/base/logstream.h>
    <deal.II/base/multithread_info.h>
    <deal.II/base/parallel.h>
    <deal.II/base/parameter_acceptor.h>
    <deal.II/base/partitioner.h>
    <deal.II/base/point.h>
    <deal.II/base/quadrature.h>
    <deal.II/base/quadrature_lib.h>
    <deal.II/base/revision.h>
    <deal.II/base/tensor.h>
    <deal.II/base/timer.h>
    <deal.II/base/utilities.h>
    <deal.II/base/vectorization.h>
    <deal.II/base/work_stream.h>
    <deal.II/distributed/tria.h>
    <deal.II/dofs/dof_handler.h>
    <deal.II/dofs/dof_renumbering.h>
    <deal.II/dofs/dof_tools.h>
    <deal.II/fe/fe.h>
    <deal.II/fe/fe_q.h>
    <deal.II/fe/fe_system.h>
    <deal.II/fe/fe_values.h>
    <deal.II/fe/mapping.h>
    <deal.II/fe/mapping_q.h>
    <deal.II/grid/grid_generator.h>
    <deal.II/grid/grid_in.h>
    <deal.II/grid/grid_out.h>
    <deal.II/grid/grid_tools.h>
    <deal.II/grid/intergrid_map.h>
    <deal.II/grid/manifold_lib.h>
    <deal.II/grid/tria.h>
    <deal.II/lac/affine_constraints.h>
    <deal.II/lac/dynamic_sparsity_pattern.h>
    <deal.II/lac/full_matrix.h>
    <deal.II/lac/la_parallel_vector.h>
    <deal.II/lac/la_parallel_vector.templates.h>
    <deal.II/lac/linear_operator.h>
    <deal.II/lac/precondition.h>
    <deal.II/lac/solver_cg.h>
    <deal.II/lac/sparse_matrix.h>
    <deal.II/lac/sparse_matrix.templates.h>
    <deal.II/lac/vector.h>
    <deal.II/matrix_free/fe_evaluation.h>
    <deal.II/matrix_free/matrix_free.h>
    <deal.II/multigrid/mg_transfer_matrix_free.h>
    <deal.II/numerics/data_out.h>
    <deal.II/numerics/vector_tools.h>
    <deal.II/numerics/vector_tools.templates.h>
    )
endif()
endif()

set_property(SOURCE time_loop.cc APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )

set_property(TARGET ryujin
  PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/run
  )

if(LIKWID_PERFMON)
  target_link_libraries(ryujin likwid likwid-hwloc likwid-lua)
endif()

# FIXME: This is wrong if we link against libc++.
target_link_libraries(ryujin stdc++fs)
