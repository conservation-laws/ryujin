##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

#
# A minimal project using an installed ryujin library:
#
#   cmake -DRYUJIN_DIR=<installation prefix>
#     -DCMAKE_CXX_COMPILER=<compiler of ryujin> <path to this directory>
#

cmake_minimum_required(VERSION 3.15)

project(ryujin_example CXX)

find_package(ryujin 2.5 REQUIRED HINTS ${RYUJIN_DIR} $ENV{RYUJIN_DIR})

add_executable(example main.cc)
ryujin_setup_target(example)
