//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2024 - 2024 by Maximilian Bergbauer
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/config.h>

#include <deal.II/base/ndarray.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/utilities.h>
#include <deal.II/matrix_free/tensor_product_point_kernels.h>

#include <deal.II/matrix_free/shape_info.h>

namespace ryujin
{

  namespace internal
  {
    /**
     * This is a copy of the integrate_add_tensor_product_value_linear()
     * function shipped with deal.II that fixes a compilation error when
     * instantiating with float.
     *
     * @ingroup Mesh
     */
    template <int dim, typename Number, typename Number2, bool add>
    inline void integrate_add_tensor_product_value_linear(
        const Number2 &value,
        Number2 *values,
        const dealii::Point<dim, Number> &p)
    {
      static_assert(dim >= 0 && dim <= 3, "Only dim=0,1,2,3 implemented");

      if (dim == 0) {
        if (add)
          values[0] += value;
        else
          values[0] = value;
      } else if (dim == 1) {
        const auto x0 = Number(1.) - p[0], x1 = p[0];

        if (add) {
          values[0] += value * x0;
          values[1] += value * x1;
        } else {
          values[0] = value * x0;
          values[1] = value * x1;
        }
      } else if (dim == 2) {
        const auto x0 = Number(1.) - p[0], x1 = p[0], y0 = Number(1.) - p[1],
                   y1 = p[1];

        const auto test_value_y0 = value * y0;
        const auto test_value_y1 = value * y1;

        if (add) {
          values[0] += x0 * test_value_y0;
          values[1] += x1 * test_value_y0;
          values[2] += x0 * test_value_y1;
          values[3] += x1 * test_value_y1;
        } else {
          values[0] = x0 * test_value_y0;
          values[1] = x1 * test_value_y0;
          values[2] = x0 * test_value_y1;
          values[3] = x1 * test_value_y1;
        }
      } else if (dim == 3) {
        const auto x0 = Number(1.) - p[0], x1 = p[0], y0 = Number(1.) - p[1],
                   y1 = p[1], z0 = Number(1.) - p[2], z1 = p[2];

        const auto test_value_z0 = value * z0;
        const auto test_value_z1 = value * z1;

        const auto test_value_y00 = test_value_z0 * y0;
        const auto test_value_y01 = test_value_z0 * y1;
        const auto test_value_y10 = test_value_z1 * y0;
        const auto test_value_y11 = test_value_z1 * y1;

        if (add) {
          values[0] += x0 * test_value_y00;
          values[1] += x1 * test_value_y00;
          values[2] += x0 * test_value_y01;
          values[3] += x1 * test_value_y01;
          values[4] += x0 * test_value_y10;
          values[5] += x1 * test_value_y10;
          values[6] += x0 * test_value_y11;
          values[7] += x1 * test_value_y11;
        } else {
          values[0] = x0 * test_value_y00;
          values[1] = x1 * test_value_y00;
          values[2] = x0 * test_value_y01;
          values[3] = x1 * test_value_y01;
          values[4] = x0 * test_value_y10;
          values[5] = x1 * test_value_y10;
          values[6] = x0 * test_value_y11;
          values[7] = x1 * test_value_y11;
        }
      }
    }

    template <bool is_linear, int dim, typename Number, typename Number2>
    inline void integrate_tensor_product_value(
        const dealii::ndarray<Number, 2, dim> *shapes,
        const unsigned int n_shapes,
        const Number2 &value,
        Number2 *values,
        const dealii::Point<dim, Number> &p,
        const bool do_add)
    {
      if (do_add) {
        if (is_linear)
          integrate_add_tensor_product_value_linear<dim, Number, Number2, true>(
              value, values, p);
        else
          dealii::internal::integrate_add_tensor_product_value_shapes<dim,
                                                                      Number,
                                                                      Number2,
                                                                      true>(
              shapes, n_shapes, value, values);
      } else {
        if (is_linear)
          integrate_add_tensor_product_value_linear<dim,
                                                    Number,
                                                    Number2,
                                                    false>(value, values, p);
        else
          dealii::internal::integrate_add_tensor_product_value_shapes<dim,
                                                                      Number,
                                                                      Number2,
                                                                      false>(
              shapes, n_shapes, value, values);
      }
    }
  } // end of namespace internal

} // namespace ryujin
