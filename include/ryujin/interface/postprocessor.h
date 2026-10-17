//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <state_vector.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free interface of a Postprocessor: the extraction
     * of selected (and postprocessed) components out of a state vector.
     *
     * A component can be selected by any of its conserved, primitive,
     * precomputed, initial-precomputed, or parabolic name. In addition, the
     * following postprocessed quantities can be selected:
     *
     *  - "schlieren_<name>": the magnitude of the gradient of the
     *    component <name>,
     *    \f[
     *      q_i = \Big|\frac{1}{m_i}\sum_{j\in \mathcal{J}(i)}
     *            \mathbf{c}_{ij} u_j\Big|;
     *    \f]
     *  - "vorticity_<name>": the curl of the vector field formed by the
     *    component <name> and its dim - 1 subsequent components, in 2D the
     *    (signed) scalar curl, in 3D the magnitude of the curl,
     *    \f[
     *      \mathbf q_i = \frac{1}{m_i}\sum_{j\in \mathcal{J}(i)}
     *                    \mathbf{c}_{ij} \times \mathbf{u}_j.
     *    \f]
     *
     * A single postprocessor serves several callers: every caller stores
     * the selection returned by prepare() and passes it to extract().
     *
     * Intended usage:
     * ```
     * const auto selection = postprocessor.prepare(selected_component_names);
     * // ...
     * const auto components = postprocessor.extract(selection, state_vector);
     * // or
     * postprocessor.extract(selection, state_vector, indices, n_points,
     * values);
     * ```
     *
     * @ingroup TimeLoop
     */
    template <int dim, typename Number = double>
    class Postprocessor
    {
    public:
      /**
       * Destructor.
       */
      virtual ~Postprocessor() = default;

      /**
       * A selection of quantities as returned by prepare(). The encoding
       * of sections and operations is defined by the implementation.
       */
      struct Selection {
        /**
         * Return the number of selected quantities.
         */
        unsigned int size() const
        {
          return static_cast<unsigned int>(quantities.size());
        }

        /**
         * The components to extract: every selected component, and the
         * operands of every postprocessed quantity, each a section and the
         * index in the list of names of the section.
         */
        std::vector<std::pair<std::uint8_t, unsigned int>> components;

        /**
         * The selected quantities: the operation, and the position of its
         * first operand in components.
         */
        std::vector<std::pair<std::uint8_t, unsigned int>> quantities;
      };

      /**
       * Look up the list of @p selected component names and return the
       * corresponding selection.
       *
       * @note The component names are read from the hyperbolic and
       * parabolic system when this function is called. It must thus be
       * called after the run time parameters have been parsed.
       */
      virtual Selection
      prepare(const std::vector<std::string> &selected) const = 0;

      /**
       * Extract all quantities of @p selection out of @p state_vector and
       * return them as a vector of host vectors with one entry per selected
       * quantity, in the order in which the quantities have been selected
       * in prepare().
       *
       * @note The ghost values of the returned vectors are updated.
       * Postprocessed quantities have their constrained degrees of freedom
       * populated.
       */
      virtual std::vector<ScalarHostVector<Number>>
      extract(const Selection &selection,
              const StateVector<Number> &state_vector) const = 0;

      /**
       * Extract all quantities of @p selection at the degrees of freedom
       * indices[p], p = 0, ..., n_points - 1, out of @p state_vector:
       * ```
       * values[p * selection.size() + k] = value of the k-th selected
       *                                    component at indices[p].
       * ```
       *
       * The extraction runs on the memory space selected_memory_space_t,
       * and both pointers refer to it. Depending on the transfer policy of
       * the individual vectors this either asserts that the memory space is
       * resident, or triggers an implicit memory transfer.
       *
       * @note Postprocessed quantities cannot be extracted at points.
       */
      virtual void extract(const Selection &selection,
                           const StateVector<Number> &state_vector,
                           const unsigned int *indices,
                           const unsigned int n_points,
                           Number *values) const = 0;
    };
  } // namespace Interface
} // namespace ryujin
