//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "state_vector.h"

namespace ryujin
{
  template <typename Description, int dim, typename Number>
  struct SelectedComponentsExtractor {
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    using StateVector = typename View::StateVector;
    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;
    using ScalarVector = Vectors::ScalarVector<Number>;

    SelectedComponentsExtractor() = delete;

    static void check(const std::vector<std::string> &selected)
    {
      const auto search = [&](const auto entry, const auto &names) {
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        return pos != std::end(names);
      };

      for (const auto &entry : selected) {
        const auto found = search(entry, View::component_names) ||
                           search(entry, View::primitive_component_names) ||
                           search(entry, View::precomputed_names) ||
                           search(entry, View::initial_precomputed_names) ||
                           (entry == "alpha");
        AssertThrow(
            found,
            dealii::ExcMessage(
                "Invalid component name: \"" + entry +
                "\" is not a valid conserved/primitive/precomputed/initial "
                "component name."));
      }
    }

    static std::vector<ScalarVector>
    extract(const HyperbolicSystem &hyperbolic_system,
            const StateVector &state_vector,
            const InitialPrecomputedVector &initial_precomputed,
            const ScalarVector &alpha,
            const std::vector<std::string> &selected)
    {
      /*
       * Match the selected_components strings against conserved,
       * primitive, and initial component names and record an index pair
       * matching return vector position and component index:
       */

      std::vector<std::tuple<std::size_t, std::size_t>> conserved_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> primitive_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> precomputed_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> initial_indices;
      std::vector<std::size_t> alpha_indices;

      for (std::size_t i = 0; const auto &entry : selected) {
        const auto search = [&](const auto &names, auto &indices) {
          const auto pos = std::find(std::begin(names), std::end(names), entry);
          if (pos != std::end(names)) {
            const auto index = std::distance(std::begin(names), pos);
            indices.push_back({i++, index});
            return true;
          }
          return false;
        };

        if (search(View::component_names, conserved_indices))
          ;
        else if (search(View::primitive_component_names, primitive_indices))
          ;
        else if (search(View::precomputed_names, precomputed_indices))
          ;
        else if (search(View::initial_precomputed_names, initial_indices))
          ;
        else if (entry == "alpha")
          alpha_indices.push_back(i++);
        else
          AssertThrow(false, dealii::ExcInternalError());
      }

      std::vector<ScalarVector> extracted_components(selected.size());
      const auto &scalar_partitioner = alpha.get_partitioner();
      for (auto &it : extracted_components)
        it.reinit(scalar_partitioner);

      for (const auto &[i, k] : conserved_indices) {
        const auto &U = std::get<0>(state_vector);
        U.extract_component(extracted_components[i], k);
      }

      if (!primitive_indices.empty()) {
        const auto &U = std::get<0>(state_vector);
        const unsigned int n_owned = scalar_partitioner->locally_owned_size();
        const auto view = hyperbolic_system.template view<dim, Number>();
        for (unsigned int i = 0; i < n_owned; ++i) {
          const auto U_i = U.get_tensor(i);
          const auto PU_i = view.to_primitive_state(U_i);
          for (const auto &[j, k] : primitive_indices)
            extracted_components[j].local_element(i) = PU_i[k];
        }
      }

      for (const auto &[i, k] : precomputed_indices) {
        const auto &prec = std::get<1>(state_vector);
        prec.extract_component(extracted_components[i], k);
      }

      for (const auto &[i, k] : initial_indices) {
        initial_precomputed.extract_component(extracted_components[i], k);
      }

      for (const auto &i : alpha_indices)
        extracted_components[i] = alpha;

      return extracted_components;
    }
  };
} // namespace ryujin
