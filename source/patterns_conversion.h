//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <deal.II/base/parameter_acceptor.h>

#include <numeric>

DEAL_II_NAMESPACE_OPEN


template <typename T>
struct ConversionHelper : std::false_type {
};


template <typename T>
struct Patterns::Tools::
    Convert<T, typename std::enable_if_t<ConversionHelper<T>::value>> {

  static std::unique_ptr<Patterns::PatternBase> to_pattern()
  {
    static const auto conversion_table = ConversionHelper<T>().conversion_table;
    std::string s;
    for (const auto &it : conversion_table)
      s = s.empty() ? std::get<1>(it) : s + "|" + std::get<1>(it);
    return std::make_unique<Patterns::Selection>(s);
  }

  static std::string
  to_string(const T &t,
            const Patterns::PatternBase & = *Convert<T>::to_pattern())
  {
    static const auto conversion_table = ConversionHelper<T>().conversion_table;
    for (const auto &it : conversion_table) {
      if (std::get<0>(it) == t)
        return std::get<1>(it);
    }

    AssertThrow(false,
                dealii::ExcMessage("Incomplete conversion table - unable to "
                                   "identify matching string"));
  }

  static T
  to_value(const std::string &s,
           const Patterns::PatternBase &pattern = *Convert<T>::to_pattern())
  {
    AssertThrow(pattern.match(s), ExcNoMatch(s, pattern.description()))

    static const auto conversion_table = ConversionHelper<T>().conversion_table;
    for (const auto &it : conversion_table) {
      if (std::get<1>(it) == s)
        return std::get<0>(it);
    }

    AssertThrow(false, dealii::ExcInternalError());
  }
};

DEAL_II_NAMESPACE_CLOSE

#define LIST(...) __VA_ARGS__

#define DECLARE_ENUM(type, s)                                                  \
  namespace dealii                                                             \
  {                                                                            \
    template <>                                                                \
    struct ::dealii::ConversionHelper<type> : std::true_type {                 \
      const std::vector<std::tuple<type, std::string>> conversion_table = {s}; \
    };                                                                         \
  }
