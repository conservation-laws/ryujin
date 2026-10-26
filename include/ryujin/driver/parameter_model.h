//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ryujin
{
  /**
   * Parameter values, indexed by the path of subsections and the name of
   * the entry.
   *
   * @ingroup Driver
   */
  using ParameterValues =
      std::map<std::pair<std::vector<std::string>, std::string>, std::string>;


  /**
   * Read all values of a parameter file in the PRM format given by
   * @p text. In contrast to dealii::ParameterHandler::parse_input() the
   * entries do not have to be declared, and values are not checked
   * against a pattern.
   *
   * @ingroup Driver
   */
  ParameterValues read_parameter_values(const std::string &text);


  /**
   * An editable copy of all sections and entries of a
   * dealii::ParameterHandler. The parameter editor works on this model
   * and writes the modified values back with apply().
   *
   * @ingroup Driver
   */
  class ParameterModel
  {
  public:
    /**
     * A single parameter.
     */
    struct Entry {
      /**
       * The kind of input an entry takes, determined by its pattern.
       */
      enum class Kind : std::uint8_t {
        /** Patterns::Bool, "true" or "false". */
        boolean,
        /** Patterns::Selection, or an Integer of a small range: one of the
         * choices. */
        selection,
        /** Patterns::Integer. */
        integer,
        /** Patterns::Double. */
        floating,
        /** Any other pattern: a free form string. */
        text,
      };

      /* The path of subsections: */
      std::vector<std::string> section;
      std::string name;

      std::string value;
      std::string default_value;
      std::string documentation;
      std::string pattern_description;

      Kind kind = Kind::text;
      /* The choices of a selection, or "true" and "false": */
      std::vector<std::string> choices;

      /* The pattern reconstructed from its description, or null if the
       * description is not understood by dealii::Patterns::pattern_factory:
       */
      std::unique_ptr<const dealii::Patterns::PatternBase> pattern;

      /* The value the entry had when the model was created: */
      std::string initial_value;

      /**
       * Return whether the value matches the pattern.
       */
      bool valid() const;

      /**
       * Return whether the value has been changed.
       */
      bool modified() const
      {
        return value != initial_value;
      }
    };

    /**
     * A subsection with the indices of its entries.
     */
    struct Section {
      std::vector<std::string> path;
      std::vector<std::size_t> entries;

      /* The enclosing section, if any: */
      std::optional<std::size_t> parent;

      /* The selection in the enclosing section that names this section
       * as one of its choices, such as "geometry" or "equation of state":
       */
      std::optional<std::size_t> selector;
    };

    /**
     * Constructor. Read all sections and entries of @p prm in
     * declaration order.
     */
    explicit ParameterModel(const dealii::ParameterHandler &prm);

    std::vector<Entry> &entries()
    {
      return entries_;
    }

    const std::vector<Entry> &entries() const
    {
      return entries_;
    }

    /**
     * All sections in declaration order, depth first. A section comes
     * before its subsections.
     */
    const std::vector<Section> &sections() const
    {
      return sections_;
    }

    /**
     * Return the entry @p name in the section @p section, or a null
     * pointer if there is no such entry.
     */
    const Entry *find(const std::vector<std::string> &section,
                      const std::string &name) const;

    Entry *find(const std::vector<std::string> &section,
                const std::string &name);

    /**
     * Return the number of values of the section with index @p section
     * that do not match their pattern.
     */
    std::size_t n_invalid(std::size_t section) const;

    /**
     * Return the number of values of all sections in use (see active())
     * that do not match their pattern.
     */
    std::size_t n_invalid() const;

    /**
     * Return whether any value has been changed.
     */
    bool modified() const;

    /**
     * Return whether the section with index @p section is in use: the
     * selectors of the section and of all enclosing sections name the
     * respective section.
     */
    bool active(std::size_t section) const;

    /**
     * Return the values of all entries.
     */
    ParameterValues values() const;

    /**
     * Take over the values of all entries listed in @p values, as the
     * values the entries start with. Values that do not match the pattern
     * of their entry are skipped for booleans and selections, and for all
     * entries if @p keep_invalid is not set.
     */
    void assign(const ParameterValues &values, bool keep_invalid = true);

    /**
     * Set all valid values in @p prm.
     */
    void apply(dealii::ParameterHandler &prm) const;

    /**
     * Remove all sections that are not in use (see active()) from the
     * parameter file @p text in the PRM format.
     */
    std::string remove_inactive_sections(const std::string &text) const;

  private:
    std::vector<Entry> entries_;
    std::vector<Section> sections_;
  };
} // namespace ryujin
