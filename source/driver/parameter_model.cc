//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/parameter_model.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <cstdio>
#include <ranges>
#include <sstream>
#include <utility>

namespace ryujin
{
  namespace
  {
    using ptree = boost::property_tree::ptree;
    using Entry = ParameterModel::Entry;


    /*
     * dealii::Patterns::pattern_factory() knows all patterns but
     * Patterns::Tuple, on which it aborts, also as part of a List or a Map:
     */
    std::unique_ptr<dealii::Patterns::PatternBase>
    create_pattern(const std::string &description)
    {
      if (description.find("[Tuple") != std::string::npos)
        return nullptr;
      return dealii::Patterns::pattern_factory(description);
    }


    /* Split the choices of a "[Selection a|b|c ]" description: */
    std::vector<std::string> selection_choices(const std::string &description)
    {
      const std::string prefix = "[Selection ";
      auto choices = description.substr(prefix.size());
      choices = choices.substr(0, choices.rfind(" ]"));

      std::vector<std::string> result;
      for (const auto choice : std::views::split(choices, '|'))
        result.emplace_back(choice.begin(), choice.end());
      return result;
    }


    void classify(Entry &entry)
    {
      namespace P = dealii::Patterns;

      entry.pattern = create_pattern(entry.pattern_description);
      const auto pattern = entry.pattern.get();

      /* Bool is derived from Selection, test it first: */
      if (dynamic_cast<const P::Bool *>(pattern) != nullptr) {
        entry.kind = Entry::Kind::boolean;
        entry.choices = {"true", "false"};
      } else if (dynamic_cast<const P::Selection *>(pattern) != nullptr) {
        entry.kind = Entry::Kind::selection;
        entry.choices = selection_choices(entry.pattern_description);
      } else if (dynamic_cast<const P::Integer *>(pattern) != nullptr) {
        entry.kind = Entry::Kind::integer;

        /* Offer a small range of integers as a selection: */
        long lower = 0;
        long upper = 0;
        if (std::sscanf(entry.pattern_description.c_str(),
                        "[Integer range %ld...%ld",
                        &lower,
                        &upper) == 2 &&
            upper >= lower && upper - lower < 8) {
          entry.kind = Entry::Kind::selection;
          for (long i = lower; i <= upper; ++i)
            entry.choices.push_back(std::to_string(i));
        }
      } else if (dynamic_cast<const P::Double *>(pattern) != nullptr) {
        entry.kind = Entry::Kind::floating;
      } else {
        entry.kind = Entry::Kind::text;
      }
    }


    /* An entry is a node with a value and a pattern, see parameter_handler.h:
     */
    bool is_entry(const ptree &node)
    {
      return node.find("value") != node.not_found() &&
             node.find("pattern") != node.not_found();
    }
  } // namespace


  ParameterValues read_parameter_values(const std::string &text)
  {
    const auto trim = [](std::string string) {
      const auto begin = string.find_first_not_of(" \t\r");
      if (begin == std::string::npos)
        return std::string();
      const auto end = string.find_last_not_of(" \t\r");
      return string.substr(begin, end - begin + 1);
    };

    ParameterValues values;
    std::vector<std::string> path;

    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
      /* Join continued lines: */
      while (!line.empty() && line.back() == '\\') {
        line.pop_back();
        std::string next;
        if (!std::getline(stream, next))
          break;
        line += next;
      }

      line = trim(line.substr(0, line.find('#')));

      if (line.starts_with("subsection ")) {
        path.push_back(trim(line.substr(11)));
      } else if (line == "end") {
        if (!path.empty())
          path.pop_back();
      } else if (line.starts_with("set ")) {
        const auto equal = line.find('=');
        if (equal == std::string::npos)
          continue;
        values[{path, trim(line.substr(4, equal - 4))}] =
            trim(line.substr(equal + 1));
      }
    }

    return values;
  }


  bool ParameterModel::Entry::valid() const
  {
    return pattern == nullptr || pattern->match(value);
  }


  ParameterModel::ParameterModel(const dealii::ParameterHandler &prm)
  {
    /*
     * The tree of the ParameterHandler is private. Print it as JSON, with
     * demangled names, and read it back in:
     */
    std::stringstream stream;
    prm.print_parameters(stream,
                         dealii::ParameterHandler::JSON |
                             dealii::ParameterHandler::KeepDeclarationOrder);
    ptree tree;
    boost::property_tree::read_json(stream, tree);

    const auto walk = [this](const auto &self,
                             const ptree &node,
                             const std::vector<std::string> &path) -> void {
      const auto section = sections_.size();
      sections_.emplace_back().path = path;

      std::vector<std::pair<std::string, const ptree *>> subsections;
      for (const auto &[name, child] : node) {
        if (!is_entry(child)) {
          subsections.emplace_back(name, &child);
          continue;
        }

        Entry entry;
        entry.section = path;
        entry.name = name;
        entry.value = child.get<std::string>("value");
        entry.default_value = child.get<std::string>("default_value", "");
        entry.documentation = child.get<std::string>("documentation", "");
        entry.pattern_description =
            child.get<std::string>("pattern_description", "");
        entry.initial_value = entry.value;
        classify(entry);

        sections_[section].entries.push_back(entries_.size());
        entries_.push_back(std::move(entry));
      }

      /* List the entries of a section before its subsections: */
      for (const auto &[name, child] : subsections) {
        auto subpath = path;
        subpath.push_back(name);
        self(self, *child, subpath);
      }
    };
    walk(walk, tree, {});

    /* Drop the top level section if it has no entries of its own: */
    if (sections_.front().entries.empty())
      sections_.erase(sections_.begin());

    /*
     * Link every section to its enclosing section, and to a selection of
     * the enclosing section that has the name of the section as a choice:
     */
    for (auto &section : sections_) {
      if (section.path.empty())
        continue;

      const std::vector parent_path(section.path.begin(),
                                    section.path.end() - 1);
      const auto parent =
          std::ranges::find(sections_, parent_path, &Section::path);
      if (parent == sections_.end())
        continue;
      section.parent = static_cast<std::size_t>(parent - sections_.begin());

      for (const auto index : parent->entries) {
        const auto &entry = entries_[index];
        if (entry.kind == Entry::Kind::selection &&
            std::ranges::find(entry.choices, section.path.back()) !=
                entry.choices.end()) {
          section.selector = index;
          break;
        }
      }
    }
  }


  auto ParameterModel::find(const std::vector<std::string> &section,
                            const std::string &name) const -> const Entry *
  {
    const auto it = std::ranges::find_if(entries_, [&](const auto &entry) {
      return entry.section == section && entry.name == name;
    });
    return it == entries_.end() ? nullptr : &*it;
  }


  auto ParameterModel::find(const std::vector<std::string> &section,
                            const std::string &name) -> Entry *
  {
    return const_cast<Entry *>(std::as_const(*this).find(section, name));
  }


  std::size_t ParameterModel::n_invalid(std::size_t section) const
  {
    return std::ranges::count_if(
        sections_[section].entries,
        [this](std::size_t index) { return !entries_[index].valid(); });
  }


  std::size_t ParameterModel::n_invalid() const
  {
    std::size_t result = 0;
    for (std::size_t i = 0; i < sections_.size(); ++i)
      result += active(i) ? n_invalid(i) : 0;
    return result;
  }


  bool ParameterModel::modified() const
  {
    return std::ranges::any_of(entries_, &Entry::modified);
  }


  bool ParameterModel::active(std::size_t section) const
  {
    for (std::optional<std::size_t> i = section; i; i = sections_[*i].parent) {
      const auto &current = sections_[*i];
      if (current.selector &&
          entries_[*current.selector].value != current.path.back())
        return false;
    }
    return true;
  }


  ParameterValues ParameterModel::values() const
  {
    ParameterValues values;
    for (const auto &entry : entries_)
      values[{entry.section, entry.name}] = entry.value;
    return values;
  }


  void ParameterModel::assign(const ParameterValues &values, bool keep_invalid)
  {
    for (auto &entry : entries_) {
      const auto it = values.find({entry.section, entry.name});
      if (it == values.end())
        continue;

      /* A boolean or a selection can only take one of its choices: */
      const bool keep = keep_invalid && entry.choices.empty();

      const auto previous = entry.value;
      entry.value = it->second;
      if (!keep && !entry.valid()) {
        entry.value = previous;
        continue;
      }
      entry.initial_value = entry.value;
    }
  }


  void ParameterModel::apply(dealii::ParameterHandler &prm) const
  {
    for (const auto &entry : entries_) {
      if (!entry.valid())
        continue;

      for (const auto &name : entry.section)
        prm.enter_subsection(name);
      try {
        prm.set(entry.name, entry.value);
      } catch (const dealii::ParameterHandler::ExcValueDoesNotMatchPattern &) {
        /* The pattern is not known to the model (a Patterns::Tuple): */
      }
      for ([[maybe_unused]] const auto &name : entry.section)
        prm.leave_subsection();
    }
  }


  std::string
  ParameterModel::remove_inactive_sections(const std::string &text) const
  {
    std::vector<std::string> path;
    /* The depth of the inactive section we are in, zero if none: */
    std::size_t skip = 0;

    std::istringstream stream(text);
    std::ostringstream result;
    for (std::string line; std::getline(stream, line);) {
      const auto content =
          line.substr(std::min(line.find_first_not_of(' '), line.size()));

      if (content.starts_with("subsection ")) {
        path.push_back(content.substr(11));
        const auto it = std::ranges::find(sections_, path, &Section::path);
        if (skip == 0 && it != sections_.end() &&
            !active(static_cast<std::size_t>(it - sections_.begin())))
          skip = path.size();
      }

      if (skip == 0)
        result << line << '\n';

      if (content == "end" && !path.empty()) {
        if (path.size() == skip)
          skip = 0;
        path.pop_back();
      }
    }

    return result.str();
  }
} // namespace ryujin
