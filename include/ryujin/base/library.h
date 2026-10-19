//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <algorithm>
#include <concepts>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ryujin
{
  /**
   * A registry of creator functions for the abstract base class @p Base.
   * A creator constructs an object of a class derived from @p Base from
   * the arguments @p Args.
   *
   * Library has only static members and is never constructed. One registry
   * exists per instance: Library<Geometry<2>, const std::string &> and
   * Library<Geometry<3>, const std::string &> are separate registries.
   *
   * Entries are not keyed by name when they are added. The name of an entry
   * is the name() of the object that it constructs: create() constructs
   * all entries and throws if two of them share a name.
   *
   * The registry starts with the built-in entries returned by defaults(),
   * which is called when the registry is first used. add() appends further
   * entries, at any time; an entry added after a call to create() is seen
   * by every subsequent call. Objects are created in the order of
   * registration.
   *
   * A client can add its own entries with add(), for example in main() or
   * in the constructor of an object at namespace scope.
   *
   * @ingroup Utilities
   */
  template <typename Base, typename... Args>
  class Library final
  {
  public:
    Library() = delete;

    /**
     * A function that constructs an object of a class derived from @p Base.
     */
    using Creator = std::function<std::unique_ptr<Base>(Args...)>;

    /**
     * Return a creator that constructs an object of class @p T.
     */
    template <typename T>
      requires std::derived_from<T, Base> && std::constructible_from<T, Args...>
    static Creator creator();

    /**
     * Append an entry to the creators() table.
     */
    static void add(Creator creator);

    /**
     * Append an entry to the creators() table that constructs an object of
     * class @p T with the specified arguments.
     */
    template <typename T>
      requires std::derived_from<T, Base> && std::constructible_from<T, Args...>
    static void add();

    /**
     * Construct one object per entry, in the order of registration.
     * Throws if two objects share a name.
     */
    static std::vector<std::unique_ptr<Base>> create(Args... args);

    /**
     * Return the names of the objects that create() constructs, in the same
     * order.
     */
    static std::vector<std::string> names(Args... args);

    /**
     * Return a dealii::Patterns::Selection of the names of the given
     * @p objects, as returned by create(). It is the pattern of the run time
     * parameter that selects one of the objects by name.
     */
    static dealii::Patterns::Selection
    selection(const std::vector<std::unique_ptr<Base>> &objects);

    /**
     * Return the object of the given @p objects, as returned by create(),
     * whose name() is @p name. Throws if there is none.
     */
    static Base &select(const std::vector<std::unique_ptr<Base>> &objects,
                        const std::string &name);

  private:
    /**
     * The registry, initialized with defaults() on first use. An entry
     * added during static initialization thus follows the built-in
     * entries, whatever the order of static initialization.
     */
    static std::vector<Creator> &creators();

    /**
     * The built-in entries. Explicitly specialized for every instance.
     */
    static std::vector<Creator> defaults();
  };


  // ------------------------------- inline functions --------------------------


  template <typename Base, typename... Args>
  template <typename T>
    requires std::derived_from<T, Base> && std::constructible_from<T, Args...>
  inline auto Library<Base, Args...>::creator() -> Creator
  {
    return [](Args... args) -> std::unique_ptr<Base> {
      return std::make_unique<T>(args...);
    };
  }


  template <typename Base, typename... Args>
  template <typename T>
    requires std::derived_from<T, Base> && std::constructible_from<T, Args...>
  inline void Library<Base, Args...>::add()
  {
    add(creator<T>());
  }


  // ----------------------------- member functions ---------------------------


  template <typename Base, typename... Args>
  auto Library<Base, Args...>::creators() -> std::vector<Creator> &
  {
    static std::vector<Creator> creators = defaults();
    return creators;
  }


  template <typename Base, typename... Args>
  void Library<Base, Args...>::add(Creator creator)
  {
    creators().push_back(std::move(creator));
  }


  template <typename Base, typename... Args>
  std::vector<std::unique_ptr<Base>>
  Library<Base, Args...>::create(Args... args)
  {
    std::vector<std::unique_ptr<Base>> objects;
    std::set<std::string> names;

    for (const auto &creator : creators()) {
      auto object = creator(args...);
      const auto [it, inserted] = names.insert(object->name());
      AssertThrow(
          inserted,
          dealii::ExcMessage("Two entries of a library share the name »" +
                             object->name() + "«"));
      objects.push_back(std::move(object));
    }

    return objects;
  }


  template <typename Base, typename... Args>
  std::vector<std::string> Library<Base, Args...>::names(Args... args)
  {
    std::vector<std::string> names;
    for (const auto &object : create(args...))
      names.push_back(object->name());
    return names;
  }


  template <typename Base, typename... Args>
  dealii::Patterns::Selection Library<Base, Args...>::selection(
      const std::vector<std::unique_ptr<Base>> &objects)
  {
    std::string names;
    for (const auto &object : objects)
      names += (names.empty() ? "" : "|") + object->name();
    return dealii::Patterns::Selection(names);
  }


  template <typename Base, typename... Args>
  Base &Library<Base, Args...>::select(
      const std::vector<std::unique_ptr<Base>> &objects,
      const std::string &name)
  {
    const auto it =
        std::find_if(objects.begin(), objects.end(), [&](const auto &object) {
          return object->name() == name;
        });

    AssertThrow(it != objects.end(),
                dealii::ExcMessage("No entry of the library has the name »" +
                                   name + "«"));

    return **it;
  }
} // namespace ryujin
