//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/library.h>

#include <iostream>

//
// Test the class template Library: built-in entries from defaults(), an
// entry added during static initialization, entries added after a first
// create(), and the detection of a duplicate name.
//

using namespace ryujin;

class Shape
{
public:
  Shape(const std::string &name, const std::string &subsection)
      : name_(subsection + "/" + name)
  {
  }

  virtual ~Shape() = default;

  ACCESSOR_READ_ONLY(name)

private:
  const std::string name_;
};

class Circle final : public Shape
{
public:
  Circle(const std::string &subsection)
      : Shape("circle", subsection)
  {
  }
};

class Square final : public Shape
{
public:
  Square(const std::string &subsection)
      : Shape("square", subsection)
  {
  }
};

class Triangle final : public Shape
{
public:
  Triangle(const std::string &subsection)
      : Shape("triangle", subsection)
  {
  }
};

using ShapeLibrary = Library<Shape, const std::string &>;

/* The header of the instance: */

namespace ryujin
{
  template <>
  auto ShapeLibrary::defaults() -> std::vector<Creator>;

  extern template class Library<Shape, const std::string &>;
} // namespace ryujin

/* The translation unit that instantiates it: */

namespace ryujin
{
  template <>
  auto ShapeLibrary::defaults() -> std::vector<Creator>
  {
    return {creator<Circle>(), creator<Square>()};
  }

  template class Library<Shape, const std::string &>;
} // namespace ryujin

/* A client that registers an entry during static initialization: */

struct RegisterTriangle {
  RegisterTriangle()
  {
    ShapeLibrary::add<Triangle>();
  }
} register_triangle;

void print_names()
{
  std::string separator;
  for (const auto &name : ShapeLibrary::names("/S")) {
    std::cout << separator << name;
    separator = " ";
  }
  std::cout << std::endl;
}

int main()
{
  print_names();

  ShapeLibrary::add([](const std::string &subsection) {
    return std::make_unique<Shape>("hexagon", subsection);
  });
  print_names();

  ShapeLibrary::add<Square>();
  try {
    ShapeLibrary::create("/S");
  } catch (const dealii::ExceptionBase &exception) {
    exception.print_info(std::cout);
  }
}
