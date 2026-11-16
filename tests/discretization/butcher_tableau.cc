//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/discretization/butcher_tableau.h>

#include <cmath>
#include <iostream>
#include <vector>

//
// Verify the Butcher tableaux of all Runge-Kutta schemes: the row sums of
// the explicit tableau have to coincide with the coefficients c, the row
// sums of the implicit tableau have to coincide with c for every stage
// with an implicit solve, and the order conditions have to be satisfied
// up to the reported order. For implicit-explicit schemes this includes
// all coupling conditions (up to order three), formulated with the row
// sums of the two tableaux. We also report the number of implicit solves.
//

using namespace ryujin;

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

constexpr double tolerance = 1.e-12;

/* Convert a tableau with omitted trailing zeros into a square matrix: */
Matrix expand(const Matrix &a, const std::size_t n)
{
  Matrix result(n, Vector(n, 0.));
  for (std::size_t l = 0; l < n; ++l)
    for (std::size_t k = 0; k < n; ++k)
      result[l][k] = ButcherTableau<double>::entry(a, l, k);
  return result;
}

Vector apply(const Matrix &a, const Vector &x)
{
  Vector result(x.size(), 0.);
  for (std::size_t l = 0; l < x.size(); ++l)
    for (std::size_t k = 0; k < x.size(); ++k)
      result[l] += a[l][k] * x[k];
  return result;
}

double dot(const Vector &x, const Vector &y)
{
  double result = 0.;
  for (std::size_t l = 0; l < x.size(); ++l)
    result += x[l] * y[l];
  return result;
}

Vector multiply(const Vector &x, const Vector &y)
{
  Vector result(x.size());
  for (std::size_t l = 0; l < x.size(); ++l)
    result[l] = x[l] * y[l];
  return result;
}

void test(const ButcherTableau<double> &tableau, const std::string &name)
{
  using Type = ButcherTableau<double>::Type;

  const auto &c = tableau.c;
  const std::size_t n = c.size();
  const auto type = tableau.type();

  /* All tableaux that take part in the scheme: */
  std::vector<Matrix> tableaux;
  if (type != Type::dirk)
    tableaux.push_back(expand(tableau.a_explicit, n));
  if (type != Type::erk)
    tableaux.push_back(expand(tableau.a_implicit, n));

  const Vector one(n, 1.);

  /* The abscissae of a tableau are its row sums: */
  std::vector<Vector> abscissae;
  for (const auto &a : tableaux)
    abscissae.push_back(apply(a, one));

  unsigned int solves = 0;
  if (type != Type::erk)
    for (std::size_t l = 0; l < n; ++l)
      solves += tableaux.back()[l][l] != 0. ? 1 : 0;

  bool consistent = std::abs(c.front()) < tolerance && //
                    std::abs(c.back() - 1.) < tolerance;
  for (std::size_t s = 0; s < tableaux.size(); ++s) {
    const bool is_explicit = type != Type::dirk && s == 0;
    for (std::size_t l = 0; l < n; ++l)
      if (is_explicit || l + 1 == n || tableaux[s][l][l] != 0.)
        consistent = consistent && std::abs(abscissae[s][l] - c[l]) < tolerance;
  }

  /* The weights are the last row of the tableau: */

  const auto check = [&](const auto &defect) {
    return std::abs(defect) < tolerance;
  };

  bool order_1 = true;
  bool order_2 = true;
  bool order_3 = true;
  bool order_4 = type != Type::imex; /* not verified for imex */

  for (const auto &a_b : tableaux) {
    const auto &b = a_b.back();
    order_1 = order_1 && check(dot(b, one) - 1.);

    for (std::size_t s = 0; s < tableaux.size(); ++s) {
      const auto &c_s = abscissae[s];
      order_2 = order_2 && check(dot(b, c_s) - 1. / 2.);

      for (const auto &c_t : abscissae)
        order_3 = order_3 && check(dot(b, multiply(c_s, c_t)) - 1. / 3.);

      for (const auto &a : tableaux)
        order_3 = order_3 && check(dot(b, apply(a, c_s)) - 1. / 6.);
    }

    /* Fourth order conditions of a single tableau: */
    if (type != Type::imex) {
      const auto &a = a_b;
      const auto c2 = multiply(c, c);
      const auto ac = apply(a, c);
      order_4 = order_4 && check(dot(b, multiply(c2, c)) - 1. / 4.);
      order_4 = order_4 && check(dot(b, multiply(c, ac)) - 1. / 8.);
      order_4 = order_4 && check(dot(b, apply(a, c2)) - 1. / 12.);
      order_4 = order_4 && check(dot(b, apply(a, ac)) - 1. / 24.);
    }
  }

  const unsigned int order = !order_1   ? 0
                             : !order_2 ? 1
                             : !order_3 ? 2
                             : !order_4 ? 3
                                        : 4;

  const std::string type_name = type == Type::erk    ? "erk"
                                : type == Type::dirk ? "dirk"
                                                     : "imex";

  std::cout << name << ": stages = " << n - 1 << ", " << type_name
            << ", consistent = " << (consistent ? "true" : "false")
            << ", order = " << order << ", solves = " << solves << std::endl;
}

int main()
{
  using enum RungeKuttaMethod;

  const auto run = [](const RungeKuttaMethod method, const std::string &name) {
    test(butcher_tableau<double>(method), name);
  };

  run(erk_11, "erk 11");
  run(erk_22, "erk 22");
  run(erk_33, "erk 33");
  run(erk_43, "erk 43");
  run(erk_54, "erk 54");
  run(imex_11, "imex 11");
  run(imex_22, "imex 22");
  run(imex_33, "imex 33");
  run(gsa_imex_22, "gsa imex 22");
  run(gsa_imex_43, "gsa imex 43");
  run(dirk_midpoint, "dirk midpoint");
  run(dirk_22, "dirk 22");
  run(dirk_23, "dirk 23");

  return 0;
}
