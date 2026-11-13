//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <discretization/butcher_tableau.h>

#include <cmath>
#include <iostream>
#include <vector>

//
// Verify the Butcher tableaux of all Runge-Kutta schemes: the row sums of
// the explicit and implicit tableaux have to coincide with the
// coefficients c, and the order conditions have to be satisfied up to the
// reported order. For implicit-explicit schemes this includes all
// coupling conditions (up to order three).
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

void test(const TimeSteppingScheme scheme, const std::string &name)
{
  const auto tableau = butcher_tableau<double>(scheme);
  const auto &c = tableau.c;
  const std::size_t n = c.size();

  /* All tableaux that take part in the scheme: */
  std::vector<Matrix> tableaux{expand(tableau.a_explicit, n)};
  if (tableau.is_imex())
    tableaux.push_back(expand(tableau.a_implicit, n));

  const Vector one(n, 1.);

  bool consistent = std::abs(c.front()) < tolerance && //
                    std::abs(c.back() - 1.) < tolerance;
  for (const auto &a : tableaux) {
    const auto row_sum = apply(a, one);
    for (std::size_t l = 0; l < n; ++l)
      consistent = consistent && std::abs(row_sum[l] - c[l]) < tolerance;
  }

  /* The weights are the last row of the tableau: */

  const auto check = [&](const auto &defect) {
    return std::abs(defect) < tolerance;
  };

  bool order_1 = true;
  bool order_2 = true;
  bool order_3 = true;
  bool order_4 = !tableau.is_imex(); /* only verified for explicit schemes */

  const auto c2 = multiply(c, c);
  const auto c3 = multiply(c2, c);

  for (const auto &a_b : tableaux) {
    const auto &b = a_b.back();
    order_1 = order_1 && check(dot(b, one) - 1.);
    order_2 = order_2 && check(dot(b, c) - 1. / 2.);
    order_3 = order_3 && check(dot(b, c2) - 1. / 3.);
    order_4 = order_4 && check(dot(b, c3) - 1. / 4.);

    for (const auto &a : tableaux) {
      const auto ac = apply(a, c);
      order_3 = order_3 && check(dot(b, ac) - 1. / 6.);
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

  std::cout << name << ": stages = " << n - 1 << ", "
            << (tableau.is_imex() ? "imex" : "explicit")
            << ", consistent = " << (consistent ? "true" : "false")
            << ", order = " << order << std::endl;
}

int main()
{
  test(TimeSteppingScheme::erk_11, "erk 11");
  test(TimeSteppingScheme::erk_22, "erk 22");
  test(TimeSteppingScheme::erk_33, "erk 33");
  test(TimeSteppingScheme::erk_43, "erk 43");
  test(TimeSteppingScheme::erk_54, "erk 54");
  test(TimeSteppingScheme::strang_erk_33_cn, "strang erk 33 cn");
  test(TimeSteppingScheme::strang_erk_43_cn, "strang erk 43 cn");
  test(TimeSteppingScheme::imex_11, "imex 11");
  test(TimeSteppingScheme::imex_22, "imex 22");
  test(TimeSteppingScheme::imex_33, "imex 33");
  test(TimeSteppingScheme::gsa_imex_22, "gsa imex 22");
  test(TimeSteppingScheme::gsa_imex_43, "gsa imex 43");

  return 0;
}
