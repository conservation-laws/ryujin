//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef GRID_AIRFOIL_H
#define GRID_AIRFOIL_H

#include <compile_time_options.h>

#include "cubic_spline.h"
#include "geometry.h"
#include "transfinite_interpolation.template.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tensor_product_manifold.h>
#include <deal.II/grid/tria.h>

namespace ryujin
{
  namespace Manifolds
  {
    /**
     * @todo Documentation
     */
    template <int dim>
    class AirfoilManifold : public dealii::ChartManifold<dim>
    {
      static_assert(dim == 2, "not implemented for dim != 2");

    public:
      AirfoilManifold(const dealii::Point<dim> airfoil_center,
                      const std::function<double(const double)> &psi_front,
                      const std::function<double(const double)> &psi_upper,
                      const std::function<double(const double)> &psi_lower,
                      const bool upper_side,
                      const double psi_ratio = 1.0)
          : airfoil_center(airfoil_center)
          , psi_front(psi_front)
          , psi_upper(psi_upper)
          , psi_lower(psi_lower)
          , upper_side(upper_side)
          , ratio_(psi_ratio * psi_front(0.) / psi_front(M_PI))
          , polar_manifold()
      {
        Assert(std::abs(psi_upper(0.) - psi_front(0.5 * M_PI)) < 1.0e-10,
               dealii::ExcInternalError());
        Assert(std::abs(psi_lower(0.) + psi_front(1.5 * M_PI)) < 1.0e-10,
               dealii::ExcInternalError());
      }

      virtual dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final override
      {
        auto coordinate = dealii::Point<dim>() + (space_point - airfoil_center);

        /* transform: */

        dealii::Point<dim> chart_point;
        if (coordinate[0] > 0.) {
          if (upper_side) {
            /* upper back airfoil part */
            chart_point[0] = 1. + coordinate[1] - psi_upper(coordinate[0]);
            chart_point[1] = 0.5 * M_PI - ratio_ * coordinate[0];
          } else {
            /* lower back airfoil part */
            chart_point[0] = 1. - coordinate[1] + psi_lower(coordinate[0]);
            chart_point[1] = 1.5 * M_PI + ratio_ * coordinate[0];
          }
        } else {
          /* front part */
          chart_point = polar_manifold.pull_back(coordinate);
          chart_point[0] = 1. + chart_point[0] - psi_front(chart_point[1]);
        }

        return chart_point;
      }

      virtual dealii::Point<dim>
      push_forward(const dealii::Point<dim> &point) const final override
      {
        auto chart_point = point;

        /* transform back */

        dealii::Point<dim> coordinate;
        if (chart_point[1] < 0.5 * M_PI) {
          Assert(upper_side, dealii::ExcInternalError());
          /* upper back airfoil part */
          coordinate[0] = (0.5 * M_PI - chart_point[1]) / ratio_;
          Assert(coordinate[0] >= -1.0e-10, dealii::ExcInternalError());
          coordinate[1] = chart_point[0] - 1. + psi_upper(coordinate[0]);
        } else if (chart_point[1] > 1.5 * M_PI) {
          Assert(!upper_side, dealii::ExcInternalError());
          /* lower back airfoil part */
          coordinate[0] = (chart_point[1] - 1.5 * M_PI) / ratio_;
          Assert(coordinate[0] >= -1.0e-10, dealii::ExcInternalError());
          coordinate[1] = 1. - chart_point[0] + psi_lower(coordinate[0]);
        } else {
          /* front part */
          chart_point[0] = chart_point[0] - 1. + psi_front(chart_point[1]);
          coordinate = polar_manifold.push_forward(chart_point);
        }

        return dealii::Point<dim>() + (coordinate + airfoil_center);
      }

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final override
      {
        const double psi_ratio = ratio_ * psi_front(M_PI) / psi_front(0.);
        return std::make_unique<AirfoilManifold<dim>>(airfoil_center,
                                                      psi_front,
                                                      psi_upper,
                                                      psi_lower,
                                                      upper_side,
                                                      psi_ratio);
      }

    private:
      const dealii::Point<dim> airfoil_center;
      const std::function<double(const double)> psi_front;
      const std::function<double(const double)> psi_upper;
      const std::function<double(const double)> psi_lower;
      const bool upper_side;

      const double ratio_;

      dealii::PolarManifold<dim> polar_manifold;
    };


    /**
     * @todo Documentation
     */
    template <int dim>
    class GradingManifold : public dealii::ChartManifold<dim>
    {
    public:
      GradingManifold(const dealii::Point<dim> center,
                      const dealii::Tensor<1, dim> direction,
                      const double grading,
                      const double epsilon)
          : center(center)
          , direction(direction)
          , grading(grading)
          , epsilon(epsilon)
      {
      }

      /* FIXME: find out why weights are not normalized. */
      virtual Point<dim>
      get_new_point(const ArrayView<const Point<dim>> &surrounding_points,
                    const ArrayView<const double> &weights) const override
      {
        if(weights[0] > 1.0)
          return surrounding_points[0];

        if(weights[1] > 1.0)
          return surrounding_points[1];

        return dealii::ChartManifold<dim>::get_new_point(surrounding_points,
                                                         weights);
      }

      virtual dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final override
      {
        auto chart_point = space_point - center;

        for (unsigned int d = 0; d < dim; ++d) {
          if (std::abs(direction[d]) > 1.0e-10) {
            const double x = chart_point[d] * std::copysign(1., direction[d]);
            Assert(x + epsilon > 0, dealii::ExcInternalError());
            const double x_hat = std::pow(x + epsilon, 1. / grading) -
                                 std::pow(epsilon, 1. / grading) + 1.e-14;
            chart_point[d] += (x_hat - x) * std::copysign(1., direction[d]);
          }
        }

        return dealii::Point<dim>() + chart_point;
      }

      virtual dealii::Point<dim>
      push_forward(const dealii::Point<dim> &chart_point) const final override
      {
        auto space_point = chart_point;

        for (unsigned int d = 0; d < dim; ++d) {
          if (std::abs(direction[d]) > 1.0e-10) {
            const double x_hat =
                space_point[d] * std::copysign(1., direction[d]);
            Assert(x_hat + std::pow(epsilon, 1. / grading) > 0,
                   dealii::ExcInternalError());
            const double x =
                std::pow(x_hat + std::pow(epsilon, 1. / grading), grading) -
                epsilon + 1.e-14;
            space_point[d] += (x - x_hat) * std::copysign(1., direction[d]);
          }
        }

        return center + (space_point - dealii::Point<dim>());
      }

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final override
      {
        return std::make_unique<GradingManifold<dim>>(
            center, direction, grading, epsilon);
      }

    private:
      const dealii::Point<dim> center;
      const dealii::Tensor<1, dim> direction;
      const double grading;
      const double epsilon;
    };


    /**
     * @todo Documentation
     */
    template <int dim>
    class ExtrudedManifold : public dealii::Manifold<dim>
    {
    public:
      ExtrudedManifold(const dealii::Manifold<dim - 1> &manifold)
          : manifold(manifold.clone())
      {
      }

      virtual std::unique_ptr<Manifold<dim>> clone() const override{
        return std::make_unique<ExtrudedManifold<dim>>(*manifold);
      }

      virtual Point<dim>
      get_new_point(const ArrayView<const Point<dim>> &surrounding_points,
                    const ArrayView<const double> &weights) const override
      {
        Assert(surrounding_points.size() == weights.size(),
               dealii::ExcInternalError());

        boost::container::small_vector<dealii::Point<dim - 1>, 100>
            surrounding_points_projected;
        std::transform(surrounding_points.begin(),
                       surrounding_points.end(),
                       surrounding_points_projected.begin(),
                       [](const dealii::Point<dim> &source) {
                         dealii::Point<dim - 1> result;
                         for (unsigned int d = 0; d < dim - 1; ++d)
                           result[d] = source[d];
                         return result;
                       });

        const auto projected = manifold->get_new_point(
            ArrayView<const Point<dim - 1>>{surrounding_points_projected.data(),
                                            weights.size()},
            weights);

        dealii::Point<dim> result;

        for (unsigned int d = 0; d < dim - 1; ++d)
          result[d] = projected[d];

        for (unsigned int i = 0; i < weights.size(); ++i)
          result[dim - 1] += weights[i] * surrounding_points[i][dim - 1];

        return result;
      }

    private:
      std::unique_ptr<const dealii::Manifold<dim - 1>> manifold;
    };

  } // namespace Manifolds


  namespace
  {
    /**
     * @todo Documentation
     */
    std::array<std::vector<double>, 4>
    naca_4digit_points(const std::string &serial_number,
                       const unsigned int n_samples)
    {
      AssertThrow(serial_number.size() == 4,
                  dealii::ExcMessage("Invalid NACA 4 digit serial number"));
      std::array<unsigned int, 4> digit;
      std::transform(serial_number.begin(),
                     serial_number.end(),
                     digit.begin(),
                     [](auto it) { return it - '0'; });

      /* thickness */
      const double t = 0.1 * digit[2] + 0.01 * digit[3];
      AssertThrow(t > 0.,
                  dealii::ExcMessage("Invalid NACA 4 digit serial number"));

      /* maximal chamber */
      const double m = 0.01 * digit[0];
      /* x position of maximal chamber */
      const double p = 0.1 * digit[1];

      std::vector<double> x_upper;
      std::vector<double> y_upper;
      std::vector<double> x_lower;
      std::vector<double> y_lower;

      for (unsigned int i = 0; i < n_samples; i++) {
        const double x = 1. * i / (n_samples - 1);
        const double y =
            5. * t *
            (0.2969 * std::sqrt(x) +
             x * (-0.126 + x * (-0.3516 + x * (0.2843 + x * (-0.1036)))));

        const double y_c = (x < p) ? m / (p * p) * (2. * p * x - x * x)
                                   : m / ((1. - p) * (1. - p)) *
                                         (1. - 2. * p + 2. * p * x - x * x);

        const double dy_c = (x < p) ? 2. * m / (p * p) * (p - x)
                                    : 2. * m / ((1. - p) * (1. - p)) * (p - x);

        const double theta = std::atan(dy_c);

        x_upper.emplace_back(x - y * std::sin(theta));
        y_upper.emplace_back(y_c + y * std::cos(theta));
        x_lower.emplace_back(x + y * std::sin(theta));
        y_lower.emplace_back(y_c - y * std::cos(theta));
      }

      /* Fix up roundoff errors: */
      y_upper.front() = 0.;
      y_upper.back() = 0.;
      y_lower.front() = 0.;
      y_lower.back() = 0.;

      return {{x_upper, y_upper, x_lower, y_lower}};
    }


    /**
     * @todo Documentation
     */
    std::array<std::vector<double>, 4>
    nasa_sc2(const std::string &serial_number)
    {
      if (serial_number == "0714") {
        std::vector<double> x_upper{
            .0,  .002, .005, .01, .02, .03, .04, .05, .07, .1,  .12, .15,
            .17, .2,   .22,  .25, .27, .3,  .33, .35, .38, .4,  .43, .45,
            .48, .50,  .53,  .55, .57, .6,  .62, .65, .68, .7,  .72, .75,
            .77, .8,   .82,  .85, .87, .9,  .92, .95, .97, .98, .99, 1.};

        std::vector<double> y_upper{
            .0,    .0095, .0158, .0219, .0293,  .0343,  .0381,  .0411,
            .0462, .0518, .0548, .0585, .0606,  .0632,  .0646,  .0664,
            .0673, .0685, .0692, .0696, .0698,  .0697,  .0695,  .0692,
            .0684, .0678, .0666, .0656, .0645,  .0625,  .0610,  .0585,
            .0555, .0533, .0509, .0469, .0439,  .0389,  .0353,  .0294,
            .0251, .0181, .0131, .0049, -.0009, -.0039, -.0071, -.0104};

        std::vector<double> x_lower{
            .0,  .002, .005, .01, .02, .03, .04, .05, .07, .1,  .12, .15, .17,
            .20, .22,  .25,  .28, .3,  .32, .35, .37, .4,  .42, .45, .48, .5,
            .53, .55,  .58,  .6,  .63, .65, .68, .70, .73, .75, .77, .80, .83,
            .85, .87,  .89,  .92, .94, .95, .96, .97, .98, .99, 1.};

        std::vector<double> y_lower{
            .0,     -.0093, -.016,  -.0221, -.0295, -.0344, -.0381, -.0412,
            -.0462, -.0517, -.0547, -.0585, -.0606, -.0633, -.0647, -.0666,
            -.068,  -.0687, -.0692, -.0696, -.0696, -.0692, -.0688, -.0676,
            -.0657, -.0644, -.0614, -.0588, -.0543, -.0509, -.0451, -.041,
            -.0346, -.0302, -.0235, -.0192, -.0150, -.0093, -.0048, -.0024,
            -.0013, -.0008, -.0016, -.0035, -.0049, -.0066, -.0085, -.0109,
            -.0137, -.0163};

        return {{x_upper, y_upper, x_lower, y_lower}};

      } else {

        AssertThrow(false,
                    dealii::ExcMessage("Invalid NASA SC(2) serial number"));
      }
    }


    /**
     * @todo Documentation
     */
    std::array<std::vector<double>, 4>
    onera(const std::string &serial_number)
    {
      if (serial_number == "OAT15a") {
        std::vector<double> x_upper{
            0.,          4e-06,       5.26667e-05, 0.000155333, 0.00031,
            0.000516667, 0.000773333, 0.001082,    0.00144467,  0.00186267,
            0.00234,     0.00287933,  0.00348533,  0.00416133,  0.004912,
            0.00574133,  0.00665333,  0.00765067,  0.00873667,  0.00991467,
            0.011186,    0.0125533,   0.0140173,   0.01558,     0.017242,
            0.0190047,   0.02087,     0.022838,    0.0249107,   0.0270893,
            0.0293747,   0.0317687,   0.0342727,   0.0368887,   0.039618,
            0.042462,    0.0454233,   0.0485033,   0.051704,    0.0550267,
            0.0584747,   0.0620493,   0.0657527,   0.0695867,   0.073554,
            0.0776567,   0.081896,    0.0862747,   0.0907953,   0.0954593,
            0.100269,    0.105227,    0.110335,    0.115595,    0.121009,
            0.126579,    0.132307,    0.138195,    0.144244,    0.150457,
            0.156836,    0.163381,    0.170095,    0.17698,     0.184037,
            0.191267,    0.198671,    0.206251,    0.214008,    0.221943,
            0.230057,    0.238351,    0.246825,    0.255481,    0.264317,
            0.273335,    0.282534,    0.291915,    0.301477,    0.311219,
            0.321141,    0.331243,    0.341522,    0.351978,    0.362609,
            0.373412,    0.384387,    0.395529,    0.406838,    0.418309,
            0.429938,    0.441723,    0.453659,    0.465742,    0.477967,
            0.490327,    0.502818,    0.515433,    0.528165,    0.541008,
            0.553953,    0.566992,    0.580116,    0.593315,    0.606581,
            0.619901,    0.633264,    0.646659,    0.660071,    0.67349,
            0.686899,    0.700284,    0.713629,    0.726918,    0.740133,
            0.753259,    0.766279,    0.779176,    0.791934,    0.804535,
            0.816961,    0.829194,    0.841213,    0.852997,    0.864522,
            0.875766,    0.886705,    0.897311,    0.907561,    0.917426,
            0.926878,    0.935889,    0.944429,    0.952467,    0.959973,
            0.966915,    0.973261,    0.978979,    0.984038,    0.988404,
            0.99205,     0.994973,    0.997203,    1.};

        std::vector<double> y_upper{
            0.,         0.000323333, 0.001204,   0.00210267, 0.00301867,
            0.00395333, 0.00490533,  0.005876,   0.006866,   0.007874,
            0.0089,     0.00994267,  0.0110013,  0.0120733,  0.0131573,
            0.01425,    0.015348,    0.01645,    0.0175513,  0.0186513,
            0.019746,   0.020834,    0.021914,   0.022984,   0.0240433,
            0.0250913,  0.0261267,   0.02715,    0.0281607,  0.0291593,
            0.0301447,  0.0311187,   0.03208,    0.03303,    0.0339693,
            0.0348973,  0.0358147,   0.0367227,  0.03762,    0.0385087,
            0.039388,   0.040258,    0.0411193,  0.0419727,  0.042818,
            0.0436547,  0.044484,    0.0453047,  0.0461173,  0.046922,
            0.0477173,  0.0485047,   0.049284,   0.050054,   0.0508153,
            0.051568,   0.0523127,   0.0530473,  0.0537727,  0.054488,
            0.0551933,  0.0558873,   0.05657,    0.0572407,  0.057898,
            0.0585427,  0.0591733,   0.0597893,  0.06039,    0.0609753,
            0.0615427,  0.0620927,   0.062624,   0.063136,   0.0636273,
            0.0640973,  0.0645447,   0.064968,   0.065368,   0.0657413,
            0.0660887,  0.066408,    0.066698,   0.066958,   0.0671867,
            0.0673813,  0.067542,    0.067666,   0.0677527,  0.0677987,
            0.0678033,  0.067764,    0.0676793,  0.0675453,  0.0673613,
            0.0671233,  0.0668293,   0.0664767,  0.0660627,  0.0655833,
            0.0650367,  0.0644193,   0.063728,   0.06296,    0.062112,
            0.0611807,  0.0601627,   0.0590553,  0.057856,   0.0565607,
            0.0551667,  0.05367,     0.0520687,  0.05036,    0.0485473,
            0.0466367,  0.0446407,   0.0425753,  0.04046,    0.038314,
            0.0361553,  0.034,       0.0318593,  0.0297447,  0.027664,
            0.0256267,  0.023638,    0.0217053,  0.0198347,  0.0180313,
            0.0163,     0.0146447,   0.01307,    0.01158,    0.0101793,
            0.00887467, 0.00767267,  0.00658067, 0.005608,   0.00476267,
            0.00405333, 0.00348333,  0.00304733, 0.0025};

        std::vector<double> x_lower{
            0.,         1.13333e-05, 7.73333e-05, 0.000202,   0.000386667,
            0.00063,    0.000929333, 0.00128,     0.00167867, 0.00212067,
            0.00260267, 0.00312267,  0.00367733,  0.004266,   0.004888,
            0.00554333, 0.00623133,  0.006954,    0.00771133, 0.00850533,
            0.00933667, 0.010208,    0.01112,     0.012076,   0.0130767,
            0.0141253,  0.0152227,   0.016372,    0.0175753,  0.0188347,
            0.0201527,  0.021532,    0.0229747,   0.0244833,  0.0260607,
            0.0277087,  0.02943,     0.031228,    0.0331047,  0.0350627,
            0.0371053,  0.0392353,   0.0414547,   0.0437673,  0.0461753,
            0.0486813,  0.0512887,   0.054,       0.056818,   0.0597447,
            0.062784,   0.0659373,   0.069208,    0.0725987,  0.0761113,
            0.0797487,  0.0835127,   0.087406,    0.0914313,  0.09559,
            0.099886,   0.10432,     0.108896,    0.113615,   0.118479,
            0.123492,   0.128655,    0.13397,     0.13944,    0.145067,
            0.150852,   0.156798,    0.162907,    0.16918,    0.175619,
            0.182227,   0.189003,    0.19595,     0.203069,   0.210361,
            0.217828,   0.22547,     0.233289,    0.241284,   0.249457,
            0.257807,   0.266336,    0.275043,    0.283929,   0.292991,
            0.302232,   0.311649,    0.321241,    0.331009,   0.340949,
            0.35106,    0.361341,    0.371788,    0.382401,   0.393176,
            0.40411,    0.415201,    0.426444,    0.437837,   0.449375,
            0.461054,   0.47287,     0.484818,    0.496893,   0.509089,
            0.521401,   0.533823,    0.546347,    0.558967,   0.571675,
            0.584465,   0.597328,    0.610255,    0.623238,   0.636268,
            0.649334,   0.662426,    0.675534,    0.688646,   0.701751,
            0.714836,   0.727889,    0.740896,    0.753842,   0.766712,
            0.779489,   0.792154,    0.804687,    0.817068,   0.829273,
            0.841278,   0.853057,    0.864586,    0.875837,   0.886783,
            0.897397,   0.907653,    0.91752,     0.926972,   0.93598,
            0.944515,   0.952546,    0.960044,    0.966977,   0.973315,
            0.979025,   0.984075,    0.988434,    0.992073,   0.99499,
            0.997215,   1.};

        std::vector<double> y_lower{
            0.,           -0.000538667, -0.00138067, -0.002202,   -0.00299933,
            -0.003772,    -0.00451933,  -0.00524067, -0.00593667, -0.00660933,
            -0.00725933,  -0.00789067,  -0.008504,   -0.00910333, -0.00969067,
            -0.010268,    -0.010838,    -0.0114027,  -0.0119633,  -0.012522,
            -0.0130807,   -0.01364,     -0.0142013,  -0.014766,   -0.015336,
            -0.0159113,   -0.0164927,   -0.017082,   -0.01768,    -0.0182867,
            -0.0189033,   -0.01953,     -0.020168,   -0.0208173,  -0.0214793,
            -0.0221527,   -0.0228393,   -0.023538,   -0.0242487,  -0.024972,
            -0.0257073,   -0.026454,    -0.0272113,  -0.0279787,  -0.0287553,
            -0.02954,     -0.0303327,   -0.0311313,  -0.0319353,  -0.0327433,
            -0.0335553,   -0.0343707,   -0.0351873,  -0.036006,   -0.0368247,
            -0.0376447,   -0.038464,    -0.039282,   -0.0400993,  -0.0409153,
            -0.041728,    -0.042538,    -0.0433433,  -0.044144,   -0.044938,
            -0.0457247,   -0.046502,    -0.047268,   -0.0480213,  -0.0487593,
            -0.0494807,   -0.0501833,   -0.0508653,  -0.0515233,  -0.0521573,
            -0.0527633,   -0.0533413,   -0.053888,   -0.0544013,  -0.05488,
            -0.0553213,   -0.055724,    -0.056084,   -0.0564,     -0.056668,
            -0.0568867,   -0.057052,    -0.05716,    -0.0572093,  -0.0571953,
            -0.0571153,   -0.056966,    -0.0567447,  -0.056448,   -0.0560733,
            -0.0556187,   -0.0550813,   -0.05446,    -0.0537527,  -0.0529587,
            -0.0520787,   -0.051112,    -0.0500593,  -0.0489233,  -0.0477033,
            -0.046402,    -0.0450213,   -0.0435627,  -0.042028,   -0.0404207,
            -0.038742,    -0.0369967,   -0.035188,   -0.0333207,  -0.0313987,
            -0.0294287,   -0.027416,    -0.025368,   -0.0232913,  -0.0211947,
            -0.0190867,   -0.0169773,   -0.014878,   -0.0128,     -0.0107573,
            -0.008766,    -0.00684267,  -0.005006,   -0.003274,   -0.00166733,
            -0.000203333, 0.00110067,   0.00223133,  0.00317867,  0.00393733,
            0.004508,     0.00489467,   0.005108,    0.00516133,  0.005072,
            0.00485933,   0.00454067,   0.00413667,  0.00366467,  0.00314133,
            0.002582,     0.002002,     0.00141333,  0.000829333, 0.000262667,
            -0.000276,    -0.000774,    -0.001222,   -0.00160933, -0.00192867,
            -0.002178,    -0.00249667};

        return {{x_upper, y_upper, x_lower, y_lower}};

      } else {

        AssertThrow(false, dealii::ExcMessage("Invalid ONERA serial number"));
      }
    }


    /**
     * @todo Documentation
     */
    std::array<std::function<double(const double)>, 3>
    create_psi(const std::vector<double> &x_upper,
               const std::vector<double> &y_upper,
               const std::vector<double> &x_lower,
               const std::vector<double> &y_lower,
               const double x_center,
               const double y_center,
               const double scaling = 1.)
    {
      Assert(x_upper.size() >= 2, dealii::ExcInternalError());
      Assert(x_upper.front() == 0. && x_upper.back() == 1.,
             dealii::ExcInternalError());
      Assert(std::is_sorted(x_upper.begin(), x_upper.end()),
             dealii::ExcInternalError());

      Assert(x_lower.size() >= 2, dealii::ExcInternalError());
      Assert(x_lower.front() == 0. && x_lower.back() == 1.,
             dealii::ExcInternalError());
      Assert(std::is_sorted(x_lower.begin(), x_lower.end()),
             dealii::ExcInternalError());

      Assert(y_upper.size() == x_upper.size(), dealii::ExcInternalError());
      Assert(y_upper.front() == 0., dealii::ExcInternalError());

      Assert(y_lower.size() == x_lower.size(), dealii::ExcInternalError());
      Assert(y_lower.front() == 0., dealii::ExcInternalError());

      Assert(y_lower.back() < y_upper.back(), dealii::ExcInternalError());

      Assert(0. < x_center && x_center < 1., dealii::ExcInternalError());

#ifdef DEAL_II_WITH_GSL
      CubicSpline upper_airfoil(x_upper, y_upper);
      auto psi_upper =
          [upper_airfoil, x_center, y_center, scaling](const double x_hat) {
            /* Past the trailing edge return the the final upper y position: */
            const double x = x_hat / scaling;
            if (x > 1. - x_center)
              return scaling * (upper_airfoil.eval(1.0) - y_center);
            return scaling * (upper_airfoil.eval(x + x_center) - y_center);
          };

      CubicSpline lower_airfoil(x_lower, y_lower);

      auto psi_lower =
          [lower_airfoil, x_center, y_center, scaling](const double x_hat) {
            /* Past the trailing edge return the the final lower y position: */
            const double x = x_hat / scaling;
            if (x > 1. - x_center)
              return scaling * (lower_airfoil.eval(1.0) - y_center);
            return scaling * (lower_airfoil.eval(x + x_center) - y_center);
          };

      /*
       * Create a combined point set for psi_front:
       */

      std::vector<double> x_combined;
      std::vector<double> y_combined;

      for (std::size_t i = 0; i < x_upper.size(); ++i) {
        if (x_upper[i] >= x_center)
          break;
        x_combined.push_back(x_upper[i]);
        y_combined.push_back(y_upper[i]);
      }

      /*
       * We are about to create a spline interpolation in polar coordinates
       * for the front part. In order to blend this interpolation with the
       * two splines for the upper and lower part that we have just created
       * we have to add some additional sample points around the
       * coordinates were we glue together
       */
      for (double x : {x_center, x_center + 0.01, x_center + 0.02}) {
        x_combined.push_back(x);
        y_combined.push_back(upper_airfoil.eval(x));
      }

      std::reverse(x_combined.begin(), x_combined.end());
      std::reverse(y_combined.begin(), y_combined.end());
      x_combined.pop_back();
      y_combined.pop_back();

      for (std::size_t i = 0; i < x_lower.size(); ++i) {
        if (x_lower[i] >= x_center)
          break;
        x_combined.push_back(x_lower[i]);
        y_combined.push_back(y_lower[i]);
      }

      for (double x : {x_center, x_center + 0.01, x_center + 0.02}) {
        x_combined.push_back(x);
        y_combined.push_back(lower_airfoil.eval(x));
      }

      /* Translate into polar coordinates: */

      for (unsigned int i = 0; i < y_combined.size(); ++i) {
        const auto x = x_combined[i] - x_center;
        const auto y = y_combined[i] - y_center;

        const auto rho = std::sqrt(x * x + y * y);
        auto phi = std::atan2(y, x);
        if (phi < 0)
          phi += 2 * dealii::numbers::PI;

        x_combined[i] = phi;
        y_combined[i] = rho;
      }

      /* Ensure that x_combined is monotonically increasing: */
      if (x_combined.back() == 0.)
        x_combined.back() = 2. * dealii::numbers::PI;
      Assert(std::is_sorted(x_combined.begin(), x_combined.end()),
             dealii::ExcInternalError());

      CubicSpline front_airfoil(x_combined, y_combined);
      auto psi_front = [front_airfoil, x_center, scaling](const double phi) {
        /* By convention we return the "back length" for phi == 0.: */
        if (phi == 0.)
          return scaling * (1. - x_center);

        return scaling * front_airfoil.eval(phi);
      };

      return {{psi_front, psi_upper, psi_lower}};
#else
      AssertThrow(false, dealii::
                  ExcNotImplemented("Airfoil grid needs deal.II with GSL"));
      return {};
#endif
    }


  } // namespace

  /**
   * A namespace for a number of benchmark geometries and dealii::GridIn
   * wrappers.
   *
   * @ingroup Mesh
   */
  namespace Geometries
  {
    /**
     * A generic 2D Airfoil
     *
     * This class implements a generic 2D airfoil. Various runtime
     * parameters select the airfoil type (such as NACA 4 digit, some ONERA
     * airfoils, etc) and meshing behavior. The mesh construction is
     * divided into various steps:
     *
     * 1/ Parametrization:
     *
     * Depending on various runtime parameters a parameterization \f$ y =
     * \psi_{\text{up./lo.}}(x)\f$ on the upper and lower trailing edge as
     * well as a parameterization in polar coordinates
     * \f$ r = \psi_{\text{fr.}}(\phi) \f$ for the nose part is constructed.
     * This is done by taking a sample of points on the upper and lower
     * part of the airfoil and computing intermediate points with a cubic
     * spline interpolation. Relevant runtime parameters are `airfoil type`
     * to specify the type and serial number, `psi samples` to control the
     * number of samples taken (if admissible), and `psi center` to control
     * the center point for constructing the parametrizations
     * \f$\psi_{\text{x}}\f$. The samples are assumed to be normalized so
     * that the front is located at \f$(0,0)\f$ and the trailing (or blunt
     * edge) at \f$(1,0)\f$, or \f$(1,\ast)\f$, respectively. The
     * coordinate system is then shifted by the `psi center` point so that
     * the parameterizations \f$y = \hat\psi_{\text{up./lo.}}(x)\f$ expect
     * input in the range \f$\hat x\in[0,1-x_{\text{psi center}}]\f$. In the
     * same spirit \f$\hat\psi_{\text{fr.}}(\phi)\f$ expects input in the
     * range \f$\pi/2\le\phi\le3\pi/2\f$ and will return
     * \f$\psi_{\text{fr.}}(\pi) = -x_{\text{psi center}}\f$. Also a final
     * step the normalized parametrizations \f$\hat\psi\f$ are rescaled
     * with the runtime parameter `airfoil length` so that the resulting
     * airfoil has overall length `airfoil length` instead of 1.
     *
     * 2/ Meshing:
     *
     * TODO
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Airfoil : public Geometry<dim>
    {
    public:
      Airfoil(const std::string subsection)
          : Geometry<dim>("airfoil", subsection)
      {
        /* Parameters affecting parameterization: */

        airfoil_type_ = "NASA SC(2) 0714";
        this->add_parameter(
            "airfoil type", airfoil_type_, "airfoil type and serial number");

        airfoil_length_ = 2.;
        this->add_parameter("airfoil length",
                            airfoil_length_,
                            "length of airfoil (leading to trailing edge)");

        psi_samples_ = 100;
        this->add_parameter("psi samples",
                            psi_samples_,
                            "number of samples used for generating spline psi");

        psi_center_[0] = 0.05;
        this->add_parameter("psi center",
                            psi_center_,
                            "center position of airfoil for sampling psi");

        psi_ratio_ = 0.30;
        this->add_parameter(
            "psi ratio",
            psi_ratio_,
            "Scaling parameter for averages in curved nose region, can be "
            "adjusted by hand to equliabrate the size of faces at the nose "
            "part of the airfoil");


        airfoil_center_[0] = -.5;
        this->add_parameter("airfoil center",
                            airfoil_center_,
                            "position of airfoil center in the mesh");

        /* Parameters affecting mesh generation: */

        grading_ = 5.5;
        this->add_parameter(
            "grading exponent", grading_, "graded mesh: exponent");

        grading_epsilon_ = 0.02;
        this->add_parameter("grading epsilon",
                            grading_epsilon_,
                            "graded mesh: regularization parameter");

        height_ = 6.;
        this->add_parameter(
            "height", height_, "height of computational domain");

        width_ = 1.;
        this->add_parameter("width", width_, "width of computational domain");

        n_anisotropic_refinements_airfoil_ = 1;
        this->add_parameter(
            "anisotropic pre refinement airfoil",
            n_anisotropic_refinements_airfoil_,
            "number of anisotropic pre refinement steps for the airfoil");

        n_anisotropic_refinements_trailing_ = 3;
        this->add_parameter("anisotropic pre refinement trailing",
                            n_anisotropic_refinements_trailing_,
                            "number of anisotropic pre refinement steps for "
                            "the blunt trailing edge cell");

        subdivisions_z_ = 2;
        this->add_parameter("subdivisions z",
                            subdivisions_z_,
                            "number of subdivisions in z direction");
      }

      virtual void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final override
      {
        /*
         * Step 1: Create parametrization:
         *
         * Runtime parameters: airfoil_type_, airfoil_length_, psi_center_,
         * psi_samples_
         */

        const auto [x_upper, y_upper, x_lower, y_lower] = [&]() {
          if (airfoil_type_.rfind("NACA ", 0) == 0) {
            return naca_4digit_points(airfoil_type_.substr(5), psi_samples_);
          } else if (airfoil_type_.rfind("NASA SC(2) ", 0) == 0) {
            return nasa_sc2(airfoil_type_.substr(11));
          } else if (airfoil_type_.rfind("ONERA ", 0) == 0) {
            return onera(airfoil_type_.substr(6));
          }
          AssertThrow(false, ExcMessage("Unknown airfoil type"));
        }();

        const auto [psi_front, psi_upper, psi_lower] =
            create_psi(x_upper,
                       y_upper,
                       x_lower,
                       y_lower,
                       psi_center_[0],
                       psi_center_[1],
                       airfoil_length_);

        /*
         * Step 2: Create coarse mesh.
         *
         * Runtime parameters: airfoil_center_, height_,
         */

        /* The radius of the radial front part of the mesh: */
        const auto outer_radius = 0.5 * height_;

        /* by convention, psi_front(0.) returns the "back length" */
        const auto back_length = psi_front(0.);

        /* sharp trailing edge? */
        const bool sharp_trailing_edge =
            std::abs(psi_upper(back_length) - psi_lower(back_length)) < 1.0e-10;
        AssertThrow(
            sharp_trailing_edge ||
                std::abs(psi_upper(back_length) - psi_lower(back_length)) >
                    0.001 * back_length,
            dealii::ExcMessage("Blunt trailing edge has a width of less than "
                               "0.1% of the trailing airfoil length."));

        /* Front part: */
        dealii::Triangulation<2> tria_front;

        {
          std::vector<dealii::Point<2>> vertices{
              {-outer_radius, 0.0},                                       // 0
              {airfoil_center_[0] - psi_front(M_PI), airfoil_center_[1]}, // 1
              {-0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius},  // 2
              {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius},   // 3
              {0., airfoil_center_[1] + psi_lower(-airfoil_center_[0])},  // 4
              {airfoil_center_[0] + back_length,                          //
               airfoil_center_[1] + psi_lower(back_length)},              // 5
              {0., airfoil_center_[1] + psi_upper(-airfoil_center_[0])},  // 6
              {-0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius},   // 7
              {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius},    // 8
          };

          std::vector<dealii::CellData<2>> cells(4);
          cells[0].vertices = {2, 3, 4, 5};
          cells[1].vertices = {0, 2, 1, 4};
          cells[2].vertices = {7, 0, 6, 1};
          if (sharp_trailing_edge) {
            cells[3].vertices = {8, 7, 5, 6};
          } else {
            vertices.push_back({airfoil_center_[0] + back_length,
                                airfoil_center_[1] + psi_upper(back_length)});
            cells[3].vertices = {8, 7, 9, 6};
          }

          tria_front.create_triangulation(
              vertices, cells, dealii::SubCellData());
        }

        /* Back part: */
        dealii::Triangulation<2> tria_back;

        if (sharp_trailing_edge) {
          /* Back part for sharp trailing edge: */

          const std::vector<dealii::Point<2>> vertices{
              {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 0
              {airfoil_center_[0] + back_length,
               airfoil_center_[1] + psi_lower(back_length)},           // 1
              {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius}, // 2
              {outer_radius, -0.5 * outer_radius},                     // 3
              {outer_radius, 0.0},                                     // 4
              {outer_radius, 0.5 * outer_radius},                      // 5
          };

          std::vector<dealii::CellData<2>> cells(2);
          cells[0].vertices = {0, 3, 1, 4};
          cells[1].vertices = {1, 4, 2, 5};

          tria_back.create_triangulation(
              vertices, cells, dealii::SubCellData());

        } else {
          /* Back part for blunt trailing edge: */

          /* Good width for the anisotropically refined center trailing cell: */
          double trailing_height =
              0.5 / (0.5 + std::pow(2., n_anisotropic_refinements_airfoil_)) *
              0.5 * outer_radius;

          const std::vector<dealii::Point<2>> vertices{
              {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 0
              {airfoil_center_[0] + back_length,
               airfoil_center_[1] + psi_lower(back_length)}, // 1
              {airfoil_center_[0] + back_length,
               airfoil_center_[1] + psi_upper(back_length)},           // 2
              {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius}, // 3
              {outer_radius, -0.5 * outer_radius},                     // 4
              {outer_radius, -trailing_height},                        // 5
              {outer_radius, trailing_height},                         // 6
              {outer_radius, 0.5 * outer_radius},                      // 7
          };

          std::vector<dealii::CellData<2>> cells(3);
          cells[0].vertices = {0, 4, 1, 5};
          cells[1].vertices = {1, 5, 2, 6};
          cells[2].vertices = {2, 6, 3, 7};

          tria_back.create_triangulation(
              vertices, cells, dealii::SubCellData());
        }

        dealii::Triangulation<2> coarse_triangulation;
        GridGenerator::merge_triangulations(
            {&tria_front, &tria_back}, coarse_triangulation, 1.e-12, true);

        /*
         * Step 3: Set manifold IDs and attach manifolds to preliminary
         * coarse triangulation:
         *
         * Curvature for boundaries:
         *   1 -> upper airfoil (inner boundary)
         *   2 -> lower airfoil (inner boundary)
         *   3 -> spherical manifold (outer boundary)
         *
         * Transfinite interpolation with grading on coarse cells:
         *
         *  10 -> bottom center cell
         *  11 -> bottom front cell
         *  12 -> top front cell
         *  13 -> top center cell
         *  14 -> bottom trailing cell
         *  15 -> top trailing cell (sharp), center trailing cell (blunt)
         *  16 -> top trailing cell (blunt)
         */

        /* Colorize boundary faces and add curvature information: */

        for (auto cell : coarse_triangulation.active_cell_iterators()) {
          for (auto f : dealii::GeometryInfo<2>::face_indices()) {
            const auto face = cell->face(f);
            if (!face->at_boundary())
              continue;

            bool airfoil = true;
            bool spherical_boundary = true;
            for (const auto v : dealii::GeometryInfo<1>::vertex_indices())
              if (std::abs((face->vertex(v)).norm() - outer_radius) < 1.0e-10)
                airfoil = false;
              else
                spherical_boundary = false;

            if (spherical_boundary) {
              face->set_manifold_id(3);
            } else if (airfoil) {
              if (face->center()[0] <
                  airfoil_center_[0] + back_length - 1.e-6) {
                if (face->center()[1] >= airfoil_center_[1]) {
                  face->set_manifold_id(1);
                } else {
                  face->set_manifold_id(2);
                }
              }
            }
          } /* f */
        }   /* cell */

        Manifolds::AirfoilManifold airfoil_manifold_upper{
            airfoil_center_, psi_front, psi_upper, psi_lower, true, psi_ratio_};
        coarse_triangulation.set_manifold(1, airfoil_manifold_upper);

        Manifolds::AirfoilManifold airfoil_manifold_lower{airfoil_center_,
                                                          psi_front,
                                                          psi_upper,
                                                          psi_lower,
                                                          false,
                                                          psi_ratio_};
        coarse_triangulation.set_manifold(2, airfoil_manifold_lower);

        dealii::SphericalManifold<2> spherical_manifold;
        coarse_triangulation.set_manifold(3, spherical_manifold);

        /* Create transfinite interpolation manifolds: */

        Assert(!sharp_trailing_edge || (coarse_triangulation.n_cells() == 6),
               dealii::ExcInternalError());
        Assert(sharp_trailing_edge || (coarse_triangulation.n_cells() == 7),
               dealii::ExcInternalError());

        std::vector<std::unique_ptr<dealii::Manifold<dim, dim>>> manifolds;
        manifolds.resize(sharp_trailing_edge ? 6 : 7);

        /* FIXME: Remove workaround - mark cells as off limit: */
        std::next(coarse_triangulation.begin_active(), 4)->set_material_id(42);
        std::next(coarse_triangulation.begin_active(),
                  sharp_trailing_edge ? 5 : 6)
            ->set_material_id(42);

        for (auto i : {0, 1, 2, 3, 5}) {
          const auto index = 10 + i;

          dealii::Point<2> center;
          dealii::Tensor<1, 2> direction;
          if (i < 4) {
            /* cells: bottom center, bottom front, top front, top center */
            direction[1] = 1.;
          } else {
            Assert(i == 5, dealii::ExcInternalError());
            /* cell: center trailing (blunt) */
            center[0] = 1.;
            direction[0] = -1.;
          }

          Manifolds::GradingManifold<dim> grading{center,
                                                  direction,
                                                  grading_,
                                                  (i == 5 ? 0.1 : 0.0) +
                                                      grading_epsilon_};

          auto transfinite =
              std::make_unique<ryujin::TransfiniteInterpolationManifold<2>>();
          transfinite->initialize(coarse_triangulation, grading);

          coarse_triangulation.set_manifold(index, *transfinite);
          manifolds[i] = std::move(transfinite);
        }

        /* Remove erroneous manifold: */
        if (sharp_trailing_edge)
          coarse_triangulation.reset_manifold(5);

        /*
         * Use transfinite interpolation manifold for all geometric objects
         * and remove unneeded manifolds:
         */

        coarse_triangulation.reset_manifold(1);
        coarse_triangulation.reset_manifold(2);
        coarse_triangulation.reset_manifold(3);
        for (unsigned int i = 0; i < (sharp_trailing_edge ? 6 : 7); ++i) {
          const auto &cell = std::next(coarse_triangulation.begin_active(), i);
          const auto index = 10 + i;
          if (i == 4 || i == (sharp_trailing_edge ? 5 : 6))
            cell->set_manifold_id(index);
          else
            cell->set_all_manifold_ids(index);
        }

        /*
         * Attach separate transfinite interpolation manifolds (without a
         * grading) to the top and bottom trailing cells:
         */

        /* FIXME: Remove workaround - mark cells as off limit: */
        for (auto cell : coarse_triangulation.active_cell_iterators())
          cell->set_material_id(42);
        std::next(coarse_triangulation.begin_active(), 4)->set_material_id(0);
        std::next(coarse_triangulation.begin_active(),
                  sharp_trailing_edge ? 5 : 6)
            ->set_material_id(0);

        for (auto i : {4, sharp_trailing_edge ? 5 : 6}) {
          const auto index = 10 + i;
          auto transfinite =
              std::make_unique<ryujin::TransfiniteInterpolationManifold<2>>();
          transfinite->initialize(coarse_triangulation);
          coarse_triangulation.set_manifold(index, *transfinite);
          manifolds[i] = std::move(transfinite);
        }

        /*
         * Step 4: Anisotropic pre refinement.
         *
         * Runtime parameters: n_anisotropic_refinements_airfoil_,
         * n_anisotropic_refinements_trailing_
         */

        const double trailing_height =
            0.5 / (0.5 + std::pow(2., n_anisotropic_refinements_airfoil_)) *
            0.5 * outer_radius;

        /* Mark critical cells with a temporary material id: */
        for (auto cell : coarse_triangulation.active_cell_iterators()) {

          /* in case of a blunt edge we refine the trailing cell: */
          if (!sharp_trailing_edge)
            if (cell->center()[0] > airfoil_center_[0] + back_length &&
                std::abs(cell->center()[1]) <= trailing_height)
              cell->set_material_id(2);

          /*
           * Let us also insert additional radials on the upper (back) and
           * lower (back) side of the airfoil
           */
          if (cell->center()[0] > airfoil_center_[0] &&
              cell->center()[0] <
                  0.5 * (airfoil_center_[0] + back_length + 0.5 * outer_radius))
            cell->set_material_id(3);
        }

        /* Upper and lower cell on airfoil: */
        for (unsigned int i = 0; i < n_anisotropic_refinements_airfoil_; ++i) {
          for (auto cell : coarse_triangulation.active_cell_iterators())
            if (cell->material_id() == 3)
              cell->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));

          coarse_triangulation.execute_coarsening_and_refinement();
        }

        /* Tailing cell: */
        for (unsigned int i = 0; i < n_anisotropic_refinements_trailing_; ++i) {
          for (auto cell : coarse_triangulation.active_cell_iterators())
            if (cell->material_id() == 2)
              cell->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));
            else
              cell->set_refine_flag();
          coarse_triangulation.execute_coarsening_and_refinement();
        }

        /*
         * Step 5: Flatten triangulation, create distributed coarse
         * triangulation, and reattach manifolds
         *
         * Runtime parameters: width_, subdivisions_z_ (for dim == 3)
         */

        dealii::Triangulation<2> tria3;
        tria3.set_mesh_smoothing(triangulation.get_mesh_smoothing());
        GridGenerator::flatten_triangulation(coarse_triangulation, tria3);

        if constexpr (dim == 2) {
          triangulation.copy_triangulation(tria3);

          /*
           * Somewhere during flattening the triangulation and copying we
           * lost all manifold ids on faces:
           */
          for (auto &cell : triangulation.active_cell_iterators()) {
            const auto id = cell->manifold_id();
            cell->set_all_manifold_ids(id);
          }

          unsigned int index = 10;
          for (const auto &manifold : manifolds)
            triangulation.set_manifold(index++, *manifold);

        } else {

          /* extrude mesh: */
          dealii::Triangulation<3, 3> tria4;
          tria4.set_mesh_smoothing(triangulation.get_mesh_smoothing());
          GridGenerator::extrude_triangulation(
              tria3, subdivisions_z_, width_, tria4);
          triangulation.copy_triangulation(tria4);

          AssertThrow(false,
                      dealii::ExcMessage(
                          "manifold ids for 3D airfoil not implemented"));
        }

        /* Set boundary ids: */

        for (auto cell : triangulation.active_cell_iterators()) {
          for (auto f : dealii::GeometryInfo<dim>::face_indices()) {
            const auto face = cell->face(f);

            /* Handle boundary faces: */
            if (!face->at_boundary())
              continue;

            bool airfoil = true;
            bool spherical_boundary = true;

            const auto &indices =
                dealii::GeometryInfo<dim - 1>::vertex_indices();
            for (const auto v : indices) {
              const auto vert = face->vertex(v);
              const auto radius_sqr = vert[0] * vert[0] + vert[1] * vert[1];
              if (radius_sqr >= outer_radius * outer_radius - 1.0e-10 ||
                  vert[0] > airfoil_center_[0] + 1.001 * back_length)
                airfoil = false;
              else
                spherical_boundary = false;
            }

            if (spherical_boundary) {
              face->set_boundary_id(Boundary::dynamic);
            } else if (airfoil) {
              face->set_boundary_id(Boundary::no_slip);
            } else {
              face->set_boundary_id(Boundary::periodic);
            }
          }
        }
      }

    private:
      dealii::Point<2> airfoil_center_;
      double airfoil_length_;
      std::string airfoil_type_;
      dealii::Point<2> psi_center_;
      double psi_ratio_;
      unsigned int psi_samples_;
      double height_;
      double width_;
      double grading_;
      double grading_epsilon_;
      unsigned int n_anisotropic_refinements_airfoil_;
      unsigned int n_anisotropic_refinements_trailing_;
      unsigned int subdivisions_z_;
    };

  } /* namespace Geometries */

} /* namespace ryujin */

#endif /* GRID_AIRFOIL_H */
