//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "geometry_common_includes.h"

#include "cubic_spline.h"
#include "transfinite_interpolation.template.h"

namespace ryujin
{
  namespace
  {
    /**
     * A small helper function to assign values to either a c-style
     * array (how vertices are stored in deal.II version 9.2 and
     * earlier), or a std::array (how vertices are stored in deal.II
     * version 9.3 onwards).
     */
    template <typename T, typename T2>
    inline void assign(T &array,
                       const std::initializer_list<T2> &initializer_list)
    {
      /* accomodate for a possible c-style array... */
      Assert(std::size(array) == std::size(initializer_list),
             dealii::ExcMessage(
                 "size of initializer list and array does not match"));
      std::copy(
          initializer_list.begin(), initializer_list.end(), std::begin(array));
    }
  } // namespace

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

      dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final
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

      dealii::Point<dim>
      push_forward(const dealii::Point<dim> &point) const final
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

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final
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
      Point<dim>
      get_new_point(const ArrayView<const Point<dim>> &surrounding_points,
                    const ArrayView<const double> &weights) const override
      {
        if (weights[0] > 1.0)
          return surrounding_points[0];

        if (weights[1] > 1.0)
          return surrounding_points[1];

        return dealii::ChartManifold<dim>::get_new_point(surrounding_points,
                                                         weights);
      }

      dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final
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

      dealii::Point<dim>
      push_forward(const dealii::Point<dim> &chart_point) const final
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

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final
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

      std::unique_ptr<Manifold<dim>> clone() const override
      {
        return std::make_unique<ExtrudedManifold<dim>>(*manifold);
      }

      Point<dim>
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
     * The NASA SC(2)-0714 airfoil as described in the technical report
     *   NASA Technical Paper 2969, 1990
     *   NASA Supercritical Airfoils, A Matrix of Family-Related Airfoils
     *   Charles D. Harris
     *
     * Table XXI. page 36 (pdf page 38), found here
     * https://ntrs.nasa.gov/api/citations/19890008197/downloads/19890008197.pdf
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
     * Datapoints for the Onera OAT15a airfoil.
     *
     * This data is annoyingly hard to get, see the discussion here
     * https://www.cfd-online.com/Forums/main/114894-oat15a-geometry.html
     * that mentions the following thesis (Appendix page 152) as a
     * reference https://lume.ufrgs.br/handle/10183/28925
     */
    std::array<std::vector<double>, 4> onera(const std::string &serial_number)
    {
      if (serial_number == "OAT15a") {
        std::vector<double> x_upper{
            0.,          2.95888e-05, 0.000117865, 0.000263239, 0.000464245,
            0.000719821, 0.00103013,  0.00139605,  0.00181884,  0.00230024,
            0.00284243,  0.00344764,  0.00411805,  0.00485595,  0.00566349,
            0.00654241,  0.00749421,  0.0085197,   0.0096197,   0.0107945,
            0.0120442,   0.013369,    0.0147688,   0.0162438,   0.0177939,
            0.0194194,   0.0211207,   0.0228982,   0.0247526,   0.0266845,
            0.028695,    0.0307852,   0.0329562,   0.0352094,   0.0375463,
            0.0399687,   0.0424782,   0.0450769,   0.0477668,   0.05055,
            0.0534291,   0.0564063,   0.0594842,   0.0626655,   0.0659531,
            0.0693498,   0.0728588,   0.0764831,   0.0802261,   0.0840914,
            0.0880824,   0.0922027,   0.0964564,   0.100847,    0.10538,
            0.110058,    0.114885,    0.119868,    0.125009,    0.130314,
            0.135789,    0.14139,     0.147074,    0.152839,    0.158682,
            0.164603,    0.170599,    0.17667,     0.182814,    0.189028,
            0.195312,    0.201665,    0.208083,    0.214567,    0.221115,
            0.227724,    0.234394,    0.241123,    0.24791,     0.254753,
            0.26165,     0.2686,      0.275601,    0.282653,    0.289753,
            0.2969,      0.304093,    0.311329,    0.318609,    0.325929,
            0.333289,    0.340686,    0.348121,    0.35559,     0.363093,
            0.370629,    0.378194,    0.385789,    0.393412,    0.40106,
            0.408734,    0.41643,     0.424148,    0.431886,    0.439643,
            0.447417,    0.455207,    0.46301,     0.470827,    0.478654,
            0.486491,    0.494337,    0.502189,    0.510046,    0.517907,
            0.525769,    0.533632,    0.541494,    0.549354,    0.557209,
            0.565059,    0.572902,    0.580736,    0.588559,    0.596371,
            0.60417,     0.611953,    0.61972,     0.627469,    0.635198,
            0.642906,    0.65059,     0.658251,    0.665886,    0.673493,
            0.68107,     0.688617,    0.69613,     0.70361,     0.711054,
            0.71846,     0.725827,    0.733154,    0.740438,    0.747679,
            0.754875,    0.762025,    0.769127,    0.776181,    0.783185,
            0.790139,    0.79704,     0.80389,     0.810685,    0.817426,
            0.82411,     0.830738,    0.837307,    0.843817,    0.850265,
            0.856652,    0.862974,    0.869233,    0.87538,     0.881373,
            0.887216,    0.892913,    0.898467,    0.903883,    0.909163,
            0.914311,    0.91933,     0.924224,    0.928996,    0.933648,
            0.938183,    0.942606,    0.946917,    0.95112,     0.955217,
            0.959212,    0.963107,    0.966904,    0.970605,    0.974213,
            0.977731,    0.98116,     0.984503,    0.987762,    0.990939,
            0.994036,    0.997056,    1.};

        std::vector<double> y_upper{
            0.,         0.000899353, 0.0018231,  0.00276894, 0.00373508,
            0.00472011, 0.0057226,   0.00674103, 0.0077738,  0.00881906,
            0.00987467, 0.0109383,   0.0120074,  0.0130793,  0.0141511,
            0.01522,    0.0162832,   0.0173387,  0.0183841,  0.0194179,
            0.0204389,  0.021446,    0.0224386,  0.0234164,  0.0243794,
            0.0253276,  0.0262612,   0.0271805,  0.028086,   0.0289783,
            0.0298578,  0.0307252,   0.0315811,  0.0324262,  0.0332611,
            0.0340861,  0.0349022,   0.0357098,  0.0365095,  0.0373016,
            0.0380867,  0.0388652,   0.0396375,  0.0404039,  0.041165,
            0.0419211,  0.0426722,   0.0434189,  0.0441614,  0.0448995,
            0.0456336,  0.0463636,   0.0470894,  0.047811,   0.0485286,
            0.0492423,  0.0499518,   0.0506574,  0.0513591,  0.0520569,
            0.0527506,  0.0534343,   0.0541023,  0.054755,   0.0553921,
            0.0560138,  0.05662,     0.0572108,  0.0577861,  0.0583462,
            0.0588909,  0.0594202,   0.0599341,  0.0604325,  0.0609153,
            0.0613826,  0.0618341,   0.06227,    0.06269,    0.0630941,
            0.0634823,  0.0638544,   0.0642103,  0.06455,    0.0648734,
            0.0651806,  0.0654713,   0.0657454,  0.0660031,  0.0662442,
            0.0664685,  0.066676,    0.0668664,  0.0670398,  0.067196,
            0.0673349,  0.0674562,   0.0675598,  0.0676456,  0.0677134,
            0.0677629,  0.067794,    0.0678065,  0.0678,     0.0677743,
            0.0677293,  0.0676646,   0.0675798,  0.0674748,  0.0673492,
            0.0672027,  0.0670349,   0.0668456,  0.0666344,  0.066401,
            0.066145,   0.0658661,   0.065564,   0.0652385,  0.064889,
            0.0645151,  0.0641169,   0.0636938,  0.0632454,  0.0627715,
            0.0622718,  0.061746,    0.0611937,  0.0606145,  0.0600083,
            0.0593747,  0.0587136,   0.0580244,  0.0573069,  0.0565607,
            0.0557853,  0.0549807,   0.0541461,  0.0532814,  0.0523863,
            0.0514606,  0.0505046,   0.0495188,  0.0485042,  0.047462,
            0.0463943,  0.0453031,   0.0441914,  0.0430618,  0.0419174,
            0.0407612,  0.0395961,   0.038425,   0.0372503,  0.0360742,
            0.034899,   0.0337262,   0.0325572,  0.0313935,  0.030236,
            0.029086,   0.0279442,   0.0268114,  0.0256966,  0.0246079,
            0.023545,   0.0225073,   0.0214947,  0.0205065,  0.0195422,
            0.0186011,  0.0176826,   0.0167863,  0.0159112,  0.0150567,
            0.0142221,  0.0134066,   0.0126095,  0.0118301,  0.0110678,
            0.0103219,  0.00959177,  0.00887695, 0.00817697, 0.00749135,
            0.00681977, 0.0061621,   0.00551806, 0.00488739, 0.00427007,
            0.00366612, 0.00307588,  0.0024997};

        std::vector<double> x_lower{
            0.,          3.22311e-05, 0.000136327, 0.000324365, 0.000606007,
            0.000986654, 0.00146626,  0.00204126,  0.00270571,  0.00345312,
            0.00427708,  0.00517234,  0.00613428,  0.00715943,  0.00824517,
            0.00938973,  0.0105917,   0.0118504,   0.0131652,   0.0145362,
            0.0159635,   0.0174476,   0.0189891,   0.0205889,   0.0222479,
            0.0239675,   0.0257488,   0.0275934,   0.0295028,   0.031479,
            0.0335237,   0.035639,    0.0378271,   0.04009,     0.0424303,
            0.0448502,   0.0473523,   0.049939,    0.0526132,   0.0553774,
            0.0582343,   0.0611868,   0.0642377,   0.0673899,   0.0706465,
            0.0740105,   0.077485,    0.0810733,   0.0847788,   0.088605,
            0.0925553,   0.0966336,   0.100844,    0.10519,     0.109675,
            0.114305,    0.119084,    0.124016,    0.129106,    0.134358,
            0.139779,    0.145328,    0.150961,    0.156676,    0.162473,
            0.168349,    0.174303,    0.180333,    0.186437,    0.192615,
            0.198863,    0.205181,    0.211568,    0.21802,     0.224537,
            0.231117,    0.237759,    0.24446,     0.251219,    0.258035,
            0.264905,    0.271829,    0.278804,    0.285828,    0.292901,
            0.30002,     0.307184,    0.314391,    0.321639,    0.328927,
            0.336253,    0.343615,    0.351011,    0.358441,    0.365902,
            0.373392,    0.38091,     0.388455,    0.396024,    0.403616,
            0.41123,     0.418864,    0.426517,    0.434187,    0.441871,
            0.44957,     0.457282,    0.465005,    0.472737,    0.480477,
            0.488225,    0.495977,    0.503733,    0.511493,    0.519252,
            0.527012,    0.53477,     0.542525,    0.550276,    0.55802,
            0.565759,    0.573488,    0.581208,    0.588918,    0.596615,
            0.604298,    0.611967,    0.61962,     0.627257,    0.634874,
            0.642473,    0.65005,     0.657605,    0.665137,    0.672645,
            0.680128,    0.687584,    0.695012,    0.702411,    0.70978,
            0.717118,    0.724424,    0.731697,    0.738935,    0.746137,
            0.753303,    0.76043,     0.767518,    0.774565,    0.78157,
            0.788531,    0.795447,    0.802316,    0.809136,    0.815905,
            0.822623,    0.829286,    0.835893,    0.842441,    0.84893,
            0.855357,    0.86172,     0.868018,    0.874204,    0.880239,
            0.886123,    0.891863,    0.89746,     0.902919,    0.908242,
            0.913433,    0.918495,    0.923431,    0.928244,    0.932938,
            0.937515,    0.941978,    0.94633,     0.950574,    0.954712,
            0.958747,    0.962681,    0.966518,    0.970259,    0.973907,
            0.977464,    0.980932,    0.984314,    0.987612,    0.990827,
            0.993963,    0.997019,    1.};

        std::vector<double> y_lower{
            0.,           -0.000899234, -0.00182108,  -0.00275889,  -0.00370397,
            -0.00464681,  -0.00557909,  -0.00649452,  -0.0073895,   -0.00826265,
            -0.00911444,  -0.00994611,  -0.0107598,   -0.0115578,   -0.0123422,
            -0.0131152,   -0.0138788,   -0.0146348,   -0.0153851,   -0.0161313,
            -0.0168749,   -0.0176173,   -0.0183597,   -0.0191032,   -0.0198489,
            -0.0205974,   -0.0213498,   -0.0221064,   -0.0228678,   -0.0236341,
            -0.0244053,   -0.0251817,   -0.0259627,   -0.0267482,   -0.0275378,
            -0.0283309,   -0.029127,    -0.0299255,   -0.0307258,   -0.0315273,
            -0.0323295,   -0.0331322,   -0.0339346,   -0.0347366,   -0.035538,
            -0.0363383,   -0.0371378,   -0.037936,    -0.0387331,   -0.0395288,
            -0.0403228,   -0.0411152,   -0.0419054,   -0.0426933,   -0.0434778,
            -0.0442587,   -0.0450347,   -0.045805,    -0.0465681,   -0.0473225,
            -0.0480666,   -0.0487929,   -0.0494941,   -0.0501695,   -0.0508179,
            -0.0514387,   -0.052031,    -0.0525942,   -0.0531277,   -0.0536309,
            -0.0541035,   -0.054545,    -0.0549547,   -0.0553323,   -0.0556773,
            -0.0559891,   -0.056267,    -0.0565107,   -0.0567191,   -0.0568918,
            -0.0570281,   -0.0571272,   -0.0571886,   -0.0572116,   -0.0571958,
            -0.0571405,   -0.0570453,   -0.0569098,   -0.0567339,   -0.0565171,
            -0.0562594,   -0.0559608,   -0.055621,    -0.0552405,   -0.0548193,
            -0.0543577,   -0.0538564,   -0.0533157,   -0.0527365,   -0.0521197,
            -0.0514661,   -0.0507767,   -0.0500525,   -0.0492947,   -0.0485041,
            -0.0476819,   -0.0468294,   -0.0459474,   -0.0450371,   -0.0440995,
            -0.0431356,   -0.0421465,   -0.0411333,   -0.0400971,   -0.039039,
            -0.0379601,   -0.0368617,   -0.0357451,   -0.0346114,   -0.0334621,
            -0.0322983,   -0.0311217,   -0.0299336,   -0.0287354,   -0.0275285,
            -0.0263145,   -0.025095,    -0.0238717,   -0.0226459,   -0.0214194,
            -0.020194,    -0.0189714,   -0.0177534,   -0.016542,    -0.015339,
            -0.0141466,   -0.0129671,   -0.0118026,   -0.0106558,   -0.00952898,
            -0.00842491,  -0.00734634,  -0.00629613,  -0.00527714,  -0.0042922,
            -0.00334424,  -0.00243619,  -0.00157086,  -0.000750868, 2.13187e-05,
            0.000743489,  0.00141373,   0.00203031,   0.00259174,   0.00309711,
            0.00354599,   0.00393819,   0.00427381,   0.00455352,   0.00477839,
            0.00494994,   0.00506989,   0.00514012,   0.0051629,    0.00514197,
            0.0050815,    0.0049854,    0.00485738,   0.0047009,    0.00451919,
            0.0043152,    0.00409157,   0.00385071,   0.00359475,   0.00332561,
            0.00304503,   0.00275452,   0.00245546,   0.00214904,   0.00183633,
            0.00151827,   0.00119574,   0.00086945,   0.00054002,   0.000208013,
            -0.000126036, -0.000461602, -0.000798383, -0.0011363,   -0.00147529,
            -0.00181444,  -0.00215832,  -0.0024967};

        return {{x_upper, y_upper, x_lower, y_lower}};

      } else {

        AssertThrow(false, dealii::ExcMessage("Invalid ONERA serial number"));
      }
    }

    /**
     * (nlr1t-il) NLR-1T AIRFOIL
     *
     * Bell/NASA/NLR NLR-1T rotorcraft airfoil
     * Max thickness 8.7% at 38.3% chord.
     * Max camber 1.3% at 22.3% chord
     *
     * Input by Taylor Boylan
     */
    std::array<std::vector<double>, 4> bell(const std::string &serial_number)
    {
      if (serial_number == "NLR-1T") {
        std::vector<double> x_upper{
            .0,     .00259, .00974, .02185, .03796, .05675, .07753,
            .09845, .12341, .15412, .18767, .22313, .26054, .29979,
            .34064, .38269, .42528, .46849, .51162, .55383, .59596,
            .63728, .67732, .71079, .73905, .76946, .80263, .84055,
            .87846, .90845, .93589, .96199, 1.};

        std::vector<double> y_upper{
            .0,     .00704, .01524, .02296, .02972, .03588, .04098,
            .04469, .04741, .04986, .05188, .05345, .05459, .05531,
            .05565, .0556,  .05518, .05438, .05323, .05175, .04992,
            .04774, .04524, .04291, .04017, .03644, .0314,  .02533,
            .01901, .01421, .0102,  .00651, .00104};

        std::vector<double> x_lower{
            .0,     .00259, .00974, .02185, .03796, .05675, .07753,
            .09845, .12341, .15412, .18767, .22313, .26054, .29979,
            .34064, .38269, .42528, .46849, .51162, .55383, .59596,
            .63728, .67732, .71079, .73905, .76946, .80263, .84055,
            .87846, .90845, .93589, .96199, 1.};
        std::vector<double> y_lower{
            .0,      -.00512, -.00867, -.0118,  -.01465, -.01713, -.01929,
            -.02112, -.02299, -.02494, -.02671, -.02821, -.02944, -.0304,
            -.03104, -.03142, -.0315,  -.03132, -.0308,  -.02992, -.02867,
            -.02734, -.0258,  -.02432, -.02305, -.02164, -.01996, -.01794,
            -.01571, -.01364, -.01087, -.00711, -.00104};

        return {{x_upper, y_upper, x_lower, y_lower}};

      } else {

        AssertThrow(false, dealii::ExcMessage("Invalid BELL serial number"));
      }
    }


    /**
     * @todo Documentation
     */
    std::array<std::function<double(const double)>, 3>
    create_psi(const std::vector<double> &x_upper [[maybe_unused]],
               const std::vector<double> &y_upper [[maybe_unused]],
               const std::vector<double> &x_lower [[maybe_unused]],
               const std::vector<double> &y_lower [[maybe_unused]],
               const double x_center [[maybe_unused]],
               const double y_center [[maybe_unused]],
               const double scaling [[maybe_unused]] = 1.)
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
      AssertThrow(
          false,
          dealii::ExcNotImplemented("Airfoil grid needs deal.II with GSL"));
      return {};
#endif
    }


  } // namespace


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

        grading_epsilon_trailing_ = 0.01;
        this->add_parameter(
            "grading epsilon trailing",
            grading_epsilon_trailing_,
            "graded mesh: regularization parameter for trailing cells");

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

      void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final
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
          } else if (airfoil_type_.rfind("BELL ", 0) == 0) {
            return bell(airfoil_type_.substr(5));
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
          assign(cells[0].vertices, {2, 3, 4, 5});
          assign(cells[1].vertices, {0, 2, 1, 4});
          assign(cells[2].vertices, {7, 0, 6, 1});
          if (sharp_trailing_edge) {
            assign(cells[3].vertices, {8, 7, 5, 6});
          } else {
            vertices.push_back({airfoil_center_[0] + back_length,
                                airfoil_center_[1] + psi_upper(back_length)});
            assign(cells[3].vertices, {8, 7, 9, 6});
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
          assign(cells[0].vertices, {0, 3, 1, 4});
          assign(cells[1].vertices, {1, 4, 2, 5});

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
          assign(cells[0].vertices, {0, 4, 1, 5});
          assign(cells[1].vertices, {1, 5, 2, 6});
          assign(cells[2].vertices, {2, 6, 3, 7});

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

        /*
         * Colorize boundary faces and add manifolds for curvature
         * information on boundaries:
         */

        for (auto cell : coarse_triangulation.active_cell_iterators()) {
          for (auto f : cell->reference_cell().face_indices()) {
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

        /*
         * Create transfinite interpolation manifolds for the interior of
         * the 2D coarse cells:
         */

        Assert(!sharp_trailing_edge || (coarse_triangulation.n_cells() == 6),
               dealii::ExcInternalError());
        Assert(sharp_trailing_edge || (coarse_triangulation.n_cells() == 7),
               dealii::ExcInternalError());

        std::vector<std::unique_ptr<dealii::Manifold<2, 2>>> manifolds;
        manifolds.resize(sharp_trailing_edge ? 6 : 7);

        /* FIXME: Remove workaround - mark cells as off limit: */
        // WORKAROUND
        const auto first_cell = coarse_triangulation.begin_active();
        std::next(first_cell, 4)->set_material_id(42);
        std::next(first_cell, sharp_trailing_edge ? 5 : 6)->set_material_id(42);
        // end WORKAROUND

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

          Manifolds::GradingManifold<2> grading{
              center,
              direction,
              grading_,
              i == 5 ? grading_epsilon_trailing_ : grading_epsilon_};

          auto transfinite =
              std::make_unique<TransfiniteInterpolationManifold<2>>();
          transfinite->initialize(coarse_triangulation, grading);

          coarse_triangulation.set_manifold(index, *transfinite);
          manifolds[i] = std::move(transfinite);
        }

        /* Remove erroneous manifold: */
        if (sharp_trailing_edge)
          coarse_triangulation.reset_manifold(5);

        /*
         * Remove unneeded manifolds now. Our custom
         * TransfiniteInterpolationManifolds did copy all necessary
         * geometry information from the coarse grid already. The boundary
         * manifolds are thus not needed any more.
         */

        coarse_triangulation.reset_manifold(1);
        coarse_triangulation.reset_manifold(2);
        coarse_triangulation.reset_manifold(3);

        /* We can set the final sequence of manifold ids: */
        for (unsigned int i = 0; i < (sharp_trailing_edge ? 6 : 7); ++i) {
          const auto &cell = std::next(coarse_triangulation.begin_active(), i);
          const auto index = 10 + i;
          if (i == 4 || i == (sharp_trailing_edge ? 5 : 6)) {
            cell->set_manifold_id(index);
          } else {
            cell->set_all_manifold_ids(index);
          }
        }

        /*
         * Attach separate transfinite interpolation manifolds (without a
         * grading) to the top and bottom trailing cells:
         */

        /* FIXME: Remove workaround - mark cells as off limit: */
        // WORKAROUND
        for (auto cell : coarse_triangulation.active_cell_iterators())
          cell->set_material_id(42);
        // const auto first_cell = coarse_triangulation.begin_active();
        std::next(first_cell, 4)->set_material_id(0);
        std::next(first_cell, sharp_trailing_edge ? 5 : 6)->set_material_id(0);
        // end WORKAROUND

        for (auto i : {4, sharp_trailing_edge ? 5 : 6}) {
          const auto index = 10 + i;
          auto transfinite =
              std::make_unique<ryujin::TransfiniteInterpolationManifold<2>>();
          transfinite->initialize(coarse_triangulation);
          coarse_triangulation.set_manifold(index, *transfinite);
          manifolds[i] = std::move(transfinite);
        }

        /*
         * For good measure, also set material ids. We will need those
         * in a minute to reconstruct material ids...
         */

        for (unsigned int i = 0; i < (sharp_trailing_edge ? 6 : 7); ++i) {
          const auto &cell = std::next(coarse_triangulation.begin_active(), i);
          const auto index = 10 + i;
          cell->set_material_id(index);
        }

        /*
         * Step 4: Anisotropic pre refinement.
         *
         * Runtime parameters: n_anisotropic_refinements_airfoil_,
         * n_anisotropic_refinements_trailing_
         */

        /* Additional radials in upper and lower cell on airfoil (material
         * ids 10 and 13): */
        for (unsigned int i = 0; i < n_anisotropic_refinements_airfoil_; ++i) {
          for (auto cell : coarse_triangulation.active_cell_iterators()) {
            const auto id = cell->material_id();
            if (id == 10 || id == 13)
              cell->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));
          }

          coarse_triangulation.execute_coarsening_and_refinement();
        }

        /* Anisotropic refinement into trailing cell (material id 15): */
        if (!sharp_trailing_edge)
          for (unsigned i = 0; i < n_anisotropic_refinements_trailing_; ++i) {
            for (auto cell : coarse_triangulation.active_cell_iterators())
              if (cell->material_id() == 15)
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

        if constexpr (dim == 1) {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();

        } else if constexpr (dim == 2) {
          /* Flatten manifold: */
          dealii::Triangulation<2> tria3;
          tria3.set_mesh_smoothing(triangulation.get_mesh_smoothing());
          GridGenerator::flatten_triangulation(coarse_triangulation, tria3);

          triangulation.copy_triangulation(tria3);

        } else {
          static_assert(dim == 3);

          /* Flatten manifold: */
          dealii::Triangulation<2> tria3;
          GridGenerator::flatten_triangulation(coarse_triangulation, tria3);

          /* extrude mesh: */
          dealii::Triangulation<3, 3> tria4;
          tria4.set_mesh_smoothing(triangulation.get_mesh_smoothing());
          GridGenerator::extrude_triangulation(
              tria3, subdivisions_z_, width_, tria4);

          triangulation.copy_triangulation(tria4);
        }

        /*
         * Somewhere during flattening the triangulation, extruding and
         * copying, all manifold ids got lost. Reconstruct manifold IDs
         * from the material ids we set earlier:
         */

        for (auto &cell : triangulation.active_cell_iterators()) {
          const auto id = cell->material_id();
          cell->set_all_manifold_ids(id);
        }

        /*
         * Reattach manifolds:
         */
        if constexpr (dim == 1) {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();

        } else if constexpr (dim == 2) {
          unsigned int index = 10;
          for (const auto &manifold : manifolds)
            triangulation.set_manifold(index++, *manifold);

        } else {
          static_assert(dim == 3);

          unsigned int index = 10;
          for (const auto &manifold : manifolds)
            triangulation.set_manifold(
                index++, Manifolds::ExtrudedManifold<3>(*manifold));
        }

        /* Set boundary ids: */

        for (auto cell : triangulation.active_cell_iterators()) {
          for (auto f : cell->reference_cell().face_indices()) {
            auto face = cell->face(f);

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

            bool periodic_face = (dim == 3);

            if constexpr (dim == 3) {
              const auto &indices =
                  dealii::GeometryInfo<dim - 1>::vertex_indices();
              bool not_left = false;
              bool not_right = false;
              for (const auto v : indices) {
                const auto vert = face->vertex(v);
                if (vert[2] > 1.0e-10)
                  not_left = true;
                if (vert[2] < width_ - 1.0e-10)
                  not_right = true;
                if (not_left && not_right) {
                  periodic_face = false;
                  break;
                }
              }
            }

            if (periodic_face) {
              face->set_boundary_id(Boundary::periodic);
            } else if (spherical_boundary) {
              face->set_boundary_id(Boundary::dynamic);
            } else if (airfoil) {
              face->set_boundary_id(Boundary::no_slip);
            } else {
              Assert(false, dealii::ExcInternalError());
              __builtin_trap();
            }
          }
        }

        /* Add periodicity: */

        if constexpr (dim == 3) {
          std::vector<dealii::GridTools::PeriodicFacePair<
              typename dealii::Triangulation<dim>::cell_iterator>>
              periodic_faces;

          GridTools::collect_periodic_faces(triangulation,
                                            /*b_id */ Boundary::periodic,
                                            /*direction*/ 2,
                                            periodic_faces);

          triangulation.add_periodicity(periodic_faces);
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
      double grading_epsilon_trailing_;
      unsigned int n_anisotropic_refinements_airfoil_;
      unsigned int n_anisotropic_refinements_trailing_;
      unsigned int subdivisions_z_;
    };
  } /* namespace Geometries */
} /* namespace ryujin */
