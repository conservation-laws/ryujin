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
    public:
      AirfoilManifold(const dealii::Point<dim> airfoil_center,
                      const std::function<double(const double)> &psi_front,
                      const std::function<double(const double)> &psi_upper,
                      const std::function<double(const double)> &psi_lower,
                      const bool upper_side)
          : airfoil_center(airfoil_center)
          , psi_front(psi_front)
          , psi_upper(psi_upper)
          , psi_lower(psi_lower)
          , upper_side(upper_side)
          , ratio_(psi_front(0.) / psi_front(M_PI))
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
        return std::make_unique<AirfoilManifold<dim>>(
            airfoil_center, psi_front, psi_upper, psi_lower, upper_side);
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
                      const double grading,
                      const double epsilon)
          : center(center)
          , grading(grading)
          , epsilon(epsilon)
          , polar_manifold(center)
      {
      }

      virtual dealii::Point<dim>
      pull_back(const dealii::Point<dim> &space_point) const final override
      {
        auto point = polar_manifold.pull_back(space_point);
        Assert(point[0] >= 0., dealii::ExcInternalError());
        point[0] = std::pow(point[0] + epsilon, 1. / grading) -
                   std::pow(epsilon, 1. / grading) + 1.e-14;
        const auto chart_point = polar_manifold.push_forward(point);
        return chart_point;
      }

      virtual dealii::Point<dim>
      push_forward(const dealii::Point<dim> &chart_point) const final override
      {
        auto point = polar_manifold.pull_back(chart_point);
        point[0] =
            std::pow(point[0] + std::pow(epsilon, 1. / grading), grading) -
            epsilon + 1.e-14;
        Assert(point[0] >= 0., dealii::ExcInternalError());
        return polar_manifold.push_forward(point);
      }

      std::unique_ptr<dealii::Manifold<dim, dim>> clone() const final override
      {
        return std::make_unique<GradingManifold<dim>>(center, grading, epsilon);
      }

    private:
      const dealii::Point<dim> center;
      const double grading;
      const double epsilon;

      dealii::PolarManifold<dim> polar_manifold;
    };
  } // namespace Manifolds

  /**
   * This namespace provides a collection of functions for generating
   * triangulations for some benchmark configurations.
   *
   * @ingroup Mesh
   */
  namespace GridGenerator
  {
    using namespace dealii::GridGenerator;


    /**
     * Create a 2D airfoil
     *
     * @todo documentation
     *
     * @ingroup Mesh
     */
    template <int dim, int spacedim, template <int, int> class Triangulation>
    void airfoil(Triangulation<dim, spacedim> &,
                 const dealii::Point<spacedim> &,
                 const std::function<double(const double)> &,
                 const std::function<double(const double)> &,
                 const std::function<double(const double)> &,
                 const double,
                 const double)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }


#ifndef DOXYGEN
    template <template <int, int> class Triangulation>
    void airfoil(Triangulation<2, 2> &triangulation,
                 const dealii::Point<2> &airfoil_center,
                 const std::function<double(const double)> &psi_front,
                 const std::function<double(const double)> &psi_upper,
                 const std::function<double(const double)> &psi_lower,
                 const double outer_radius,
                 const double grading,
                 const double grading_epsilon,
                 unsigned int n_anisotropic_refinements_airfoil,
                 unsigned int n_anisotropic_refinements_trailing)
    {
      /* by convention, psi_front(0.) returns the "back length" */
      const auto back_length = psi_front(0.);

      /* sharp trailing edge? */
      const bool sharp_trailing_edge =
          std::abs(psi_upper(back_length) - psi_lower(back_length)) < 1.0e-10;
      AssertThrow(
          sharp_trailing_edge ||
              std::abs(psi_upper(back_length) - psi_lower(back_length)) >
                  0.01 * back_length,
          dealii::ExcMessage("Blunt trailing edge has a width of less than "
                             "1% of the trailing airfoil length."));

      /* Front part: */

      std::vector<dealii::Point<2>> vertices1{
          {-outer_radius, 0.0},                                      // 0
          {airfoil_center[0] - psi_front(M_PI), airfoil_center[1]},  // 1
          {-0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 2
          {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius},  // 3
          {0., airfoil_center[1] + psi_lower(-airfoil_center[0])},   // 4
          {airfoil_center[0] + back_length,                          //
           airfoil_center[1] + psi_lower(back_length)},              // 5
          {0., airfoil_center[1] + psi_upper(-airfoil_center[0])},   // 6
          {-0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius},  // 7
          {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius},   // 8
      };

      std::vector<dealii::CellData<2>> cells1(4);
      cells1[0].vertices = {2, 3, 4, 5};
      cells1[1].vertices = {0, 2, 1, 4};
      cells1[2].vertices = {7, 0, 6, 1};
      if (sharp_trailing_edge) {
        cells1[3].vertices = {8, 7, 5, 6};
      } else {
        vertices1.push_back({airfoil_center[0] + back_length,
                             airfoil_center[1] + psi_upper(back_length)});
        cells1[3].vertices = {8, 7, 9, 6};
      }

      dealii::Triangulation<2> tria1;
      tria1.create_triangulation(vertices1, cells1, dealii::SubCellData());

      dealii::Triangulation<2> tria2;

      if (sharp_trailing_edge) {
        /* Back part for sharp trailing edge: */

        const std::vector<dealii::Point<2>> vertices2{
            {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 0
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_lower(back_length)},            // 1
            {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius}, // 2
            {outer_radius, -0.5 * outer_radius},                     // 3
            {outer_radius, 0.0},                                     // 4
            {outer_radius, 0.5 * outer_radius},                      // 5
        };

        std::vector<dealii::CellData<2>> cells2(2);
        cells2[0].vertices = {0, 3, 1, 4};
        cells2[1].vertices = {1, 4, 2, 5};

        tria2.create_triangulation(vertices2, cells2, dealii::SubCellData());

      } else {
        /* Back part for blunt trailing edge: */

        /* Good width for the anisotropically refined center trailing cell: */
        double trailing_height =
            0.5 / (0.5 + std::pow(2., n_anisotropic_refinements_airfoil)) *
            0.5 * outer_radius;

        const std::vector<dealii::Point<2>> vertices2{
            {0.5 * outer_radius, -std::sqrt(3.) / 2. * outer_radius}, // 0
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_lower(back_length)}, // 1
            {airfoil_center[0] + back_length,
             airfoil_center[1] + psi_upper(back_length)},            // 2
            {0.5 * outer_radius, std::sqrt(3.) / 2. * outer_radius}, // 3
            {outer_radius, -0.5 * outer_radius},                     // 4
            {outer_radius, -trailing_height},                        // 5
            {outer_radius, trailing_height},                         // 6
            {outer_radius, 0.5 * outer_radius},                      // 7
        };

        std::vector<dealii::CellData<2>> cells2(3);
        cells2[0].vertices = {0, 4, 1, 5};
        cells2[1].vertices = {1, 5, 2, 6};
        cells2[2].vertices = {2, 6, 3, 7};

        tria2.create_triangulation(vertices2, cells2, dealii::SubCellData());
      }

      dealii::Triangulation<2> coarse_triangulation;
      GridGenerator::merge_triangulations(
          {&tria1, &tria2}, coarse_triangulation, 1.e-12, true);

      /*
       * Set manifold IDs and attach manifolds to preliminary coarse
       * triangulation:
       *
       *   1 -> upper airfoil (inner boundary)
       *   2 -> lower airfoil (inner boundary)
       *   3 -> spherical manifold (outer boundary)
       *   4 -> grading front (interior face)
       *   5 -> grading upper airfoil (interior face)
       *   6 -> grading lower airfoil (interior face)
       *   7 -> grading upper back (interior face)
       *   8 -> grading lower back (interior face)
       *   9 -> transfinite interpolation
       */

      coarse_triangulation.set_all_manifold_ids(9);

      /* all possible vertices for the four (or six) radials: */
      const std::vector<dealii::Point<2>> radial_vertices{
          {airfoil_center[0] - psi_front(M_PI), airfoil_center[1]}, // front
          {0., airfoil_center[1] + psi_upper(-airfoil_center[0])},  // upper
          {0., airfoil_center[1] + psi_lower(-airfoil_center[0])},  // lower
          {airfoil_center[0] + back_length,
           airfoil_center[1] + psi_upper(back_length)}, // upper back
          {airfoil_center[0] + back_length,
           airfoil_center[1] + psi_lower(back_length)}, // lower back
      };

      for (auto cell : coarse_triangulation.active_cell_iterators()) {

        for (auto f : dealii::GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);

          if (face->at_boundary()) {
            /* Handle boundary faces: */

            bool airfoil = true;
            bool spherical_boundary = true;

            for (const auto v : dealii::GeometryInfo<1>::vertex_indices())
              if (std::abs((face->vertex(v)).norm() - outer_radius) < 1.0e-10)
                airfoil = false;
              else
                spherical_boundary = false;

            if (spherical_boundary) {
              face->set_manifold_id(3);
            }

            if (airfoil) {
              if (face->center()[0] < airfoil_center[0] + back_length - 1.e-6) {
                if (face->center()[1] >= airfoil_center[1]) {
                  face->set_manifold_id(1);
                } else {
                  face->set_manifold_id(2);
                }
              }
            }

          } else {
            /* Handle radial faces: */

            unsigned int index = 4;
            for (auto candidate : radial_vertices) {
              if (candidate.distance(face->vertex(0)) < 1.0e-10 || //
                  candidate.distance(face->vertex(1)) < 1.0e-10) {
                Assert(index < 10, dealii::ExcInternalError());
                face->set_manifold_id(index);
                break;
              }
              index++;
            }
          }
        } /* f */
      }   /* cell */

      Manifolds::AirfoilManifold airfoil_manifold_upper{
          airfoil_center, psi_front, psi_upper, psi_lower, true};
      coarse_triangulation.set_manifold(1, airfoil_manifold_upper);

      Manifolds::AirfoilManifold airfoil_manifold_lower{
          airfoil_center, psi_front, psi_upper, psi_lower, false};
      coarse_triangulation.set_manifold(2, airfoil_manifold_lower);

      dealii::SphericalManifold<2> spherical_manifold;
      coarse_triangulation.set_manifold(3, spherical_manifold);

      unsigned int index = 4;
      for (auto vertex : radial_vertices) {
        Manifolds::GradingManifold manifold{vertex, grading, grading_epsilon};
        coarse_triangulation.set_manifold(index, manifold);
        index++;
      }
      Assert(index == 9, dealii::ExcInternalError());

      ryujin::TransfiniteInterpolationManifold<2> transfinite;
      transfinite.initialize(coarse_triangulation);
      coarse_triangulation.set_manifold(9, transfinite);

      /*
       * Anisotropic pre refinement:
       */

      /* Mark critical cells with a temporary material id: */
      for (auto cell : coarse_triangulation.active_cell_iterators()) {

        /* in case of a blunt edge we refine the trailing cell: */
        if (!sharp_trailing_edge)
          if (cell->center()[0] > airfoil_center[0] + back_length &&
              std::abs(cell->center()[1]) <=
                  1.1 * std::abs(airfoil_center[1]) + 1.0e-6)
            cell->set_material_id(2);

        /*
         * Let us also insert additional radials on the upper (back) and
         * lower (back) side of the airfoil
         */
        if (cell->center()[0] > airfoil_center[0] &&
            cell->center()[0] <
                0.5 * (airfoil_center[0] + back_length + 0.5 * outer_radius))
          cell->set_material_id(3);
      }

      /* Upper and lower cell on airfoil: */
      for (unsigned int i = 0; i < n_anisotropic_refinements_airfoil; ++i) {
        for (auto cell : coarse_triangulation.active_cell_iterators())
          if (cell->material_id() == 3)
            cell->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));

        coarse_triangulation.execute_coarsening_and_refinement();
      }

      /* Tailing cell: */
      for (unsigned int i = 0; i < n_anisotropic_refinements_trailing; ++i) {
        for (auto cell : coarse_triangulation.active_cell_iterators())
          if (cell->material_id() == 2)
            cell->set_refine_flag(dealii::RefinementCase<2>::cut_axis(0));
          else
            cell->set_refine_flag();
        coarse_triangulation.execute_coarsening_and_refinement();
      }

      /* Flatten triangulation and create distributed coarse triangulation: */

      dealii::Triangulation<2> tria4;
      GridGenerator::flatten_triangulation(coarse_triangulation, tria4);
      triangulation.copy_triangulation(tria4);

      triangulation.set_all_manifold_ids(9);
      triangulation.set_manifold(9, transfinite);

      /* Set boundary ids: */

      for (auto cell : triangulation.active_cell_iterators()) {
        for (auto f : dealii::GeometryInfo<2>::face_indices()) {
          const auto face = cell->face(f);
          if (face->at_boundary()) {
            /* Handle boundary faces: */

            bool airfoil = true;
            bool spherical_boundary = true;

            for (const auto v : dealii::GeometryInfo<1>::vertex_indices())
              if (std::abs((face->vertex(v)).norm() - outer_radius) < 1.0e-10)
                airfoil = false;
              else
                spherical_boundary = false;

            if (spherical_boundary)
              face->set_boundary_id(Boundary::dirichlet);

            if (airfoil)
              face->set_boundary_id(Boundary::no_slip);
          }
        }
      }
    }
#endif
  } /* namespace GridGenerator */


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

      return {x_upper, y_upper, x_lower, y_lower};
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
      Assert(std::is_sorted(x_upper.begin(), x_upper.end()), dealii::ExcInternalError());

      Assert(x_lower.size() >= 2, dealii::ExcInternalError());
      Assert(x_lower.front() == 0. && x_lower.back() == 1.,
             dealii::ExcInternalError());
      Assert(std::is_sorted(x_lower.begin(), x_lower.end()),
             dealii::ExcInternalError());

      Assert(y_upper.size() == x_upper.size(), dealii::ExcInternalError());
      Assert(y_upper.front() == 0. && y_upper.back() >= 0.,
             dealii::ExcInternalError());

      Assert(y_lower.size() == x_lower.size(), dealii::ExcInternalError());
      Assert(y_lower.front() == 0. && y_lower.back() <= 0.,
             dealii::ExcInternalError());

      Assert(0. < x_center && x_center < 1., dealii::ExcInternalError());

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
      if(x_combined.back() == 0.)
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

      return {psi_front, psi_upper, psi_lower};
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

        airfoil_type_ = "NACA 2412";
        this->add_parameter(
            "airfoil type", airfoil_type_, "airfoil type and serial number");

        airfoil_length_ = 1.;
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


        airfoil_center_[0] = -.2;
        this->add_parameter("airfoil center",
                            airfoil_center_,
                            "position of airfoil center in the mesh");

        /* Parameters affecting mesh generation: */

        grading_ = 6.;
        this->add_parameter(
            "grading exponent", grading_, "graded mesh: exponent");

        grading_epsilon_ = 0.02;
        this->add_parameter("grading epsilon",
                            grading_epsilon_,
                            "graded mesh: regularization parameter");

        length_ = 5.;
        this->add_parameter(
            "length", length_, "length of computational domain (diameter)");

        n_anisotropic_refinements_airfoil_ = 1;
        this->add_parameter(
            "anisotropic pre refinement airfoil",
            n_anisotropic_refinements_airfoil_,
            "number of anisotropic pre refinement steps for the airfoil");

        n_anisotropic_refinements_trailing_ = 0;
        this->add_parameter("anisotropic pre refinement trailing",
                            n_anisotropic_refinements_trailing_,
                            "number of anisotropic pre refinement steps for "
                            "the blunt trailing edge cell");
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
         * Step 2: Create mesh and attach manifolds:
         *
         * Runtime parameters: double grading_, double grading_epsilon_,
         * double length_, unsigned int n_anisotropic_refinements_airfoil_,
         * unsigned int n_anisotropic_refinements_trailing_
         */

        GridGenerator::airfoil(triangulation,
                               airfoil_center_,
                               psi_front,
                               psi_upper,
                               psi_lower,
                               0.5 * length_,
                               grading_,
                               grading_epsilon_,
                               n_anisotropic_refinements_airfoil_,
                               n_anisotropic_refinements_trailing_);
      }

    private:
      dealii::Point<dim> airfoil_center_;
      double airfoil_length_;
      std::string airfoil_type_;
      dealii::Point<dim> psi_center_;
      unsigned int psi_samples_;
      double length_;
      double grading_;
      double grading_epsilon_;
      unsigned int n_anisotropic_refinements_airfoil_;
      unsigned int n_anisotropic_refinements_trailing_;
    };

  } /* namespace Geometries */

} /* namespace ryujin */

#endif /* GRID_AIRFOIL_H */
