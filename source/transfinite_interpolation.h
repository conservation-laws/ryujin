//
// SPDX-License-Identifier: LGPL-2.1+ or MIT
// Copyright (C) 2007 - 2022 by Martin Kronbichler
// Copyright (C) 2008 - 2022 by David Wells
//

#pragma once

#include <deal.II/base/config.h>
#include <deal.II/grid/manifold.h>

namespace ryujin
{
  using namespace dealii; // FIXME: namespace pollution

  /**
   * This is a copy of the TransfiniteInterpolationManifold shipped with
   * deal.II. In contrast to the deal.II version it copies the coarse grid
   * and all relevant Manifold information. That way it can be initialized
   * with one Triangulation and be used with another Triangulation.
   */
  template <int dim, int spacedim = dim>
  class TransfiniteInterpolationManifold : public Manifold<dim, spacedim>
  {
  public:
    TransfiniteInterpolationManifold();

    virtual ~TransfiniteInterpolationManifold() override = default;

    virtual std::unique_ptr<Manifold<dim, spacedim>> clone() const override;

    void initialize(
        const Triangulation<dim, spacedim> &triangulation,
        const Manifold<dim, spacedim> &chart_manifold = FlatManifold<dim>());

    virtual Point<spacedim>
    get_new_point(const ArrayView<const Point<spacedim>> &surrounding_points,
                  const ArrayView<const double> &weights) const override;

    virtual void
    get_new_points(const ArrayView<const Point<spacedim>> &surrounding_points,
                   const Table<2, double> &weights,
                   ArrayView<Point<spacedim>> new_points) const override;

  private:
    std::array<unsigned int, 20> get_possible_cells_around_points(
        const ArrayView<const Point<spacedim>> &surrounding_points) const;

    typename Triangulation<dim, spacedim>::cell_iterator compute_chart_points(
        const ArrayView<const Point<spacedim>> &surrounding_points,
        ArrayView<Point<dim>> chart_points) const;

    Point<dim>
    pull_back(const typename Triangulation<dim, spacedim>::cell_iterator &cell,
              const Point<spacedim> &p,
              const Point<dim> &initial_guess) const;

    Point<spacedim> push_forward(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const Point<dim> &chart_point) const;

    DerivativeForm<1, dim, spacedim> push_forward_gradient(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const Point<dim> &chart_point,
        const Point<spacedim> &pushed_forward_chart_point) const;

    Triangulation<dim, spacedim> triangulation;

    int level_coarse;

    std::vector<bool> coarse_cell_is_flat;

    std::unique_ptr<Manifold<dim, spacedim>> chart_manifold;
  };

} // namespace ryujin
