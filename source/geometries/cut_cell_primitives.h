//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace ryujin
{
  /**
   * A helper class that moves the vertices of a quadrilateral mesh so that
   * the mesh aligns with an elevation profile h(x). By convention the
   * negative y-direction points towards the bottom boundary described by
   * the profile. The triangulation must be in standard orientation, i.e.,
   * vertex i of every cell has the coordinate bits (i & 1, i & 2).
   *
   * After the alignment every cell is either entirely above the profile,
   * entirely below, or cut along a line connecting two of its vertices,
   * see CutCellDecomposition.
   *
   * The class works on serial and parallel::distributed triangulations and
   * the result does not depend on the number of MPI ranks: vertices are
   * moved in phases in which all decisions are taken with respect to the
   * vertex positions at the start of the phase, every rank moves the
   * vertices it owns (in the sense of
   * dealii::GridTools::get_locally_owned_vertices()), and the new
   * positions are communicated to the ghost layers.
   *
   * @note Only implemented in 2D.
   *
   * @ingroup Mesh
   */
  template <int dim>
  class MeshAlignment : dealii::ParameterAcceptor
  {
  public:
    MeshAlignment(const std::string &subsection)
        : ParameterAcceptor(subsection)
    {
      acceptable_deviation_ = 0.1;
      this->add_parameter(
          "acceptable deviation",
          acceptable_deviation_,
          "mesh alignment: a corner of the elevation profile that is cut "
          "off by a mesh edge or cell diagonal is captured by moving a "
          "vertex onto it if the orthogonal deviation of the profile from "
          "the chord exceeds this fraction of the cell size");

      minimal_jacobian_fraction_ = 0.25;
      this->add_parameter(
          "minimal jacobian fraction",
          minimal_jacobian_fraction_,
          "mesh alignment: a vertex is not moved if the move reduces a "
          "corner Jacobian of an adjacent cell below this fraction of the "
          "cell volume; edges cut by the profile are resolved regardless "
          "as long as no cell is inverted");

      max_sweeps_ = 16;
      this->add_parameter("maximal sweeps",
                          max_sweeps_,
                          "mesh alignment: maximal number of sweeps over all "
                          "cells when snapping vertices to corners");
    }


    /* The coordinate direction of the height: */
    static constexpr unsigned int profile_direction = 1;


    /*
     * Classify a point with respect to the profile: -1 below, 0 on the
     * profile (up to roundoff), +1 above.
     */
    template <typename Callable>
    static int vertex_state(const Callable &height,
                            const dealii::Point<dim> &point)
    {
      const double h = height(point);
      const double delta = point[profile_direction] - h;
      const double tolerance =
          1.e-12 *
          std::max({1., std::abs(point[0]), std::abs(point[1]), std::abs(h)});
      if (std::abs(delta) < tolerance)
        return 0;
      return delta > 0. ? 1 : -1;
    }


    /*
     * Send the data of every locally owned cell (unless it is empty) to its
     * ghost copies on the other ranks.
     */
    template <typename T>
    static void exchange_to_ghosts(dealii::Triangulation<dim> &triangulation,
                                   std::vector<T> &data)
    {
      using cell_iterator =
          typename dealii::Triangulation<dim>::active_cell_iterator;

      if (dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim> *>(
              &triangulation) == nullptr)
        return;

      dealii::GridTools::
          exchange_cell_data_to_ghosts<T, dealii::Triangulation<dim>>(
              triangulation,
              [&](const cell_iterator &cell) -> std::optional<T> {
                const auto &value = data[cell->active_cell_index()];
                if (value == T())
                  return {};
                return value;
              },
              [&](const cell_iterator &cell, const T &value) {
                data[cell->active_cell_index()] = value;
              });
    }


    /**
     * Align the triangulation with the elevation profile in two stages:
     *
     *  - Every edge that is cut by the profile is resolved by moving one of
     *    its vertices along the edge onto the cut. Afterwards every edge
     *    lies entirely above, entirely below, or has an endpoint on the
     *    profile.
     *
     *  - An edge or cell diagonal connecting two vertices on the profile is
     *    a chord of the profile and cuts off every corner of the profile in
     *    between. Such corners are located by sampling the profile along
     *    the chord and captured by moving a vertex of the cell onto them.
     *
     * A vertex on the left or right boundary only moves vertically.
     *
     * @p height is a callable taking a Point<dim> and returning the height
     * of the profile at the horizontal position of the point.
     */
    template <typename Callable>
    void align_with_elevation_profile(dealii::Triangulation<dim> &triangulation,
                                      const Callable &height) const
    {
      constexpr unsigned int y = profile_direction;
      constexpr auto invalid = dealii::numbers::invalid_unsigned_int;

      using Requests = std::map<unsigned int, dealii::Point<dim>>;

      const auto distributed =
          dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim> *>(
              &triangulation);

      /*
       * We need to collect and construct a bunch of expensive information
       * so that we can map from a vertex back to the surrounding cells:
       */

      const auto owned =
          dealii::GridTools::get_locally_owned_vertices(triangulation);
      const auto vertex_to_cells =
          dealii::GridTools::vertex_to_cell_map(triangulation);

      const unsigned int n_vertices = triangulation.n_vertices();
      const unsigned int n_cells = triangulation.n_active_cells();

      /*
       * A bunch of small lambdas for the following algorithm:
       */

      /* The state (above/below the interface) of all vertices of a cell: */
      const auto compute_vertex_states = [&]() {
        std::vector<int> result(n_vertices, 0);
        for (const auto &cell : triangulation.active_cell_iterators())
          if (!cell->is_artificial())
            for (unsigned int i = 0; i < 4; ++i)
              result[cell->vertex_index(i)] =
                  vertex_state(height, cell->vertex(i));
        return result;
      };

      /* Move vertex g to p (all cells share the vertex array): */
      const auto move_vertex = //
          [&](const unsigned int g, const dealii::Point<dim> &p) {
            const auto &cell = *vertex_to_cells[g].begin();
            for (unsigned int i = 0; i < 4; ++i)
              if (cell->vertex_index(i) == g)
                cell->vertex(i) = p;
          };

      /*
       * Whether vertex g may move to p:
       *  - a vertex on the left or right boundary cannot move horizontally,
       *  - all Jacobian of the cells around g must remain positive.
       *
       * FIXME: we detect this by temporarily moving the mesh...
       */
      const auto admissible = [&](const auto &cell,
                                  const unsigned int i,
                                  const dealii::Point<dim> &p,
                                  const double floor) {
        const auto g = cell->vertex_index(i);

        /* Boundary vertices cannot move horizontally: */
        if (p[0] != cell->vertex(i)[0]) // FIXME check with roundoff
          for (const auto &cell : vertex_to_cells[g])
            for (const unsigned int f : {0u, 1u})
              if (cell->face(f)->at_boundary())
                for (unsigned int k = 0; k < 2; ++k)
                  if (cell->face(f)->vertex_index(k) == g)
                    return false;

        /* Check Jacobian: */
        const auto saved = cell->vertex(i);
        move_vertex(g, p);
        double jacobian = std::numeric_limits<double>::max();
        for (const auto &cell : vertex_to_cells[g])
          for (unsigned int k = 0; k < 4; ++k) {
            const auto e_x = cell->vertex(k ^ 1) - cell->vertex(k);
            const auto e_y = cell->vertex(k ^ 2) - cell->vertex(k);
            const double sign = (k == 1 || k == 2) ? -1. : 1.;
            jacobian =
                std::min(jacobian, sign * (e_x[0] * e_y[1] - e_x[1] * e_y[0]));
          }
        move_vertex(g, saved);
        return jacobian > floor;
      };

      /*
       * Request a move of vertex g to p. Of two requests for the same
       * vertex the one with the smaller displacement wins (then the
       * lexicographically smaller target), independently of the order in
       * which the requests are collected:
       */
      const auto propose = [&](Requests &requests,
                               const auto &cell,
                               const unsigned int i,
                               const dealii::Point<dim> &p) {
        const auto g = cell->vertex_index(i);
        const auto key = [&](const dealii::Point<dim> &q) {
          return std::make_tuple((q - cell->vertex(i)).norm(), q[0], q[1]);
        };
        const auto [it, inserted] = requests.try_emplace(g, p);
        if (!inserted && key(p) < key(it->second))
          it->second = p;
      };

      /* Move locally owned vertices and communicate the positions: */
      const auto apply = [&](const Requests &requests) {
        for (const auto &[g, p] : requests)
          if (owned[g])
            move_vertex(g, p);
        if (distributed != nullptr)
          distributed->communicate_locally_moved_vertices(owned);
      };

      /*
       * Step 1:
       *
       * Resolve the edges cut by the profile, first the vertical, then the
       * horizontal ones. The cut is located by bisection and the closer
       * vertex is moved onto it, or the other one if the move is not
       * admissible. We first ask for a fraction of the cell volume at
       * every corner, and then only require that no cell is inverted.
       */

      for (const unsigned int direction : {y, 0u}) {
        const auto vertex_states = compute_vertex_states();
        Requests requests;

        for (const auto &cell : triangulation.active_cell_iterators()) {
          if (cell->is_artificial())
            continue;

          const double h0 = cell->diameter();
          const unsigned int bit = 1u << direction;
          for (unsigned int i = 0; i < 4; ++i) {
            const auto g_a = cell->vertex_index(i);
            const auto g_b = cell->vertex_index(i | bit);
            if ((i & bit) != 0 || vertex_states[g_a] * vertex_states[g_b] >= 0)
              continue;

            /* Bisection: */
            const auto a = cell->vertex(i);
            const auto d = cell->vertex(i | bit) - a;
            double t_a = 0., t_b = 1.;
            for (unsigned int k = 0; k < 60; ++k) {
              const double t = 0.5 * (t_a + t_b);
              const auto p = a + t * d;
              (vertex_state(height, p) == vertex_states[g_a] ? t_a : t_b) = t;
            }
            const double t = 0.5 * (t_a + t_b);
            auto cut = a + t * d;
            cut[y] = height(cut);

            const auto order =
                t <= 0.5 ? std::array{i, i | bit} : std::array{i | bit, i};
            bool done = false;
            for (const double floor :
                 {minimal_jacobian_fraction_ * h0 * h0, 0.})
              for (const auto j : order)
                if (!done && admissible(cell, j, cut, floor)) {
                  propose(requests, cell, j, cut);
                  done = true;
                }
          }
        }

        apply(requests);
      }

      /*
       * Step 2:
       *
       * Try to "snap" vertices onto corners of the profile cut off by a
       * chord. Every sweep is a phase in which
       *
       *  - every locally owned cell locates the corner of each of its
       *    chords and chooses the closest admissible vertex of the cell to
       *    capture it: an endpoint sliding along the profile or a vertex on
       *    the side of the corner. The owner of a cell sees all cells around
       *    the vertices of the cell, and sends the choice to the ghost
       *    copies of the cell;
       *
       *  - of the vertices chosen in a cell only the one with the highest
       *    (pseudo-random) priority moves, so that the admissibility checks
       *    remain valid.
       *
       * A snapped vertex is pinned. Sweeps are repeated until no vertex moves.
       */

      const auto priority = [&](const dealii::Point<dim> &v) {
        /* Hash the coordinate bits (hash_combine, splitmix64 finalizer): */
        std::uint64_t hash = 0x9e3779b97f4a7c15ull;
        for (unsigned int d = 0; d < dim; ++d) {
          const auto bits = std::bit_cast<std::uint64_t>(v[d]);
          hash ^= bits + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
          hash *= 0xbf58476d1ce4e5b9ull;
          hash ^= hash >> 31;
        }
        return hash;
      };

      constexpr std::array<std::pair<unsigned int, unsigned int>, 6> chords{
          {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {0, 3}, {1, 2}}};

      const double nan = std::numeric_limits<double>::quiet_NaN();

      std::vector<bool> pinned(n_vertices, false);

      for (unsigned int sweep = 0; sweep < max_sweeps_; ++sweep) {
        const auto vertex_states = compute_vertex_states();

        /* The target of every vertex of a cell (NaN if not chosen): */
        std::vector<std::vector<double>> records(n_cells);

        for (const auto &cell : triangulation.active_cell_iterators()) {
          if (!cell->is_locally_owned())
            continue;

          auto &record = records[cell->active_cell_index()];
          const double h0 = cell->diameter();

          for (const auto &[i_a, i_b] : chords) {
            const auto g_a = cell->vertex_index(i_a);
            const auto g_b = cell->vertex_index(i_b);
            if (vertex_states[g_a] != 0 || vertex_states[g_b] != 0)
              continue;

            /*
             * A diagonal only matters if the other two vertices lie on
             * opposite sides of the profile:
             */
            if ((i_a ^ i_b) == 3 &&
                vertex_states[cell->vertex_index(i_a ^ 1)] *
                        vertex_states[cell->vertex_index(i_a ^ 2)] >=
                    0)
              continue;

            /*
             * Locate the point of maximal deviation of the profile from the
             * chord (measured orthogonally to the chord, positive if the
             * profile lies above), refine the search once around the best
             * sample:
             */
            const auto a = cell->vertex(i_a);
            const auto d = cell->vertex(i_b) - a;
            double deviation = 0., t_best = 0., t_left = 0., t_right = 1.;
            dealii::Point<dim> corner;
            for (unsigned int level = 0; level < 2; ++level) {
              for (unsigned int s = 1; s < 64; ++s) {
                const double t = t_left + (t_right - t_left) * s / 64.;
                auto p = a + t * d;
                const double h = height(p);
                const double delta = (h - p[y]) * std::abs(d[0]) / d.norm();
                if (std::abs(delta) > std::abs(deviation)) {
                  deviation = delta;
                  p[y] = h;
                  corner = p;
                  t_best = t;
                }
              }
              const double dt = (t_right - t_left) / 64.;
              t_left = std::max(0., t_best - dt);
              t_right = std::min(1., t_best + dt);
            }

            if (std::abs(deviation) < acceptable_deviation_ * h0)
              continue;

            /*
             * Candidates are the two endpoints and the vertices on the side
             * of the corner, the closest admissible one is recorded (if a
             * vertex is chosen for two chords the smaller displacement
             * wins):
             */
            const int side = deviation > 0. ? 1 : -1;
            std::vector<std::tuple<double, double, double, unsigned int>>
                candidates;
            for (unsigned int i = 0; i < 4; ++i) {
              const auto &v = cell->vertex(i);
              if (i == i_a || i == i_b ||
                  vertex_states[cell->vertex_index(i)] == side)
                candidates.emplace_back((corner - v).norm(), v[0], v[1], i);
            }
            std::sort(candidates.begin(), candidates.end());

            for (const auto &[displacement, x, z, i] : candidates) {
              const auto g = cell->vertex_index(i);
              if (pinned[g] ||
                  !admissible(
                      cell, i, corner, minimal_jacobian_fraction_ * h0 * h0))
                continue;
              if (record.empty())
                record.assign(4 * dim, nan);
              const dealii::Point<dim> previous(record[2 * i],
                                                record[2 * i + 1]);
              if (std::isnan(previous[0]) ||
                  displacement < (previous - cell->vertex(i)).norm())
                for (unsigned int k = 0; k < dim; ++k)
                  record[2 * i + k] = corner[k];
              break;
            }
          }
        }

        exchange_to_ghosts(triangulation, records);

        Requests requests;
        for (const auto &cell : triangulation.active_cell_iterators()) {
          const auto &record = records[cell->active_cell_index()];
          if (cell->is_artificial() || record.empty())
            continue;
          for (unsigned int i = 0; i < 4; ++i)
            if (!std::isnan(record[2 * i]))
              propose(requests,
                      cell,
                      i,
                      dealii::Point<dim>(record[2 * i], record[2 * i + 1]));
        }

        /* Every locally owned cell vetoes all but one requested vertex: */
        std::vector<unsigned int> vetoes(n_cells, 0);
        for (const auto &cell : triangulation.active_cell_iterators()) {
          if (!cell->is_locally_owned())
            continue;
          unsigned int i_best = invalid;
          for (unsigned int i = 0; i < 4; ++i)
            if (requests.count(cell->vertex_index(i)) != 0 &&
                (i_best == invalid ||
                 priority(cell->vertex(i)) > priority(cell->vertex(i_best))))
              i_best = i;
          for (unsigned int i = 0; i < 4; ++i)
            if (i != i_best && requests.count(cell->vertex_index(i)) != 0)
              vetoes[cell->active_cell_index()] |= 1u << i;
        }

        exchange_to_ghosts(triangulation, vetoes);

        for (const auto &cell : triangulation.active_cell_iterators())
          if (!cell->is_artificial())
            for (unsigned int i = 0; i < 4; ++i)
              if ((vetoes[cell->active_cell_index()] & (1u << i)) != 0)
                requests.erase(cell->vertex_index(i));

        apply(requests);

        unsigned int n_moved = 0;
        for (const auto &[g, p] : requests) {
          pinned[g] = true;
          n_moved += owned[g];
        }
        n_moved = dealii::Utilities::MPI::max(
            n_moved, triangulation.get_mpi_communicator());
        if (n_moved == 0)
          break;
      }
    }

  private:
    double acceptable_deviation_;
    double minimal_jacobian_fraction_;
    unsigned int max_sweeps_;
  };


  /**
   * A helper class that creates a triangulation from the cells of a
   * triangulation aligned with an elevation profile (see MeshAlignment)
   * that lie above the profile, so that the profile becomes its bottom
   * boundary. After the alignment a cell is classified by the states of
   * its vertices:
   *
   *  - A cell without a vertex above the profile is dropped.
   *
   *  - A cut cell has a unique above vertex and, at the opposite corner,
   *    a below vertex. The other two vertices lie on the profile and the
   *    cell is replaced by the triangle spanned by the above vertex and
   *    the two on-vertices.
   *
   *  - A cell with three vertices on the profile (a corner captured by
   *    MeshAlignment) is replaced by the same triangle if the triangle of
   *    the three on-vertices lies below the profile.
   *
   *  - All other cells are kept.
   *
   * The boundary ids of the original mesh are preserved and every new
   * boundary face is assigned the given profile boundary id.
   *
   * A serial triangulation creates a serial triangulation, a
   * parallel::distributed::Triangulation creates a
   * parallel::fullydistributed::Triangulation on the same communicator.
   * Every element is owned by the owner of its cell and the result does
   * not depend on the number of MPI ranks.
   *
   * @note Only implemented in 2D.
   *
   * @ingroup Mesh
   */
  template <int dim>
  class CutCellDecomposition
  {
  public:
    /**
     * Create @p triangulation from the cells of the aligned triangulation
     * @p temporary above the profile. @p height is a callable taking a
     * Point<dim> and returning the height of the profile at the horizontal
     * position of the point.
     */
    template <typename Callable>
    void create_triangulation(
        dealii::Triangulation<dim> &triangulation,
        dealii::Triangulation<dim> &temporary,
        const Callable &height,
        const dealii::types::boundary_id profile_boundary_id) const
    {
      using Alignment = MeshAlignment<dim>;
      using Face = std::pair<unsigned int, unsigned int>;

      const auto comm = temporary.get_mpi_communicator();
      constexpr auto invalid = dealii::numbers::invalid_unsigned_int;

      const auto &vertices = temporary.get_vertices();

      /*
       * A lambda that returns all faces of a (quad or tet) element as a
       * vector of tuples of indices, in the canonical order of faces as
       * defined by the reference cell. Here, element is a vector of
       * (global) vertex indices.
       */
      const auto element_faces = [](const auto &element) {
        const auto reference_cell =
            dealii::ReferenceCells::n_vertices_to_reference_cell<dim>(
                element.size());

        std::vector<Face> result;

        for (const auto f : reference_cell.face_indices()) {
          const auto vertex_index_0 = reference_cell.face_to_cell_vertices(
              f, 0, dealii::numbers::default_geometric_orientation);
          const auto vertex_index_1 = reference_cell.face_to_cell_vertices(
              f, 1, dealii::numbers::default_geometric_orientation);
          result.emplace_back(
              std::minmax(element[vertex_index_0], element[vertex_index_1]));
        }

        return result;
      };

      /*
       * Step 1:
       *
       * We collect a record of every locally owned cell with a vertex
       * above the profile. A record consists of
       *  - a (new) global (coarse) cell ID that we form,
       *  - the x and y coordinates of all vertices of the original cell,
       *  - the element (quad or tet) encoded as 3 or 4 (cell) vertex indices.
       *
       *  Note: we need the vertex coordinates in a record as well to work
       *  around a bug in deal.II's communicate_locally_moved_vertices that
       *  fails to update some of the vertices in the ghost layer.
       */

      std::vector<std::vector<double>> records(temporary.n_active_cells());
      dealii::types::coarse_cell_id n_elements = 0;

      for (const auto &cell : temporary.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;

        std::array<int, 4> state;
        unsigned int i_above = invalid;
        for (unsigned int i = 0; i < 4; ++i) {
          state[i] = Alignment::vertex_state(height, cell->vertex(i));
          if (state[i] > 0)
            i_above = i;
        }

        if (i_above == invalid)
          continue;

        /*
         * Reading counterclockwise deal.II enumerates vertices 0->1->3->2.
         * Starting at index i_above we now select: the counterclockwise
         * neighbor a, the opposite vertex c, and its other neighbor b:
         */
        constexpr std::array<unsigned int, 4> ccw_next_vertex{{1, 3, 0, 2}};
        const auto a = ccw_next_vertex[i_above];
        const auto c = ccw_next_vertex[a];
        const auto b = ccw_next_vertex[c];
        const auto centroid =
            (cell->vertex(a) + cell->vertex(b) + cell->vertex(c)) / 3.;

        std::vector<unsigned int> element{0, 1, 2, 3}; /* uncut quad */

        /* We cut along the diagonal a-b if it separates i_above from c: */
        const bool opposite_vertex_below = state[c] < 0;
        const bool triangle_abc_below =
            state[a] == 0 && state[b] == 0 && state[c] == 0 &&
            Alignment::vertex_state(height, centroid) < 0;

        if (opposite_vertex_below || triangle_abc_below)
          element = {i_above, a, b}; /* triangle formed by removing vertex c */

        auto &record = records[cell->active_cell_index()];
        /* conversion to double is exact for n_elements < 2^53 */
        record.push_back(static_cast<double>(n_elements++));
        for (unsigned int i = 0; i < 4; ++i)
          for (int d = 0; d < dim; ++d)
            record.push_back(cell->vertex(i)[d]);
        record.insert(record.end(), element.begin(), element.end());
      }

      /*
       * Shift the rank-local ID by an appropriate offset to obtain a
       * unique global coarse cell ID:
       */
      const auto offset =
          dealii::Utilities::MPI::partial_and_total_sum(n_elements, comm).first;

      for (auto &record : records)
        if (!record.empty())
          record[0] += static_cast<double>(offset);

      /* We use the temporary mesh to exchange information: */
      Alignment::exchange_to_ghosts(temporary, records);

      /*
       * Step 2:
       *
       * Collect all elements (quads or tests) derived from locally owned
       * and ghost cells of the temporary triangulation and count how often
       * every face occurs: a face occurring once is a boundary face. It
       * inherits the boundary id of a boundary face of its cell, otherwise
       * it lies on the profile.
       */

      std::vector<dealii::CellData<dim>> cells;
      std::vector<dealii::types::coarse_cell_id> ids;
      std::vector<dealii::types::subdomain_id> owners;
      std::map<Face, std::pair<unsigned int, dealii::types::boundary_id>> faces;

      for (const auto &cell : temporary.active_cell_iterators()) {
        const auto &record = records[cell->active_cell_index()];
        if (cell->is_artificial() || record.empty())
          continue;

        /*
         * Work around a bug in communicate_locally_moved_vertices(): it
         * only sends vertices moved by the owner of a cell, so a vertex of
         * a ghost cell owned by a third rank may still be at its old position.
         * Take the vertex coordinates from the owner instead.
         */
        for (unsigned int i = 0; i < 4; ++i)
          for (int d = 0; d < dim; ++d)
            cell->vertex(i)[d] = record[1 + i * dim + d];

        dealii::CellData<dim> data;
        data.vertices.clear();
        /* Iterate over the "element" vertex indices: */
        for (auto it = record.begin() + 1 + 4 * dim; it != record.end(); ++it) {
          const auto global_index =
              cell->vertex_index(static_cast<unsigned int>(*it));
          data.vertices.push_back(global_index);
        }

        for (const auto &face : element_faces(data.vertices)) {
          auto &[count, boundary_id] = faces[face];
          ++count;
          boundary_id = profile_boundary_id;
          for (const auto f : cell->face_indices()) {
            const Face cell_face = std::minmax(cell->face(f)->vertex_index(0),
                                               cell->face(f)->vertex_index(1));
            if (cell->face(f)->at_boundary() && cell_face == face)
              boundary_id = cell->face(f)->boundary_id();
          }
        }

        cells.push_back(data);
        ids.push_back(static_cast<dealii::types::coarse_cell_id>(record[0]));
        owners.push_back(cell->subdomain_id());
      }

      /*
       * Step 3: Create a Triangulation description of the local part of the
       * new mesh, the locally owned elements and the elements sharing a vertex
       * with them, and create the triangulation.
       */

      const auto rank = dealii::Utilities::MPI::this_mpi_process(comm);

      std::vector<bool> relevant(vertices.size(), false);
      for (unsigned int cell = 0; cell < cells.size(); ++cell)
        if (owners[cell] == rank)
          for (const auto vertex_index : cells[cell].vertices)
            relevant[vertex_index] = true;

      dealii::TriangulationDescription::Description<dim> description;
      description.cell_infos.resize(1);

      std::vector<unsigned int> new_index(vertices.size(), invalid);

      for (unsigned int cell = 0; cell < cells.size(); ++cell) {
        auto data = cells[cell];

        if (std::none_of(data.vertices.begin(),
                         data.vertices.end(),
                         [&](const auto vertex_index) {
                           return relevant[vertex_index];
                         }))
          continue;

        Assert( //
            dealii::GridTools::cell_measure<dim>(vertices, data.vertices) > 0.,
            dealii::ExcMessage(
                "The cut cell decomposition created an inverted element."));

        dealii::TriangulationDescription::CellData<dim> info;
        info.id = dealii::CellId(ids[cell], std::vector<std::uint8_t>())
                      .template to_binary<dim>();
        info.subdomain_id = owners[cell];
        info.level_subdomain_id = owners[cell];

        const auto element_face = element_faces(data.vertices);
        for (unsigned int f = 0; f < element_face.size(); ++f) {
          const auto &[count, boundary_id] = faces[element_face[f]];
          if (count == 1)
            info.boundary_ids.emplace_back(f, boundary_id);
        }

        for (auto &v : data.vertices) {
          if (new_index[v] == invalid) {
            new_index[v] = description.coarse_cell_vertices.size();
            description.coarse_cell_vertices.push_back(vertices[v]);
          }
          v = new_index[v];
        }

        description.coarse_cells.push_back(data);
        description.coarse_cell_index_to_coarse_cell_id.push_back(ids[cell]);
        description.cell_infos[0].push_back(info);
      }

      description.comm = triangulation.get_mpi_communicator();
      description.settings = dealii::TriangulationDescription::Settings::
          construct_multigrid_hierarchy;
      description.smoothing = triangulation.get_mesh_smoothing();

      triangulation.clear();
      triangulation.create_triangulation(description);
    }
  };

} // namespace ryujin
