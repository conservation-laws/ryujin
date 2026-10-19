//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/interface/discretization.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/tria.h>

#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The interface of a MeshAdaptor: it decides, based on the current
     * state, whether a mesh adaptation cycle should be performed, marks
     * cells for coarsening and refinement, and/or may replace the
     * transformation of the Discretization, i.e., change the geometry.
     *
     * @ingroup Interfaces
     */
    template <int dim, typename Number = double>
    class MeshAdaptor : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs
       */
      //@{

      using Transformation = typename Discretization<dim>::Transformation;

      //@}
      /**
       * @name Setup
       */
      //@{

      /**
       * Prepare temporary storage and clean up internal data for the
       * analyze() facility.
       */
      virtual void prepare(const Number t) = 0;

      //@}
      /**
       * @name Functions for performing mesh adaptation
       */
      //@{

      /**
       * Analyze the given StateVector and decide whether a mesh adaptation
       * cycle should be performed. The result is available through
       * need_mesh_adaptation().
       */
      virtual void analyze(const StateVector<Number> &state_vector,
                           const Number t,
                           unsigned int cycle) = 0;

      /**
       * Mark cells for coarsening and refinement.
       */
      virtual void mark_cells_for_coarsening_and_refinement(
          dealii::Triangulation<dim> &triangulation) const = 0;

      /**
       * Update the transformation of the Discretization that maps points
       * of the triangulation to the actual geometry.
       */
      virtual void
      update_transformation(Transformation &transformation) const = 0;

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * A boolean indicating whether we should perform a mesh adaptation
       * step in the current cycle. The analyze() method sets this boolean
       * to true whenever an adaptation cycle should be performed.
       */
      ACCESSOR_READ_ONLY(need_mesh_adaptation)

      //@}

    protected:
      /**
       * Constructor.
       */
      explicit MeshAdaptor(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , need_mesh_adaptation_(false)
      {
      }

      /**
       * Whether a mesh adaptation step should be performed in the current
       * cycle.
       *
       * @note A derived class sets it in prepare() and analyze().
       */
      bool need_mesh_adaptation_;
    };
  } // namespace Interface
} // namespace ryujin
