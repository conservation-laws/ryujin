//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/types.h>

#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The interface of a SolutionTransfer: it attaches the state to the
     * triangulation before mesh adaptation or serialization, and projects
     * the attached data into a new state vector afterwards.
     *
     * The interface holds the handle of the data attached to the
     * triangulation.
     *
     * @ingroup Interfaces
     */
    template <int dim, typename Number = double>
    class SolutionTransfer : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Functions for solution transfer after mesh adaptation and
       * for serialization/deserialization
       */
      //@{

      /**
       * Prepare projection (and serialization) by attaching all necessary
       * data to the triangulation to perform a projection after mesh
       * adaptation, or a Triangulation::store()/load() operation.
       */
      virtual void
      prepare_projection(const StateVector<Number> &old_state_vector) = 0;

      /**
       * Project the data stored in the triangulation into a new state
       * vector @p new_state_vector. The attached data either comes from a
       * previous call to prepare_projection() prior to mesh adaptation or
       * has been read back in from a checkpoint after a call to
       * Triangulation::load().
       */
      virtual void project(StateVector<Number> &new_state_vector) = 0;

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Return the handle associated with the call back that was set by
       * prepare_projection() and the associated data attached to the
       * triangulation.
       */
      unsigned int get_handle() const
      {
        Assert(handle_ != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage(
                   "Invalid handle: Cannot retrieve a valid handle. "
                   "get_handle() can only be called after a call to "
                   "prepare_projection(), or set_handle()."));
        return handle_;
      }

      /**
       * Set the handle associated data attached to the triangulation. This
       * function has to be called after a Triangulation::load() operation
       * and prior to project().
       */
      void set_handle(unsigned int handle)
      {
        Assert(handle_ == dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage(
                   "Invalid state: Cannot set handle because we already "
                   "have a valid handle due to a prior call to "
                   "prepare_projection(), or set_handle()."));
        handle_ = handle;
      }

      /**
       * Invalidate the currently stored handle. After a call to this
       * function a new handle can be registered with prepare_projection(),
       * or set_handle().
       */
      void reset_handle()
      {
        handle_ = dealii::numbers::invalid_unsigned_int;
      }

      /**
       * Return whether data is attached to the triangulation, i.e.,
       * whether a valid handle is set.
       */
      bool has_handle() const
      {
        return handle_ != dealii::numbers::invalid_unsigned_int;
      }

      //@}

    protected:
      /**
       * Constructor.
       */
      explicit SolutionTransfer(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , handle_(dealii::numbers::invalid_unsigned_int)
      {
      }

      /**
       * The handle of the data attached to the triangulation, or
       * dealii::numbers::invalid_unsigned_int if no data is attached.
       *
       * @note A derived class sets it in prepare_projection(). The
       * functions set_handle() and reset_handle() set it otherwise.
       */
      unsigned int handle_;
    };
  } // namespace Interface
} // namespace ryujin
