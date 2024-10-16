//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <string>

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * A small abstract base class to group configuration options for an
     * equation of state.
     *
     * @ingroup EulerEquations
     */
    class EquationOfState : public dealii::ParameterAcceptor
    {
    public:
      /**
       * Constructor taking EOS name @p name and a subsection @p subsection
       * as an argument. The dealii::ParameterAcceptor is initialized with
       * the subsubsection `subsection + "/" + name`.
       */
      EquationOfState(const std::string &name, const std::string &subsection)
          : ParameterAcceptor(subsection + "/" + name)
          , name_(name)
      {
        /*
         * If necessary derived EOS can override the covolume b that is
         * used in the interpolatory NASG eos.
         */
        interpolation_b_ = 0.;

        /*
         * If necessary derived EOS can override the reference pressure
         * that is used in the interpolatory NASG eos.
         */
        interpolation_pinfty_ = 0.;

        /*
         * If necessary derived EOS can override the reference specific
         * internal energy q that is used in the interpolatory NASG eos.
         */
        interpolation_q_ = 0.;

        /*
         * If necessary derived EOS can override this boolean to indicate
         * that the dealii::ArrayView<double> variants of the pressure()
         * function (etc.) should be preferred.
         */
        prefer_vector_interface_ = false;
      }

      /**
       * Return the pressure given density @p rho and specific internal
       * energy @p e.
       */
      virtual double pressure(double rho, double e) const = 0;

      /**
       * Variant of above function operating on a contiguous range of
       * values. The result is stored in the first argument @p p,
       * overriding previous contents.
       *
       * @note The second and third arguments are writable as well. We need
       * to perform some unit transformations for certain tabulated
       * equation of state libraries, such as the sesame database. Rather
       * than creating temporaries we override values in place.
       */
      virtual void pressure(const dealii::ArrayView<double> &p,
                            const dealii::ArrayView<double> &rho,
                            const dealii::ArrayView<double> &e) const
      {
        Assert(p.size() == rho.size() && rho.size() == e.size(),
               dealii::ExcMessage("vectors have different size"));

        std::transform(std::begin(rho),
                       std::end(rho),
                       std::begin(e),
                       std::begin(p),
                       [&](double rho, double e) { return pressure(rho, e); });
      }

      /**
       * Return the specific internal energy @p e for a given density @p
       * rho and pressure @p p.
       */
      virtual double specific_internal_energy(double rho, double p) const = 0;

      /**
       * Variant of above function operating on a contiguous range of
       * values. The result is stored in the first argument @p p,
       * overriding previous contents.
       *
       * @note The second and third arguments are writable as well. We need
       * to perform some unit transformations for certain tabulated
       * equation of state libraries, such as the sesame database. Rather
       * than creating temporaries we override values in place.
       */
      virtual void
      specific_internal_energy(const dealii::ArrayView<double> &e,
                               const dealii::ArrayView<double> &rho,
                               const dealii::ArrayView<double> &p) const
      {
        Assert(p.size() == rho.size() && rho.size() == e.size(),
               dealii::ExcMessage("vectors have different size"));

        std::transform(std::begin(rho),
                       std::end(rho),
                       std::begin(p),
                       std::begin(e),
                       [&](double rho, double p) {
                         return specific_internal_energy(rho, p);
                       });
      }

      /**
       * Return the temperature @p T for a given density @p
       * rho and specific internal energy @p e.
       */
      virtual double temperature(double rho, double e) const = 0;

      /**
       * Variant of above function operating on a contiguous range of
       * values. The result is stored in the first argument @p T,
       * overriding previous contents.
       *
       * @note The second and third arguments are writable as well. We need
       * to perform some unit transformations for certain tabulated
       * equation of state libraries, such as the sesame database. Rather
       * than creating temporaries we override values in place.
       */
      virtual void temperature(const dealii::ArrayView<double> &T,
                               const dealii::ArrayView<double> &rho,
                               const dealii::ArrayView<double> &e) const
      {
        Assert(T.size() == rho.size() && rho.size() == e.size(),
               dealii::ExcMessage("vectors have different size"));

        std::transform(
            std::begin(rho),
            std::end(rho),
            std::begin(e),
            std::begin(T),
            [&](double rho, double e) { return temperature(rho, e); });
      }

      /**
       * Return the sound speed @p c for a given density @p rho and
       * specific internal energy  @p e.
       */
      virtual double speed_of_sound(double rho, double e) const = 0;

      /**
       * Variant of above function operating on a contiguous range of
       * values. The result is stored in the first argument @p p,
       * overriding previous contents.
       *
       * @note The second and third arguments are writable as well. We need
       * to perform some unit transformations for certain tabulated
       * equation of state libraries, such as the sesame database. Rather
       * than creating temporaries we override values in place.
       */
      virtual void speed_of_sound(const dealii::ArrayView<double> &c,
                                  const dealii::ArrayView<double> &rho,
                                  const dealii::ArrayView<double> &e) const
      {
        Assert(c.size() == rho.size() && rho.size() == e.size(),
               dealii::ExcMessage("vectors have different size"));

        std::transform(
            std::begin(rho),
            std::end(rho),
            std::begin(e),
            std::begin(c),
            [&](double rho, double e) { return speed_of_sound(rho, e); });
      }

      /**
       * Return the interpolation covolume constant (b).
       */
      ACCESSOR_READ_ONLY(interpolation_b)

      /**
       * Return the interpolation reference pressure (pinfty).
       */
      ACCESSOR_READ_ONLY(interpolation_pinfty)

      /**
       * Return the interpolation reference specific internal energy (q).
       */
      ACCESSOR_READ_ONLY(interpolation_q)

      /**
       * Return a boolean indicating whether the dealii::ArrayView<double>
       * variants for the pressure(), specific_internal_energy(), and
       * speed_of_sound() functions should be preferred.
       *
       * Ordinarily we use the single-valued signatures for pre-computation
       * because this leads to slightly better throughput (due to better
       * memory locality with how we store precomputed values) and less
       * memory consumption. On the other hand, some tabulated equation of
       * state libraries work best with a single call and a large dataset.
       */
      ACCESSOR_READ_ONLY(prefer_vector_interface)

      /**
       * Return the name of the EOS as (const reference) std::string
       */
      ACCESSOR_READ_ONLY(name)

    protected:
      double interpolation_b_;
      double interpolation_pinfty_;
      double interpolation_q_;
      bool prefer_vector_interface_;

    private:
      const std::string name_;
    };

  } // namespace EquationOfStateLibrary
} /* namespace ryujin */
