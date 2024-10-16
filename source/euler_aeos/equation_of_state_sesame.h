//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <filesystem>

#include <compile_time_options.h>

#include "equation_of_state.h"
#include "lazy.h"

#include <deal.II/base/array_view.h>
#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>

#ifdef WITH_EOSPAC
#include "eos_Interface.h"
#endif


namespace ryujin
{
#ifdef WITH_EOSPAC
  /**
   * A namespace with wrappers for the eospac6 library and sesame database. The
   * eospac6 wrapper needs the file 'sesameFilesDir.txt' to be defined in the
   * directory where you run your simulation. The file 'sesameFilesDir.txt'
   * should list the path pointing to the sesame database. We refer the user to
   * the EOSPAC manual for more information.
   */
  namespace eospac
  {
    /**
     * A list of table types we support through our interface. This list is
     * tailored to what we need for our EquationOfState interface and thus
     * substantially shorter than what eospac6 supports.
     */
    enum class TableType : EOS_INTEGER {
      /**
       * (total) pressure [GPa] given as a function of density [Mg/m^3] and
       * (total) specific internal energy [MJ/kg]
       */
      p_rho_e = EOS_Pt_DUt,
      /**
       * (total) specific internal energy [Mj/kg] as a function of density
       * [Mg/m^3] and (total) pressure [GPa]
       */
      e_rho_p = EOS_Ut_DPt,
    };

    class Interface
    {
    public:
      /**
       * Constructor
       *
       * takes a vector consisting of tuples with material IDs and
       * a corresponding TableType.
       */
      Interface(const std::vector<std::tuple<EOS_INTEGER, TableType>> &tables)
      {
        n_tables_ = tables.size();

        std::transform(std::begin(tables),
                       std::end(tables),
                       std::back_inserter(material_ids_),
                       [](const auto &it) { return std::get<0>(it); });

        std::transform(std::begin(tables),
                       std::end(tables),
                       std::back_inserter(table_types_),
                       [](const auto &it) {
                         return static_cast<EOS_INTEGER>(std::get<1>(it));
                       });

        table_handles_.resize(n_tables_);

        /* create tables: */

        EOS_INTEGER error_code;
        eos_CreateTables(&n_tables_,
                         table_types_.data(),
                         material_ids_.data(),
                         table_handles_.data(),
                         &error_code);
        check_tables("eos_CreateTables");

        /* set table options: */

        for (EOS_INTEGER i = 0; i < n_tables_; i++) {
          // FIXME: refactor into options
          eos_SetOption(
              &table_handles_[i], &EOS_SMOOTH, EOS_NullPtr, &error_code);
          check_error_code(error_code, "eos_SetOption", i);
        }

        /* load tables: */

        eos_LoadTables(&n_tables_, table_handles_.data(), &error_code);
        check_tables("eos_LoadTables");
      }

      /**
       * Desctructor
       */
      ~Interface() noexcept
      {
        EOS_INTEGER error_code;
        eos_DestroyTables(&n_tables_, table_handles_.data(), &error_code);
      }

      /**
       * Query the table with index @p index. The interpolation works on @p
       * n tuples in parallel.
       */
      inline DEAL_II_ALWAYS_INLINE void
      interpolate_values(const EOS_INTEGER &index,
                         const dealii::ArrayView<EOS_REAL> &F,
                         const dealii::ArrayView<EOS_REAL> &dFx,
                         const dealii::ArrayView<EOS_REAL> &dFy,
                         const dealii::ArrayView<const EOS_REAL> &X,
                         const dealii::ArrayView<const EOS_REAL> &Y)
      {
        Assert(index >= 0 && index < n_tables_,
               dealii::ExcMessage("Table index out of range"));

        EOS_INTEGER n_queries = F.size();

#ifdef DEBUG
        const decltype(dFx.size()) size = n_queries;
        Assert(dFx.size() == size && dFy.size() == size && X.size() == size &&
                   Y.size() == size,
               dealii::ExcMessage("vector sizes do not match"));
#endif

        EOS_INTEGER error_code;
        eos_Interpolate(&table_handles_[index],
                        &n_queries,
                        const_cast<EOS_REAL *>(X.data()), /* sigh */
                        const_cast<EOS_REAL *>(Y.data()), /* sigh */
                        F.data(),
                        dFx.data(),
                        dFy.data(),
                        &error_code);
      }

    private:
      /**
       * Parameters and handles for eos_createTables:
       */
      std::vector<EOS_INTEGER> material_ids_;
      std::vector<EOS_INTEGER> table_types_;
      EOS_INTEGER n_tables_;
      /* Mutable so that we can call eospac functions from a const context. */
      mutable std::vector<EOS_INTEGER> table_handles_;

      /**
       * Error handling:
       */

      void check_error_code(
          EOS_INTEGER error_code,
          const std::string &routine,
          EOS_INTEGER i = std::numeric_limits<EOS_INTEGER>::max()) const
      {
        if (error_code != EOS_OK) {
          std::array<EOS_CHAR, EOS_MaxErrMsgLen> error_message;
          eos_GetErrorMessage(&error_code, error_message.data());

          std::stringstream exception_body;
          exception_body << "Error: " << routine;
          if (i != std::numeric_limits<EOS_INTEGER>::max())
            exception_body << " (table " << i << ")";
          exception_body << ": " << error_code << " - " << error_message.data()
                         << std::flush;
          AssertThrow(false, dealii::ExcMessage(exception_body.str()));
        }
      }

      void check_tables(const std::string &routine) const
      {
        for (EOS_INTEGER i = 0; i < n_tables_; i++) {
          EOS_INTEGER table_error_code = EOS_OK;
          eos_GetErrorCode(&table_handles_[i], &table_error_code);
          if (table_error_code != EOS_OK) {
            std::array<EOS_CHAR, EOS_MaxErrMsgLen> error_message;
            eos_GetErrorMessage(&table_error_code, error_message.data());

            std::stringstream exception_body;
            exception_body << "Error: " << routine << " (table " << i
                           << "): " << table_error_code << " - "
                           << error_message.data() << std::flush;

            AssertThrow(false, dealii::ExcMessage(exception_body.str()));
          }
        }
      }
    };
  } // namespace eospac
#endif

  namespace EquationOfStateLibrary
  {
    /**
     * A tabulated equation of state based on the EOSPAC6/Sesame
     * database.
     *
     * Units are:
     *        [rho] = kg / m^3
     *          [p] = Pa = Kg / m / s^2
     *          [e] = J / Kg = N m / Kg = m^2 / s^2
     *
     * @ingroup EulerEquations
     */
    class Sesame : public EquationOfState
    {
    public:
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

#ifdef WITH_EOSPAC
      Sesame(const std::string &subsection)
          : EquationOfState("sesame", subsection)
      {
        material_id_ = 5030;
        this->add_parameter(
            "material id", material_id_, "The Sesame Material ID");

        this->prefer_vector_interface_ = true;
      }


      double pressure(double rho, double e) const final
      {
        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        EOS_INTEGER index = 0;

        double p, p_drho, p_de;
        const double rho_scaled = rho / 1.0e3; // convert from Kg/m^3 to Mg/m^3
        const double e_scaled = e / 1.0e6;     // convert from J/kg to MJ/kg

        eospac_interface_->interpolate_values(
            index,
            dealii::ArrayView<double>(&p, 1),
            dealii::ArrayView<double>(&p_drho, 1),
            dealii::ArrayView<double>(&p_de, 1),
            dealii::ArrayView<const double>(&rho_scaled, 1),
            dealii::ArrayView<const double>(&e_scaled, 1));

        return 1.0e9 * p; // convert from GPa to Pa
      }


      void pressure(const dealii::ArrayView<double> &p,
                    const dealii::ArrayView<double> &rho,
                    const dealii::ArrayView<double> &e) const final
      {
        Assert(p.size() == rho.size() && rho.size() == e.size(),
               dealii::ExcMessage("vectors have different size"));

        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        EOS_INTEGER index = 0;

        /* FIXME: this is not reentrant... */
        thread_local static std::vector<double> p_drho;
        thread_local static std::vector<double> p_de;
        p_drho.resize(p.size());
        p_de.resize(p.size());

        // convert from Kg/m^3 to Mg/m^3
        std::transform(std::begin(rho),
                       std::end(rho),
                       std::begin(rho),
                       [](double rho) { return rho / 1.0e3; });

        // convert from J/kg to MJ/kg
        std::transform(std::begin(e), //
                       std::end(e),
                       std::begin(e),
                       [](auto e) { return e / 1.0e6; });

        eospac_interface_->interpolate_values(index,
                                              p,
                                              dealii::ArrayView<double>(p_drho),
                                              dealii::ArrayView<double>(p_de),
                                              rho,
                                              e);

        // convert from GPa to Pa
        std::transform(std::begin(p), //
                       std::end(p),
                       std::begin(p),
                       [](auto it) { return it * 1.0e9; });
      }


      double specific_internal_energy(double rho, double p) const final
      {
        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        EOS_INTEGER index = 1;

        double e, e_drho, e_dp;
        const double rho_scaled = rho / 1.0e3; // convert from Kg/M^3 to Mg/M^3
        const double p_scaled = p / 1.0e9;     // convert from Pa to GPa

        eospac_interface_->interpolate_values(
            index,
            dealii::ArrayView<double>(&e, 1),
            dealii::ArrayView<double>(&e_drho, 1),
            dealii::ArrayView<double>(&e_dp, 1),
            dealii::ArrayView<const double>(&rho_scaled, 1),
            dealii::ArrayView<const double>(&p_scaled, 1));

        return 1.0e6 * e; // convert from MJ/kg to J/kg
      }


      void
      specific_internal_energy(const dealii::ArrayView<double> &e,
                               const dealii::ArrayView<double> &rho,
                               const dealii::ArrayView<double> &p) const final
      {
        Assert(e.size() == rho.size() && rho.size() == p.size(),
               dealii::ExcMessage("vectors have different size"));

        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        EOS_INTEGER index = 1;

        /* FIXME: this is not reentrant... */
        thread_local static std::vector<double> e_drho;
        thread_local static std::vector<double> e_dp;
        e_drho.resize(e.size());
        e_dp.resize(e.size());

        // convert from Kg/m^3 to Mg/m^3
        std::transform(std::begin(rho),
                       std::end(rho),
                       std::begin(rho),
                       [](double rho) { return rho / 1.0e3; });

        // convert from Pa to GPa
        std::transform(std::begin(p), //
                       std::end(p),
                       std::begin(p),
                       [](auto it) { return it / 1.0e9; });

        eospac_interface_->interpolate_values(index,
                                              e,
                                              dealii::ArrayView<double>(e_drho),
                                              dealii::ArrayView<double>(e_dp),
                                              rho,
                                              p);

        // convert from MJ/kg to J/kg
        std::transform(std::begin(e), //
                       std::end(e),
                       std::begin(e),
                       [](auto e) { return e * 1.0e6; });
      }

      /* FIXME: Implement table look up for temperature. Need to think about
       * whether it should be T(rho, e) or T(rho, p). */

      double temperature(double /*rho*/, double /*e*/) const final
      {
        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        AssertThrow(false, dealii::ExcInternalError());
        __builtin_trap();
      }


      void temperature(const dealii::ArrayView<double> & /*temp*/,
                       const dealii::ArrayView<double> & /*rho*/,
                       const dealii::ArrayView<double> & /*e*/) const final
      {
        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        AssertThrow(false, dealii::ExcInternalError());
        __builtin_trap();
      }


      double speed_of_sound(double /*rho*/, double /*e*/) const final
      {
        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        AssertThrow(false, dealii::ExcInternalError());
        __builtin_trap();
      }


      void speed_of_sound(const dealii::ArrayView<double> & /*c*/,
                          const dealii::ArrayView<double> & /*rho*/,
                          const dealii::ArrayView<double> & /*e*/) const final
      {
        eospac_guard_.ensure_initialized([&]() {
          this->set_up_database();
          return true;
        });

        AssertThrow(false, dealii::ExcInternalError());
        __builtin_trap();
      }

    private:
      /**
       * Private methods and fields for initializing the eospac interface
       */
      //@{
      //
      void set_up_database() const
      {
        AssertThrow(
            std::filesystem::exists("sesameFilesDir.txt"),
            dealii::ExcMessage(
                "For EOSPAC to find the sesame database, we assume that there "
                "exists a file named 'sesameFilesDir.txt' in the current "
                "simulation directory. This file should list the path to the "
                "sesame database. See the EOSPAC manual for more "
                "information."));
        const std::vector<std::tuple<EOS_INTEGER, eospac::TableType>> tables{
            {material_id_, eospac::TableType::p_rho_e},
            {material_id_, eospac::TableType::e_rho_p},
        };

        eospac_interface_ = std::make_unique<eospac::Interface>(tables);
      }

      Lazy<bool> eospac_guard_;
      mutable std::unique_ptr<eospac::Interface> eospac_interface_;

      //@}
      /**
       * @name Run time options
       */
      //@{

      EOS_INTEGER material_id_;

      //@}

#else /* WITHOUT_EOSPAC */

      /* We do not have eospac support */
      Sesame(const std::string &subsection)
          : EquationOfState("Sesame", subsection)
      {
      }

      static constexpr auto message =
          "ryujin has to be configured with eospac support in order to use "
          "the Sesame EOS database";

      double pressure(double /*rho*/, double /*internal_energy*/) const final
      {
        AssertThrow(false, dealii::ExcMessage(message));
        __builtin_trap();
      }

      double specific_internal_energy(double /*rho*/, double /*p*/) const final
      {
        AssertThrow(false, dealii::ExcMessage(message));
        __builtin_trap();
      }

      double temperature(double /*rho*/, double /*e*/) const final
      {
        AssertThrow(false, dealii::ExcMessage(message));
        __builtin_trap();
      }

      double speed_of_sound(double /*rho*/, double /*e*/) const final
      {
        AssertThrow(false, dealii::ExcMessage(message));
        __builtin_trap();
      }
#endif
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
