//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "equation_of_state.h"

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>

#ifdef WITH_EOSPAC
#include "eos_Interface.h"
#endif


namespace ryujin
{
  /**
   * A namespace with wrappers for the eospac6 library and sesame database.
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
      template <std::size_t n>
      inline DEAL_II_ALWAYS_INLINE std::array<std::array<EOS_REAL, n>, 3>
      interpolate_values(const EOS_INTEGER &index,
                         const std::array<EOS_REAL, n> &X,
                         const std::array<EOS_REAL, n> &Y)
      {
        Assert(index >= 0 && index < n_tables_,
               dealii::ExcMessage("Table index out of range"));

        EOS_INTEGER n_queries = n;

        std::array<EOS_REAL, n> F;
        std::array<EOS_REAL, n> dFx;
        std::array<EOS_REAL, n> dFy;

        EOS_INTEGER error_code;
        eos_Interpolate(&table_handles_[index],
                        &n_queries,
                        const_cast<EOS_REAL *>(X.data()), /* sigh */
                        const_cast<EOS_REAL *>(Y.data()), /* sigh */
                        F.data(),
                        dFx.data(),
                        dFy.data(),
                        &error_code);
        return {F, dFx, dFy};
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


  namespace EulerAEOS
  {
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
#ifdef WITH_EOSPAC
        Sesame(const std::string &subsection)
            : EquationOfState("sesame", subsection)
        {
          material_id_ = 5030;
          this->add_parameter(
              "material id", material_id_, "The Sesame Material ID");

          const auto set_up_database = [&]() {
            const std::vector<std::tuple<EOS_INTEGER, eospac::TableType>>
                tables{
                    {material_id_, eospac::TableType::p_rho_e},
                    {material_id_, eospac::TableType::e_rho_p},
                };

            eospac_interface_ = std::make_unique<eospac::Interface>(tables);
          };

          this->parse_parameters_call_back.connect(set_up_database);
        }

        double pressure(const double rho, const double e) final
        {
          EOS_INTEGER index = 0;
          const auto &[p, p_drho, p_de] =
              eospac_interface_->interpolate_values<1>(
                  index, {rho / 1.e3}, {e / 1.e6});
          return 1.e9 * p[0];
        }

        double specific_internal_energy(const double rho, const double p) final
        {
          EOS_INTEGER index = 1;
          const auto &[e, e_drho, e_dp] =
              eospac_interface_->interpolate_values<1>(
                  index, {rho / 1.e3}, {p / 1.e9});
          return 1.e6 * e[0];
        }

        double material_sound_speed(const double /*rho*/,
                                    const double /*p*/) final
        {
          __builtin_trap();
        }

      private:
        EOS_INTEGER material_id_;
        std::unique_ptr<eospac::Interface> eospac_interface_;

#else /* WITH_EOSPAC */

        /* We do not have eospac support */
        Sesame(const std::string &subsection)
            : EquationOfState("Sesame", subsection)
        {
        }

        static constexpr auto message =
            "ryujin has to be configured with eospac support in order to use "
            "the Sesame EOS database";

        double pressure(const double /*rho*/,
                        const double /*internal_energy*/) final
        {
          AssertThrow(false, dealii::ExcMessage(message));
          __builtin_trap();
        }

        double specific_internal_energy(const double /*rho*/,
                                        const double /*p*/) final
        {
          AssertThrow(false, dealii::ExcMessage(message));
          __builtin_trap();
        }

        double material_sound_speed(const double /*rho*/,
                                    const double /*p*/) final
        {
          AssertThrow(false, dealii::ExcMessage(message));
          __builtin_trap();
        }
#endif
      };
    } // namespace EquationOfStateLibrary
  }   // namespace EulerAEOS

} // namespace ryujin
