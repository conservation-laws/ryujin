//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/environment.h>
#include <ryujin/driver/equation.h>
#include <ryujin/driver/parameter_model.h>
#include <ryujin/driver/simulation.h>
#include <ryujin/driver/task_composition.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>

#include <deal.II/base/parameter_acceptor.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

//
// Test the ParameterModel of the parameter editor: read all parameters of
// the default TaskComposition, print the kind of a few entries, change
// values, and write them back into the ParameterHandler.
//

using namespace ryujin;

using HyperbolicDescription = Euler::HyperbolicDescription;
using ParabolicDescription = StubParabolicDescription;
constexpr int dim = 2;
using Number = NUMBER;


void print(const ParameterModel::Entry &entry)
{
  static const char *const kinds[] = {
      "boolean", "selection", "integer", "floating", "text"};

  std::cout << entry.name << ": " << kinds[static_cast<int>(entry.kind)]
            << ", value = " << entry.value << ", valid = " << entry.valid();
  if (!entry.choices.empty()) {
    std::cout << ", choices =";
    for (const auto &choice : entry.choices)
      std::cout << " »" << choice << "«";
  }
  std::cout << std::endl;
}


int main(int argc, char *argv[])
{
  Environment environment(argc, argv);
  std::cout << std::boolalpha;

  {
    [[maybe_unused]] const auto objects = populate_parameters([] {
      struct Objects {
        Simulation<HyperbolicDescription, ParabolicDescription, dim, Number>
            simulation{MPI_COMM_SELF};
        TaskComposition<dim, Number> task_composition{simulation};
      };
      return std::make_unique<Objects>();
    });

    auto &prm = dealii::ParameterAcceptor::prm;
    ParameterModel model(prm);

    std::cout << "first sections:" << std::endl;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto &section = model.sections()[i];
      std::cout << "  " << section.path.size() << " " << section.path.back()
                << " (" << section.entries.size() << " entries)" << std::endl;
    }

    const auto get = [&model](const std::string &section,
                              const std::string &name) {
      const auto entry = model.find({section}, name);
      AssertThrow(entry != nullptr, dealii::ExcMessage(name));
      return entry;
    };

    print(*get("M - Logger", "dashboard location"));
    print(*get("M - Logger", "dashboard color scheme"));
    print(*get("M - Logger", "dashboard max width"));
    print(*get("H - TimeIntegrator", "time stepping scheme"));
    print(*get("H - TimeIntegrator", "cfl max"));
    print(*get("A - Scheduler", "basename"));

    std::cout << "invalid: " << model.n_invalid()
              << ", modified: " << model.modified() << std::endl;

    /* Change values: */
    get("M - Logger", "dashboard location")->value = "top left";
    get("M - Logger", "dashboard color scheme")->value = "tamu";
    get("H - TimeIntegrator", "cfl max")->value = "0.25";
    get("A - Scheduler", "basename")->value = "my test";

    const auto invalid = get("M - Logger", "dashboard max width");
    invalid->value = "wide";
    print(*invalid);
    std::cout << "invalid: " << model.n_invalid()
              << ", modified: " << model.modified() << std::endl;
    invalid->value = invalid->initial_value;

    model.apply(prm);

    std::ostringstream stream;
    prm.print_parameters(stream, dealii::ParameterHandler::ShortPRM);

    /* Read the values back in with a fresh model: */
    ParameterModel result(prm);
    for (const auto &[section, name] :
         {std::pair{"M - Logger", "dashboard location"},
          std::pair{"M - Logger", "dashboard color scheme"},
          std::pair{"H - TimeIntegrator", "cfl max"},
          std::pair{"A - Scheduler", "basename"}})
      std::cout << name << " = " << result.find({section}, name)->value
                << std::endl;

    std::cout << "written: "
              << (stream.str().find("set basename") != std::string::npos)
              << std::endl;

    /* Read values with invalid values and an unknown entry: */
    const std::string text = "subsection C - Discretization\n"
                             "  set geometry = cylindr\n"
                             "end\n"
                             "subsection M - Logger\n"
                             "  set dashboard max width = wide # a comment\n"
                             "  set unknown entry   = 1\n"
                             "end\n"
                             "subsection H - TimeIntegrator\n"
                             "  set cfl max = \\\n"
                             "      0.5\n"
                             "end\n";
    const auto values = read_parameter_values(text);
    std::cout << "read " << values.size() << " values" << std::endl;

    ParameterModel strict(prm);
    strict.assign(values, /* keep invalid */ false);
    print(*strict.find({"M - Logger"}, "dashboard max width"));
    print(*strict.find({"H - TimeIntegrator"}, "cfl max"));

    ParameterModel lenient(prm);
    lenient.assign(values);
    print(*lenient.find({"M - Logger"}, "dashboard max width"));
    std::cout << lenient.find({"C - Discretization"}, "geometry")->value
              << std::endl;
    std::cout << "modified: " << lenient.modified() << std::endl;

    /*
     * The subsections of "C - Discretization" are selected by the entry
     * "geometry". Print the active ones, and the subsections that are
     * written:
     */
    const auto print_active = [](const ParameterModel &model) {
      std::cout << "active:";
      const auto &sections = model.sections();
      for (std::size_t i = 0; i < sections.size(); ++i)
        if (sections[i].path.size() > 1 &&
            sections[i].path.front() == "C - Discretization" && model.active(i))
          std::cout << " »" << sections[i].path.back() << "«";
      std::cout << std::endl;
    };

    ParameterModel geometry(prm);
    print(*geometry.find({"C - Discretization"}, "geometry"));
    print_active(geometry);

    geometry.find({"C - Discretization"}, "geometry")->value = "disk";
    print_active(geometry);

    /* An invalid value in a section not in use does not count: */
    geometry.find({"C - Discretization", "annulus"}, "inner radius")->value =
        "x";
    std::cout << "invalid: " << geometry.n_invalid() << std::endl;

    std::ostringstream full;
    prm.print_parameters(full, dealii::ParameterHandler::ShortPRM);
    std::istringstream written(geometry.remove_inactive_sections(full.str()));
    std::string line;
    while (std::getline(written, line))
      if (line.starts_with("  subsection "))
        std::cout << "written:" << line << std::endl;
  }

  dealii::ParameterAcceptor::clear();

  return 0;
}
