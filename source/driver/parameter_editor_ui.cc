//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_animation.h>
#include <driver/dashboard_theme.h>
#include <ryujin/driver/dashboard.h>
#include <ryujin/driver/parameter_editor.h>

#include <deal.II/base/parameter_acceptor.h>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/terminal.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stop_token>
#include <thread>

namespace ryujin
{
  using namespace ftxui;
  using namespace dashboard;

  namespace
  {
    /*
     * The animation behind the editor:
     */

    class Backdrop
    {
    public:
      void configure(const DashboardSettings &settings)
      {
        if (!animation_ || settings.color_scheme != settings_.color_scheme ||
            settings.animation != settings_.animation) {
          theme = make_palette(settings.color_scheme);
          animation_ = animation::create(settings.animation);
          stage_.reset();
        }
        settings_ = settings;
      }

      const DashboardSettings &settings() const
      {
        return settings_;
      }

      /* The cells covered by the editor, from the last frame: */
      Box &window()
      {
        return window_;
      }

      void step()
      {
        if (!animation_ || !stage_)
          return;
        stage_->dashboard = window_;
        stage_->changed.clear();
        animation_->step(*stage_, data_);
      }

      void render(Screen &screen, const Box &box)
      {
        const int width = box.x_max - box.x_min + 1;
        const int height = box.y_max - box.y_min + 1;
        if (width <= 0 || height <= 0)
          return;

        if (!stage_ || stage_->screen.dimx() != width ||
            stage_->screen.dimy() != height) {
          stage_ = std::make_unique<animation::Stage>(
              Screen(width, height), window_, theme.animation);
          if (animation_)
            animation_->draw(*stage_, data_);
        }

        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x)
            screen.PixelAt(box.x_min + x, box.y_min + y) =
                stage_->screen.PixelAt(x, y);
      }

    private:
      DashboardSettings settings_;
      std::unique_ptr<animation::Animation> animation_;
      std::unique_ptr<animation::Stage> stage_;
      Box window_ = {0, -1, 0, -1};
      /* The animations do not use the data: */
      DashboardData data_;
    };


    class BackdropNode final : public Node
    {
    public:
      explicit BackdropNode(Backdrop &backdrop)
          : backdrop_(backdrop)
      {
      }

      void ComputeRequirement() override
      {
        requirement_ = {};
        requirement_.flex_grow_x = 1;
        requirement_.flex_grow_y = 1;
        requirement_.flex_shrink_x = 1;
        requirement_.flex_shrink_y = 1;
      }

      void Render(Screen &screen) override
      {
        backdrop_.render(screen, box_);
      }

    private:
      Backdrop &backdrop_;
    };


    /*
     * Place @p document on the backdrop with the width, the height and
     * the placement of the dashboard settings, at most the size of the
     * terminal.
     */
    Element compose(Backdrop &backdrop, Element document)
    {
      const auto &settings = backdrop.settings();
      const auto terminal = Terminal::Size();

      const int width =
          settings.width == 0
              ? terminal.dimx
              : std::min(terminal.dimx, std::max<int>(settings.width, 80));
      document = std::move(document) | size(WIDTH, EQUAL, width);

      const int height = std::min<int>(terminal.dimy, settings.height);
      if (height > 0)
        document = std::move(document) | size(HEIGHT, EQUAL, height);

      document = std::move(document) | size(HEIGHT, LESS_THAN, terminal.dimy) |
                 clear_under | reflect(backdrop.window());
      if (settings.center)
        document = center(std::move(document));
      else
        document = vbox({hbox({std::move(document), filler()}), filler()});

      return dbox({std::make_shared<BackdropNode>(backdrop), document});
    }


    /*
     * Run the event loop of @p screen. Advance the animation every 150
     * milliseconds, and hand all events that @p component does not handle
     * to @p fallback.
     */
    void loop(ScreenInteractive &screen,
              Component component,
              Backdrop &backdrop,
              std::function<bool(Event)> fallback)
    {
      class Impl : public ComponentBase
      {
      public:
        Impl(Component child,
             Backdrop &backdrop,
             std::function<bool(Event)> fallback)
            : backdrop_(backdrop)
            , fallback_(std::move(fallback))
        {
          Add(std::move(child));
        }

        bool OnEvent(Event event) override
        {
          if (event == Event::Custom) {
            backdrop_.step();
            return true;
          }
          return ComponentBase::OnEvent(event) || fallback_(event);
        }

      private:
        Backdrop &backdrop_;
        std::function<bool(Event)> fallback_;
      };

      std::mutex mutex;
      std::condition_variable_any condition;
      std::jthread ticker([&](std::stop_token token) {
        std::unique_lock lock(mutex);
        while (!condition.wait_for(
            lock, token, std::chrono::milliseconds(150), [&token] {
              return token.stop_requested();
            }))
          screen.PostEvent(Event::Custom);
      });

      screen.Loop(std::make_shared<Impl>(component, backdrop, fallback));

      ticker.request_stop();
    }


    /*
     * Elements and components:
     */

    Element header(const std::string &title, Elements right)
    {
      Elements elements = {
          badge(false),
          text(" "),
          text(" " + title + " ") | color(theme.ink) | bgcolor(theme.highlight),
          filler(),
      };
      for (auto &element : right)
        elements.push_back(std::move(element));
      elements.push_back(text(" "));

      return borderRounded(hbox(std::move(elements)) | color(Color::Default)) |
             color(theme.accent);
    }


    Component button(const std::string &name, std::function<void()> on_click)
    {
      ButtonOption option;
      option.label = name;
      option.on_click = std::move(on_click);
      option.transform = [](const EntryState &state) {
        auto element = text(" " + state.label + " ");
        if (state.focused)
          return element | bold | color(theme.ink) | bgcolor(theme.highlight);
        return element | color(theme.caption) | bgcolor(theme.track);
      };
      return Button(option);
    }


    /*
     * A single line input that leaves the arrow keys up and down to the
     * container, so that they always move to the previous or next entry:
     */
    Component single_line(Component input)
    {
      class Impl : public ComponentBase
      {
      public:
        explicit Impl(Component input)
        {
          Add(std::move(input));
        }

        bool OnEvent(Event event) override
        {
          if (event == Event::ArrowUp || event == Event::ArrowDown)
            return false;
          return ComponentBase::OnEvent(event);
        }
      };

      return std::make_shared<Impl>(std::move(input));
    }


    /*
     * A horizontal choice of a few entries, such as "on" and "off", with
     * the selected entry highlighted:
     */
    MenuOption choice_option()
    {
      MenuOption option = MenuOption::Horizontal();
      option.entries_option.transform = [](const EntryState &state) {
        auto element = text(" " + state.label + " ");
        if (state.active)
          element |= bold | color(theme.ink) | bgcolor(theme.highlight);
        else
          element |= color(theme.caption) | bgcolor(theme.track);
        return element;
      };
      option.elements_infix = [] { return text(""); };
      return option;
    }


    Element hints(const std::string &keys)
    {
      return text(" " + keys) | color(theme.muted);
    }


    /*
     * Read the settings of the dashboard from the subsection
     * "M - Logger", if present:
     */
    DashboardSettings read_settings(const ParameterModel &model)
    {
      DashboardSettings settings;

      const auto read = [&model](const std::string &name, auto &value) {
        using T = std::remove_reference_t<decltype(value)>;
        const auto entry = model.find({"M - Logger"}, name);
        if (entry == nullptr || !entry->valid())
          return;
        try {
          value = dealii::Patterns::Tools::Convert<T>::to_value(entry->value);
        } catch (...) {
          /* keep the default */
        }
      };

      read("dashboard width", settings.width);
      read("dashboard height", settings.height);
      read("dashboard center", settings.center);
      read("dashboard color scheme", settings.color_scheme);
      read("dashboard animation", settings.animation);
      return settings;
    }


    /*
     * Write the values of @p model to @p parameter_file, the sections in
     * use only. Return whether the file has been written:
     */
    bool write(const ParameterModel &model, const std::string &parameter_file)
    {
      auto &prm = dealii::ParameterAcceptor::prm;
      model.apply(prm);

      std::ostringstream stream;
      prm.print_parameters(stream,
                           dealii::ParameterHandler::ShortPRM |
                               dealii::ParameterHandler::KeepDeclarationOrder);
      std::ofstream file(parameter_file);
      file << model.remove_inactive_sections(stream.str());
      return static_cast<bool>(file);
    }


    std::string join(const std::vector<std::string> &path)
    {
      std::string result;
      for (const auto &name : path)
        result += (result.empty() ? "" : " / ") + name;
      return result;
    }


    const char *kind_name(ParameterModel::Entry::Kind kind)
    {
      static const char *const names[] = {
          "bool", "selection", "integer", "number", "text"};
      return names[static_cast<int>(kind)];
    }


    /* The width of the names: the longest name, within [12, 36]: */
    int name_width(const ParameterModel &model)
    {
      std::size_t width = 0;
      for (const auto &entry : model.entries())
        width = std::max(width, entry.name.size());
      return std::clamp<int>(width, 12, 36);
    }


    /*
     * The parameter editor:
     */

    class Editor
    {
    public:
      using Entry = ParameterModel::Entry;
      using Kind = Entry::Kind;

      Editor(ParameterModel &model, EditorSession &session)
          : model_(model)
          , session_(session)
          , screen_(ScreenInteractive::FullscreenAlternateScreen())
          , selected_(model.entries().size(), 0)
          , rows_(model.entries().size())
          , name_width_(name_width(model))
      {
        backdrop_.configure(read_settings(model_));
      }

      EditorOutcome run()
      {
        const auto &sections = model_.sections();

        Components pages;
        for (std::size_t i = 0; i < sections.size(); ++i) {
          const auto &section = sections[i];
          if (section.path == session_.section)
            section_ = static_cast<int>(i);

          std::string label(2 * (section.path.size() - 1), ' ');
          label += section.path.empty() ? "(top level)" : section.path.back();
          labels_.push_back(label);

          Components rows;
          for (const auto index : section.entries) {
            rows_[index] = row(index);
            rows.push_back(rows_[index]);
          }
          pages.push_back(Container::Vertical(std::move(rows)));
        }

        refresh();

        MenuOption option = MenuOption::Vertical();
        option.entries = &menu_labels_;
        option.selected = &menu_index_;
        option.focused_entry = &menu_focus_;
        option.on_change = [this] {
          section_ = static_cast<int>(visible_.at(menu_index_));
          session_.section = model_.sections().at(section_).path;
        };
        /*
         * Mark sections with invalid entries, and mute the sections not in
         * use (shown with "show all"), which are not written:
         */
        option.entries_option.transform = [this](const EntryState &state) {
          const auto section = visible_.at(state.index);
          auto element = text((state.active ? "> " : "  ") + state.label);
          if (!model_.active(section))
            element |= color(theme.muted);
          else if (model_.n_invalid(section) > 0)
            element |= color(theme.bad);
          if (state.focused)
            element |= inverted;
          if (state.active)
            element |= bold;
          return element;
        };
        auto menu = Menu(option);
        auto tab = Container::Tab(std::move(pages), &section_);

        CheckboxOption show_all = CheckboxOption::Simple();
        show_all.label = "show all sections";
        show_all.checked = &session_.show_all;
        show_all.on_change = [this] { refresh(); };

        auto buttons = Container::Horizontal({
            button("load", [this] { open_file_dialog("load", &Editor::load); }),
            button("save", [this] { save(); }),
            button("save as",
                   [this] { open_file_dialog("save as", &Editor::save_as); }),
            button("quit", [this] { discard(EditorOutcome::quit); }),
            Checkbox(show_all) | Renderer([](Element checkbox) {
              return hbox({text("   "), std::move(checkbox)});
            }),
        });

        auto layout = Container::Vertical({
            Container::Horizontal({menu, tab}),
            buttons,
        });

        auto renderer = Renderer(layout, [this, menu, tab, buttons] {
          return render(menu, tab, buttons);
        });

        auto dialog = Container::Horizontal({
            button("discard changes", [this] { exit(pending_); }),
            button("cancel", [this] { confirm_ = false; }),
        });
        auto confirmation = Renderer(dialog, [dialog] {
          return panel("unsaved changes",
                       vbox({
                           text("Discard the unsaved changes?"),
                           text(""),
                           dialog->Render(),
                       })) |
                 bgcolor(Color::Black);
        });

        InputOption file_option;
        file_option.content = &file_name_;
        file_option.cursor_position = &file_cursor_;
        file_option.multiline = false;
        file_option.on_enter = [this] { (this->*file_action_)(); };
        auto file_input = Input(file_option);

        auto file_buttons = Container::Horizontal({
            button("ok", [this] { (this->*file_action_)(); }),
            button("cancel", [this] { file_dialog_ = false; }),
        });
        auto file_dialog = Renderer(
            Container::Vertical({file_input, file_buttons}),
            [this, file_input, file_buttons] {
              return panel(file_title_,
                           vbox({
                               text("File name:"),
                               file_input->Render() | bgcolor(theme.track) |
                                   size(WIDTH, GREATER_THAN, 48),
                               text(""),
                               file_buttons->Render(),
                           })) |
                     bgcolor(Color::Black);
            });

        auto component = Modal(Modal(renderer, confirmation, &confirm_),
                               file_dialog,
                               &file_dialog_);

        /* Escape returns to the sections first, and quits from there: */
        loop(screen_, component, backdrop_, [this, menu](const Event &event) {
          if (event == Event::CtrlS && !file_dialog_ && !confirm_)
            save();
          else if (event == Event::CtrlA) {
            session_.show_all = !session_.show_all;
            refresh();
          } else if (event == Event::Escape && confirm_)
            confirm_ = false;
          else if (event == Event::Escape && file_dialog_)
            file_dialog_ = false;
          else if (event == Event::Escape && !menu->Focused())
            menu->TakeFocus();
          else if (event == Event::Escape)
            discard(EditorOutcome::quit);
          else
            return false;
          return true;
        });

        return outcome_;
      }

    private:
      /*
       * Update the list of sections in the menu: all sections with
       * "show all", and otherwise only the sections in use. Stay on the
       * current section, or move to its closest enclosing section in the
       * menu:
       */
      void refresh()
      {
        const auto &sections = model_.sections();

        visible_.clear();
        menu_labels_.clear();
        for (std::size_t i = 0; i < sections.size(); ++i) {
          if (session_.show_all || model_.active(i)) {
            visible_.push_back(i);
            menu_labels_.push_back(labels_[i]);
          }
        }

        auto it = visible_.end();
        for (std::optional<std::size_t> i = section_; i && it == visible_.end();
             i = sections[*i].parent)
          it = std::ranges::find(visible_, *i);
        if (it == visible_.end())
          it = visible_.begin();

        menu_index_ = static_cast<int>(it - visible_.begin());
        menu_focus_ = menu_index_;
        section_ = static_cast<int>(*it);
      }

      /* The editor row of entry @p index: */
      Component row(std::size_t index)
      {
        auto widget = this->widget(index);
        return Renderer(widget, [this, index, widget] {
          const auto &entry = model_.entries()[index];

          auto name = text(entry.name) | size(WIDTH, EQUAL, name_width_);
          name = entry.value == entry.default_value
                     ? std::move(name) | color(theme.caption)
                     : std::move(name) | bold;

          return hbox({
              text(entry.modified() ? "● " : "  ") | color(theme.highlight),
              name,
              text(" "),
              widget->Render() | flex,
              text(entry.valid() ? "  " : " ✗") | color(theme.bad),
          });
        });
      }

      /* The input component of entry @p index: */
      Component widget(std::size_t index)
      {
        auto &entry = model_.entries()[index];

        switch (entry.kind) {
        case Kind::boolean: {
          selected_[index] = entry.value == "true" ? 0 : 1;
          MenuOption option = choice_option();
          option.entries = &on_off_;
          option.selected = &selected_[index];
          option.on_change = [this, index] {
            auto &entry = model_.entries()[index];
            entry.value = entry.choices[selected_[index]];
            changed(index);
          };
          return Menu(option);
        }

        case Kind::selection: {
          const auto it = std::ranges::find(entry.choices, entry.value);
          selected_[index] = static_cast<int>(it - entry.choices.begin());

          DropdownOption option;
          option.radiobox.entries = &entry.choices;
          option.radiobox.selected = &selected_[index];
          option.radiobox.focused_entry = selected_[index];
          option.radiobox.on_change = [this, index] {
            auto &entry = model_.entries()[index];
            entry.value = entry.choices.at(selected_[index]);
            changed(index);
          };
          option.transform = [](bool open, Element checkbox, Element radiobox) {
            if (!open)
              return checkbox;
            return vbox({
                checkbox,
                std::move(radiobox) | vscroll_indicator | frame |
                    size(HEIGHT, LESS_THAN, 12) | borderRounded |
                    color(theme.accent),
            });
          };
          return Dropdown(option);
        }

        case Kind::integer:
        case Kind::floating:
        case Kind::text: {
          InputOption option;
          option.content = &entry.value;
          option.cursor_position = static_cast<int>(entry.value.size());
          option.multiline = false;
          option.on_change = [this, index] {
            filter(model_.entries()[index]);
            changed(index);
          };
          option.transform = [](InputState state) {
            if (state.focused)
              return state.element | bgcolor(theme.track);
            return state.element;
          };
          return single_line(Input(option));
        }
        }

        __builtin_unreachable();
      }

      /* Drop all characters a number cannot contain: */
      static void filter(Entry &entry)
      {
        if (entry.kind != Kind::integer && entry.kind != Kind::floating)
          return;

        const std::string allowed =
            entry.kind == Kind::integer ? "0123456789+-" : "0123456789+-.eE";
        std::erase_if(entry.value, [&allowed](char c) {
          return allowed.find(c) == std::string::npos;
        });
      }

      void changed(std::size_t index)
      {
        const auto &entry = model_.entries()[index];
        status_.clear();

        if (entry.section == std::vector<std::string>{"M - Logger"})
          backdrop_.configure(read_settings(model_));

        /* A selection may change the sections in use: */
        if (entry.kind == Kind::selection)
          refresh();

        /* A different equation or dimension declares other parameters: */
        if (entry.section == std::vector<std::string>{"B - Equation"} &&
            (entry.name == "equation" || entry.name == "dimension") &&
            entry.modified() && entry.valid())
          exit(EditorOutcome::rebuild);
      }

      void report(std::string message, bool good = false)
      {
        status_ = std::move(message);
        status_good_ = good;
      }

      void open_file_dialog(std::string title, void (Editor::*action)())
      {
        file_title_ = std::move(title);
        file_action_ = action;
        file_name_ = session_.parameter_file;
        file_cursor_ = static_cast<int>(file_name_.size());
        file_dialog_ = true;
      }

      void load()
      {
        file_dialog_ = false;
        if (!std::filesystem::exists(file_name_)) {
          report("The parameter file »" + file_name_ + "« does not exist.");
          return;
        }
        discard(EditorOutcome::load);
      }

      void save()
      {
        write_to(session_.parameter_file);
      }

      void save_as()
      {
        if (file_name_.empty())
          return;
        file_dialog_ = false;
        if (write_to(file_name_))
          session_.parameter_file = file_name_;
      }

      /* Write the parameters to @p parameter_file, the changes are saved: */
      bool write_to(const std::string &parameter_file)
      {
        if (model_.n_invalid() > 0) {
          report("Fix the invalid entries (✗) before saving.");
          return false;
        }
        if (!write(model_, parameter_file)) {
          report("Could not write the parameter file »" + parameter_file +
                 "«.");
          return false;
        }

        for (auto &entry : model_.entries())
          entry.initial_value = entry.value;
        session_.modified = false;
        report("Parameter file »" + parameter_file + "« written.", true);
        return true;
      }

      /* Exit with @p outcome, after a confirmation if there are unsaved
       * changes: */
      void discard(EditorOutcome outcome)
      {
        pending_ = outcome;
        if (session_.modified || model_.modified())
          confirm_ = true;
        else
          exit(outcome);
      }

      void exit(EditorOutcome outcome)
      {
        if (outcome == EditorOutcome::rebuild)
          session_.modified |= model_.modified();
        if (outcome == EditorOutcome::load) {
          session_.parameter_file = file_name_;
          session_.modified = false;
        }
        outcome_ = outcome;
        screen_.Exit();
      }

      Element documentation() const
      {
        /* The active row of the section shown, also while the menu has
         * the focus: */
        const auto &indices = model_.sections().at(section_).entries;
        const auto it =
            std::ranges::find_if(indices, [this](std::size_t index) {
              return rows_[index]->Active();
            });
        if (it == indices.end())
          return text("");

        const auto &entry = model_.entries()[*it];
        return vbox({
            hbox({
                text(entry.name) | bold,
                text("  "),
                chip(kind_name(entry.kind), theme.highlight),
            }),
            paragraph(entry.documentation.empty() ? "(no documentation)"
                                                  : entry.documentation),
            hbox({label("pattern"),
                  paragraph(entry.pattern_description) | color(theme.muted)}),
            hbox({label("default"),
                  text(entry.default_value) | color(theme.muted)}),
        });
      }

      Element render(const Component &menu,
                     const Component &tab,
                     const Component &buttons)
      {
        const auto invalid = model_.n_invalid();

        const auto equation = model_.find({"B - Equation"}, "equation");
        const auto dimension = model_.find({"B - Equation"}, "dimension");

        Elements right;
        if (equation != nullptr && dimension != nullptr) {
          right.push_back(text("equation ") | color(theme.caption));
          right.push_back(text(equation->value) | bold);
          right.push_back(text("   dimension ") | color(theme.caption));
          right.push_back(text(dimension->value) | bold);
        }
        right.push_back(text("   file ") | color(theme.caption));
        right.push_back(text(session_.parameter_file) | bold);

        const auto &section = model_.sections().at(section_);
        Element entries = tab->Render();
        if (section.entries.empty())
          entries = text("This section has no parameters of its own.") |
                    color(theme.muted);

        Element status = text("");
        if (!status_.empty())
          status = text(status_ + " ") |
                   color(status_good_ ? theme.good : theme.bad);
        else if (invalid > 0)
          status =
              text(std::format(
                  "{} invalid entr{} ", invalid, invalid == 1 ? "y" : "ies")) |
              color(theme.bad);
        else if (session_.modified || model_.modified())
          status = text("unsaved changes ") | color(theme.warn);

        /* Fit the height to the sections in the menu: */
        const auto height = static_cast<int>(visible_.size()) + 2 /* borders */;

        auto document = vbox({
            header("parameters", std::move(right)),
            hbox({
                panel("sections",
                      menu->Render() | vscroll_indicator | frame | flex) |
                    size(WIDTH, EQUAL, 32),
                panel(join(section.path),
                      std::move(entries) | vscroll_indicator | frame | flex) |
                    flex,
            }) | size(HEIGHT, GREATER_THAN, height) |
                flex,
            panel("documentation", documentation()) | size(HEIGHT, EQUAL, 13),
            hbox({
                text(" "),
                buttons->Render(),
                filler(),
                std::move(status),
            }),
            hints("arrows: move   enter: select   esc: back/quit   "
                  "ctrl-s: save   ctrl-a: all sections"),
        });

        return compose(backdrop_, std::move(document));
      }

      ParameterModel &model_;
      EditorSession &session_;
      ScreenInteractive screen_;
      Backdrop backdrop_;

      /* The selected choice of all booleans and selections: */
      std::vector<int> selected_;
      /* The editor rows of all entries: */
      Components rows_;
      const std::vector<std::string> on_off_ = {"on", "off"};

      /* The labels of all sections: */
      std::vector<std::string> labels_;
      /* The sections in the menu, and their labels: */
      std::vector<std::size_t> visible_;
      std::vector<std::string> menu_labels_;
      /* The selected entry of the menu, and the entry with the cursor: */
      int menu_index_ = 0;
      int menu_focus_ = 0;
      /* The section shown, an index into ParameterModel::sections(): */
      int section_ = 0;
      const int name_width_;

      std::string status_;
      bool status_good_ = false;

      /* The confirmation to discard the unsaved changes, and the outcome
       * to exit with: */
      bool confirm_ = false;
      EditorOutcome pending_ = EditorOutcome::quit;

      /* The dialogs "load" and "save as" for a file name: */
      bool file_dialog_ = false;
      std::string file_title_;
      void (Editor::*file_action_)() = nullptr;
      std::string file_name_;
      int file_cursor_ = 0;
      EditorOutcome outcome_ = EditorOutcome::quit;
    };
  } // namespace


  EditorOutcome run_parameter_editor(ParameterModel &model,
                                     EditorSession &session)
  {
    Editor editor(model, session);
    return editor.run();
  }
} // namespace ryujin
