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
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <thread>

namespace ryujin
{
  using namespace ftxui;
  using namespace dashboard;

  namespace
  {
    /* The animation behind the editor, outside of the editor window: */
    class Backdrop : public Node
    {
    public:
      /* Start at the initial position, the center: */
      Backdrop()
      {
        position_.postpone();
      }

      /* Take over the size, location, colors, and animation of @p settings: */
      void configure(const DashboardSettings &settings)
      {
        settings_ = settings;
        theme = make_palette(settings.color_scheme);
        animation_ = animation::create(settings.animation);
        stage_.reset();
      }

      /* Size and place @p window on the terminal as configured: */
      Element place(Element window)
      {
        const auto [columns, rows] = Terminal::Size();
        const int width =
            settings_.width == 0
                ? columns
                : std::min(columns, std::max<int>(settings_.width, 80));

        window |= size(WIDTH, EQUAL, width);
        if (settings_.height > 0)
          window |= size(HEIGHT, EQUAL, std::min<int>(settings_.height, rows));
        window |= size(HEIGHT, LESS_THAN, rows);

        const int height = Dimension::Fit(window, true).dimy;
        const auto [left, top] = position_.offset(
            settings_.location, columns - width, rows - height);
        if (const Box box{left, left + width - 1, top, top + height - 1};
            box != window_) {
          window_ = box;
          stage_.reset();
        }

        return dashboard::place(std::move(window) | clear_under, left, top);
      }

      /* Advance the animation, and move while there is no input: */
      void step()
      {
        position_.move(settings_.location);
        if (animation_ && stage_) {
          stage_->changed.clear();
          animation_->step(*stage_);
        }
      }

      /* Postpone the next move of the screen saver: */
      void postpone()
      {
        position_.postpone();
      }

      /* Draw the animation on the whole screen: */
      void Render(Screen &screen) override
      {
        if (!stage_ || stage_->screen.dimx() != screen.dimx() ||
            stage_->screen.dimy() != screen.dimy()) {
          stage_.emplace(Screen(screen.dimx(), screen.dimy()), window_);
          if (animation_)
            animation_->draw(*stage_);
        }

        for (int y = 0; y < screen.dimy(); ++y)
          for (int x = 0; x < screen.dimx(); ++x)
            screen.PixelAt(x, y) = stage_->screen.PixelAt(x, y);
      }

    private:
      DashboardSettings settings_;
      DashboardPosition position_;
      Box window_;
      std::unique_ptr<animation::Animation> animation_;
      std::optional<animation::Stage> stage_;
    };


    /* A button or a choice, highlighted if @p on: */
    Element toggle(const std::string &label, bool on)
    {
      auto element = text(" " + label + " ");
      if (on)
        return element | bold | color(theme.ink) | bgcolor(theme.highlight);
      return element | color(theme.caption) | bgcolor(theme.track);
    }


    /* A button @p name that calls @p on_click: */
    Component button(const std::string &name, std::function<void()> on_click)
    {
      ButtonOption option;
      option.label = name;
      option.on_click = std::move(on_click);
      option.transform = [](const EntryState &state) {
        return toggle(state.label, state.focused);
      };
      return Button(option);
    }


    /* A single line input that leaves the arrow keys up and down alone: */
    class SingleLine : public ComponentBase
    {
    public:
      /* Wrap the input component @p input: */
      explicit SingleLine(Component input)
      {
        Add(std::move(input));
      }

      /* Hand up and down to the container, everything else to the input: */
      bool OnEvent(Event event) override
      {
        return event != Event::ArrowUp && event != Event::ArrowDown &&
               ComponentBase::OnEvent(event);
      }
    };


    /* Read the dashboard settings from the subsection "M - Logger": */
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
      read("dashboard location", settings.location);
      read("dashboard color scheme", settings.color_scheme);
      read("dashboard animation", settings.animation);
      return settings;
    }


    /* Write the sections of @p model in use to @p parameter_file: */
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


    /* The parameter editor: */
    class Editor : public ComponentBase
    {
    public:
      using Entry = ParameterModel::Entry;
      using Kind = Entry::Kind;

      /* Build all components for the sections and entries of @p model: */
      Editor(ParameterModel &model,
             EditorSession &session,
             ScreenInteractive &screen)
          : model_(model)
          , session_(session)
          , screen_(screen)
          , selected_(model.entries().size())
          , rows_(model.entries().size())
      {
        backdrop_->configure(read_settings(model_));

        for (const auto &entry : model_.entries())
          name_width_ = std::max<int>(name_width_, entry.name.size());
        name_width_ = std::min(name_width_, 36);

        Components pages;
        for (const auto &section : model_.sections()) {
          if (section.path == session_.section)
            section_ = static_cast<int>(labels_.size());

          const auto &path = section.path;
          labels_.push_back(path.empty()
                                ? "(top level)"
                                : std::string(2 * path.size() - 2, ' ') +
                                      path.back());

          Components rows;
          for (const auto index : section.entries)
            rows.push_back(rows_[index] = row(index));
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
        menu_ = Menu(option);
        tab_ = Container::Tab(std::move(pages), &section_);

        CheckboxOption show_all = CheckboxOption::Simple();
        show_all.label = "show all sections";
        show_all.checked = &session_.show_all;
        show_all.on_change = [this] { refresh(); };

        buttons_ = Container::Horizontal({
            button("load", open("load", &Editor::load)),
            button("save", [this] { save(session_.parameter_file); }),
            button("save as", open("save as", &Editor::save_as)),
            button("quit", [this] { discard(EditorOutcome::quit); }),
            Checkbox(show_all) | Renderer([](Element checkbox) {
              return hbox({text("   "), std::move(checkbox)});
            }),
        });

        auto main = Renderer(Container::Vertical({
                                 Container::Horizontal({menu_, tab_}),
                                 buttons_,
                             }),
                             [this] { return render(); });

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

        Add(main | Modal(confirmation, &confirm_) |
            Modal(file_dialog, &file_dialog_));
      }

      /* Animate on ticks, and handle the shortcuts the children leave: */
      bool OnEvent(Event event) override
      {
        if (event == Event::Custom) {
          backdrop_->step();
          return true;
        }

        backdrop_->postpone();
        if (ComponentBase::OnEvent(event))
          return true;

        if (event == Event::CtrlS && !file_dialog_ && !confirm_)
          save(session_.parameter_file);
        else if (event == Event::CtrlA) {
          session_.show_all = !session_.show_all;
          refresh();
        } else if (event == Event::Escape && (confirm_ || file_dialog_))
          confirm_ = file_dialog_ = false;
        else if (event == Event::Escape && !menu_->Focused())
          menu_->TakeFocus();
        else if (event == Event::Escape)
          discard(EditorOutcome::quit);
        else
          return false;
        return true;
      }

      /* How the editor has been left: */
      EditorOutcome outcome() const
      {
        return outcome_;
      }

    private:
      /* Update the sections in the menu, and stay as close as possible: */
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

        menu_index_ = menu_focus_ = static_cast<int>(it - visible_.begin());
        section_ = static_cast<int>(*it);
      }

      /* The row of entry @p index: marker, name, widget, and validity: */
      Component row(std::size_t index)
      {
        auto widget = this->widget(index);
        return Renderer(widget, [this, index, widget] {
          const auto &entry = model_.entries()[index];
          const bool changed = entry.value != entry.default_value;

          return hbox({
              text(entry.modified() ? "● " : "  ") | color(theme.highlight),
              (changed ? text(entry.name) | bold : caption(entry.name)) |
                  size(WIDTH, EQUAL, name_width_),
              text(" "),
              widget->Render() | flex,
              text(entry.valid() ? "  " : " ✗") | color(theme.bad),
          });
        });
      }

      /* The input component of entry @p index, depending on its kind: */
      Component widget(std::size_t index)
      {
        auto &entry = model_.entries()[index];

        const auto on_change = [this, index] {
          auto &entry = model_.entries()[index];
          if (entry.kind == Kind::boolean || entry.kind == Kind::selection)
            entry.value = entry.choices.at(selected_[index]);
          changed(entry);
        };

        switch (entry.kind) {
        case Kind::boolean: {
          static const std::vector<std::string> on_off = {"on", "off"};
          selected_[index] = entry.value == "true" ? 0 : 1;

          MenuOption option = MenuOption::Horizontal();
          option.entries = &on_off;
          option.selected = &selected_[index];
          option.on_change = on_change;
          option.entries_option.transform = [](const EntryState &state) {
            return toggle(state.label, state.active);
          };
          option.elements_infix = [] { return text(""); };
          return Menu(option);
        }

        case Kind::selection: {
          const auto it = std::ranges::find(entry.choices, entry.value);
          selected_[index] = static_cast<int>(it - entry.choices.begin());

          DropdownOption option;
          option.radiobox.entries = &entry.choices;
          option.radiobox.selected = &selected_[index];
          option.radiobox.focused_entry = selected_[index];
          option.radiobox.on_change = on_change;
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
          option.on_change = on_change;
          option.transform = [](InputState state) {
            return state.focused ? state.element | bgcolor(theme.track)
                                 : state.element;
          };
          return std::make_shared<SingleLine>(Input(option));
        }
        }

        __builtin_unreachable();
      }

      /* Filter numbers, and update everything that depends on @p entry: */
      void changed(Entry &entry)
      {
        status_.clear();

        if (entry.kind == Kind::integer || entry.kind == Kind::floating) {
          const std::string_view allowed =
              entry.kind == Kind::integer ? "0123456789+-" : "0123456789+-.eE";
          std::erase_if(entry.value, [&](char c) {
            return allowed.find(c) == std::string_view::npos;
          });
        }

        if (entry.section == std::vector<std::string>{"M - Logger"})
          backdrop_->configure(read_settings(model_));

        if (entry.kind == Kind::selection)
          refresh();

        if (entry.section == std::vector<std::string>{"B - Equation"} &&
            (entry.name == "equation" || entry.name == "dimension") &&
            entry.modified() && entry.valid())
          exit(EditorOutcome::rebuild);
      }

      /* Show @p message in the status line, in green if @p good: */
      void report(std::string message, bool good = false)
      {
        status_ = std::move(message);
        status_good_ = good;
      }

      /* The button action that opens the file dialog for @p action: */
      std::function<void()> open(std::string title, void (Editor::*action)())
      {
        return [this, title, action] {
          file_title_ = title;
          file_action_ = action;
          file_name_ = session_.parameter_file;
          file_cursor_ = static_cast<int>(file_name_.size());
          file_dialog_ = true;
        };
      }

      /* Load the parameter file of the file dialog: */
      void load()
      {
        file_dialog_ = false;
        if (std::filesystem::exists(file_name_))
          discard(EditorOutcome::load);
        else
          report("The parameter file »" + file_name_ + "« does not exist.");
      }

      /* Save to the parameter file of the file dialog: */
      void save_as()
      {
        if (file_name_.empty())
          return;
        file_dialog_ = false;
        if (save(file_name_))
          session_.parameter_file = file_name_;
      }

      /* Write @p parameter_file, return whether it has been written: */
      bool save(const std::string &parameter_file)
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

      /* Exit with @p outcome, after confirming to drop unsaved changes: */
      void discard(EditorOutcome outcome)
      {
        pending_ = outcome;
        if (session_.modified || model_.modified())
          confirm_ = true;
        else
          exit(outcome);
      }

      /* Exit with @p outcome right away: */
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

      /* The documentation of the active row of the section shown: */
      Element documentation() const
      {
        static const char *const kinds[] = {
            "bool", "selection", "integer", "number", "text"};

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
                chip(kinds[static_cast<int>(entry.kind)], theme.highlight),
            }),
            paragraph(entry.documentation.empty() ? "(no documentation)"
                                                  : entry.documentation),
            hbox({label("pattern"),
                  paragraph(entry.pattern_description) | color(theme.muted)}),
            hbox({label("default"), muted(entry.default_value)}),
        });
      }

      /* The editor window on the backdrop: */
      Element render()
      {
        Elements header = {
            badge(false),
            text(" "),
            text(" parameters ") | color(theme.ink) | bgcolor(theme.highlight),
            filler(),
        };
        for (const auto name : {"equation", "dimension"})
          if (const auto entry = model_.find({"B - Equation"}, name)) {
            header.push_back(caption(std::format("{} ", name)));
            header.push_back(text(entry->value + "   ") | bold);
          }
        header.push_back(caption("file "));
        header.push_back(text(session_.parameter_file + " ") | bold);

        const auto &section = model_.sections().at(section_);
        std::string path;
        for (const auto &name : section.path)
          path += (path.empty() ? "" : " / ") + name;

        Element entries = tab_->Render();
        if (section.entries.empty())
          entries = muted("This section has no parameters of its own.");

        const auto invalid = model_.n_invalid();
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

        auto document = vbox({
            borderRounded(hbox(std::move(header)) | color(Color::Default)) |
                color(theme.accent),
            hbox({
                panel("sections",
                      menu_->Render() | vscroll_indicator | frame | flex) |
                    size(WIDTH, EQUAL, 32),
                panel(path,
                      std::move(entries) | vscroll_indicator | frame | flex) |
                    flex,
            }) | /* Fit the height to the sections in the menu: */
                size(HEIGHT, GREATER_THAN, int(visible_.size()) + 2) |
                flex,
            panel("documentation", documentation()) | size(HEIGHT, EQUAL, 13),
            hbox({text(" "), buttons_->Render(), filler(), std::move(status)}),
            muted(" arrows: move   enter: select   esc: back/quit   "
                  "ctrl-s: save   ctrl-a: all sections"),
        });

        return dbox({backdrop_, backdrop_->place(std::move(document))});
      }

      ParameterModel &model_;
      EditorSession &session_;
      ScreenInteractive &screen_;
      std::shared_ptr<Backdrop> backdrop_ = std::make_shared<Backdrop>();

      Component menu_;
      Component tab_;
      Component buttons_;

      std::vector<int> selected_;
      Components rows_;
      int name_width_ = 12;

      std::vector<std::string> labels_;
      std::vector<std::size_t> visible_;
      std::vector<std::string> menu_labels_;
      int menu_index_ = 0;
      int menu_focus_ = 0;
      int section_ = 0;

      std::string status_;
      bool status_good_ = false;

      bool confirm_ = false;
      EditorOutcome pending_ = EditorOutcome::quit;
      EditorOutcome outcome_ = EditorOutcome::quit;

      bool file_dialog_ = false;
      std::string file_title_;
      void (Editor::*file_action_)() = nullptr;
      std::string file_name_;
      int file_cursor_ = 0;
    };
  } // namespace


  /* Run the editor on the full screen until it is left: */
  EditorOutcome run_parameter_editor(ParameterModel &model,
                                     EditorSession &session)
  {
    auto screen = ScreenInteractive::FullscreenAlternateScreen();
    auto editor = std::make_shared<Editor>(model, session, screen);

    /* Advance the animation every 150 milliseconds: */
    std::jthread ticker([&screen](std::stop_token token) {
      for (; !token.stop_requested(); screen.PostEvent(Event::Custom))
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    });

    screen.Loop(editor);
    return editor->outcome();
  }
} // namespace ryujin
