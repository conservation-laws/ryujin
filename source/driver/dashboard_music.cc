//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/dashboard_music.h>

#include <deal.II/base/exceptions.h>

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numbers>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ryujin
{
  namespace
  {
    /*
     * A tune: bars of sixteen steps (sixteenth notes). Every bar has a
     * chord of four notes, the lowest one being the root, and a lead line
     * of sixteen steps, where a note is written as in "F#5", "-" holds the
     * previous note, and "." rests.
     *
     * The bass line, the arpeggio, and the drums repeat the same pattern
     * of sixteen steps in every bar. In the patterns of the bass line and
     * the arpeggio, the digits '0' to '3' play the notes of the chord, and
     * '4' to '7' the same notes an octave higher; the bass line sounds two
     * octaves below, the arpeggio one octave below the chord. In the drum
     * patterns, 'x' strikes. In all patterns, '.' rests.
     */

    struct Bar {
      const char *chord;
      const char *lead;
    };

    struct Tune {
      /* Beats (quarter notes) per minute: */
      float tempo;
      std::span<const Bar> bars;
      std::string_view bass;
      std::string_view arpeggio;
      std::string_view kick;
      std::string_view snare;
      std::string_view hihat;
      /*
       * The frequency of the modulator of the lead relative to the
       * carrier: 1 is a bright brass, 2 a hollow square wave, 3 a glassy
       * bell.
       */
      float lead_ratio;
    };


    /*
     * DashboardMusic::ryujin: a sunny ride along the coast at 132 beats
     * per minute in D major.
     */

    /* clang-format off */
    constexpr std::array<Bar, 16> ryujin_bars{{
        {"G4 B4 D5 F#5",  "F#5 - - A5 . B5 - - D6 - B5 - A5 - F#5 -"}, /* Gmaj7 */
        {"F#4 A4 C#5 E5", "E5 - - - - - F#5 A5 - - C#6 - - B5 - A5"},  /* F#m7 */
        {"E4 G4 B4 D5",   "B5 - - G5 - - E5 - F#5 - G5 - B5 - D6 -"},  /* Em7 */
        {"A4 C#5 E5 G5",  "C#6 - - - - - - - . . A5 B5 C#6 - E6 -"},   /* A7 */
        {"G4 B4 D5 F#5",  "F#6 - - E6 - - D6 - . D6 E6 - D6 - B5 -"},  /* Gmaj7 */
        {"F#4 A4 C#5 E5", "C#6 - - A5 - - F#5 - . A5 - C#6 - E6 - -"}, /* F#m7 */
        {"B4 D5 F#5 A5",  "D6 - - - - - C#6 - B5 - - A5 - - F#5 -"},   /* Bm7 */
        {"A4 C#5 E5 G5",  "E5 - - - - - - - . . . . A5 - C#6 -"},      /* A7 */
        {"B4 D5 F#5 A5",  "D6 - - . D6 - - . C#6 - D6 - F#6 - - -"},   /* Bm7 */
        {"G4 B4 D5 F#5",  "- - - - E6 - D6 - B5 - - - A5 - B5 -"},     /* Gmaj7 */
        {"E4 G4 B4 D5",   "G5 - - . G5 - - . F#5 - G5 - B5 - - -"},    /* Em7 */
        {"A4 C#5 E5 G5",  "- - - - A5 - G5 - E5 - - - C#5 - E5 -"},    /* A7 */
        {"B4 D5 F#5 A5",  "F#5 - A5 - D6 - F#6 - - - E6 - D6 - E6 -"}, /* Bm7 */
        {"G4 B4 D5 F#5",  "F#6 - - - - - - - D6 - E6 - F#6 - A6 -"},   /* Gmaj7 */
        {"A4 C#5 E5 G5",  "G6 - - - F#6 - E6 - C#6 - - - A5 - - -"},   /* A7 */
        {"D4 F#4 A4 C#5", "D6 - - - - - - - - - - - . . . ."},         /* Dmaj7 */
    }};
    /* clang-format on */

    constexpr Tune ryujin_tune{
        .tempo = 132.f,
        .bars = ryujin_bars,
        .bass = "0.40..4.0.40.24.",
        .arpeggio = "0123456701234567",
        .kick = "x.....x.x.....x.",
        .snare = "....x.......x...",
        .hihat = "x.xxx.x.x.xxx.xx",
        .lead_ratio = 1.f,
    };


    /*
     * DashboardMusic::matthias: a maze chase at 150 beats per minute in A
     * minor.
     */

    /* clang-format off */
    constexpr std::array<Bar, 16> matthias_bars{{
        {"A4 C5 E5 G5",   "A5 . E5 . A5 C6 B5 A5 G#5 - A5 - E5 - . ."},     /* Am7 */
        {"A4 C5 E5 G5",   "C6 . B5 . C6 E6 D6 C6 B5 - C6 - A5 - - -"},      /* Am7 */
        {"F4 A4 C5 E5",   "F5 - A5 - C6 - E6 - D6 C6 A5 - F5 - A5 -"},      /* Fmaj7 */
        {"G4 B4 D5 F5",   "G5 - B5 - D6 - F6 - E6 D6 B5 - G5 - B5 -"},      /* G7 */
        {"A4 C5 E5 G5",   "E6 . E6 D#6 E6 . C6 . A5 . C6 . E6 - - ."},      /* Am7 */
        {"A4 C5 E5 G5",   "G6 . G6 F#6 G6 . E6 . C6 . E6 . G6 - A6 -"},     /* Am7 */
        {"D4 F4 A4 C5",   "F6 - - E6 D6 - - C6 A5 - C6 - D6 - F6 -"},       /* Dm7 */
        {"E4 G#4 B4 D5",  "E6 - - - D6 - B5 - G#5 - B5 - D6 - E6 -"},       /* E7 */
        {"F4 A4 C5 E5",   "A6 - - G6 F6 - E6 - F6 - - - C6 - - ."},         /* Fmaj7 */
        {"G4 B4 D5 F5",   "B6 - - A6 G6 - F6 - D6 - - - B5 - - ."},         /* G7 */
        {"E4 G4 B4 D5",   "E6 . B5 . E6 . G6 . F#6 . E6 . D6 . B5 ."},      /* Em7 */
        {"A4 C5 E5 G5",   "C6 - - - A5 - - - . . E5 F5 F#5 G5 G#5 -"},      /* Am7 */
        {"D4 F4 A4 C5",   "A5 - D6 - F6 - A6 - G6 F6 E6 D6 C6 - A5 -"},     /* Dm7 */
        {"E4 G#4 B4 D5",  "B5 - E6 - G#6 - B6 - A6 G#6 F6 E6 D6 - B5 -"},   /* E7 */
        {"A4 C5 E5 G5",   "A6 . E6 . C6 . A5 . E6 . C6 . A5 . E5 ."},       /* Am7 */
        {"E4 G#4 B4 D5",  "G#5 - - - B5 - - - D6 - E6 - G#6 - - -"},        /* E7 */
    }};
    /* clang-format on */

    constexpr Tune matthias_tune{
        .tempo = 150.f,
        .bars = matthias_bars,
        .bass = "0.4.0.4.0.4.2.4.",
        .arpeggio = "0246135702461357",
        .kick = "x...x...x...x...",
        .snare = "....x.......x.x.",
        .hihat = "..x...x...x...x.",
        .lead_ratio = 2.f,
    };


    /*
     * DashboardMusic::wyatt: a slow drift through the aquarium at 104
     * beats per minute in C major, with a touch of Lydian from the D7
     */

    /* clang-format off */
    constexpr std::array<Bar, 16> wyatt_bars{{
        {"C4 E4 G4 B4",   "E5 - - - G5 - B5 - - - - - A5 - G5 -"},     /* Cmaj7 */
        {"D4 F#4 A4 C5",  "F#5 - - - - - E5 - D5 - E5 - F#5 - A5 -"},  /* D7 */
        {"B3 D4 F#4 A4",  "B5 - - - - - - - A5 - F#5 - D5 - - -"},     /* Bm7 */
        {"E4 G4 B4 D5",   "E5 - - - - - - - . . B4 - D5 - E5 -"},      /* Em7 */
        {"A3 C4 E4 G4",   "C6 - - - B5 - A5 - - - G5 - E5 - - -"},     /* Am7 */
        {"D4 F#4 A4 C5",  "F#5 - A5 - C6 - - - D6 - C6 - A5 - - -"},   /* D7 */
        {"G3 B3 D4 F#4",  "B5 - - - - - - - A5 - B5 - D6 - - -"},      /* Gmaj7 */
        {"C4 E4 G4 B4",   "E6 . D6 . B5 . G5 . E5 - - - - - - -"},     /* Cmaj7 */
        {"F4 A4 C5 E5",   "A5 - - - C6 - E6 - - - - - D6 - C6 -"},     /* Fmaj7 */
        {"E4 G4 B4 D5",   "B5 - - - - - G5 - - - E5 - G5 - B5 -"},     /* Em7 */
        {"D4 F4 A4 C5",   "A5 - - - - - F5 - - - D5 - F5 - A5 -"},     /* Dm7 */
        {"G3 B3 D4 F4",   "B5 - - - D6 - - - F6 - - - D6 - B5 -"},     /* G7 */
        {"C4 E4 G4 B4",   "C6 - - - - - B5 - G5 - - - E5 - G5 -"},     /* Cmaj7 */
        {"A3 C4 E4 G4",   "A5 - - - - - G5 - E5 - - - C5 - E5 -"},     /* Am7 */
        {"D4 F#4 A4 C5",  "F#5 - - - - - - - A5 - - - C6 - - -"},      /* D7 */
        {"G3 B3 D4 F4",   "B5 - - - - - - - . . . . D5 E5 F5 D5"},     /* G7 */
    }};
    /* clang-format on */

    constexpr Tune wyatt_tune{
        .tempo = 104.f,
        .bars = wyatt_bars,
        .bass = "0.....4.2.....4.",
        .arpeggio = "0246753102467531",
        .kick = "x.......x.....x.",
        .snare = "........x.......",
        .hihat = "....x.......x...",
        .lead_ratio = 3.f,
    };


    /* The tune of @p music, null for DashboardMusic::none: */
    const Tune *find_tune(DashboardMusic music)
    {
      switch (music) {
      case DashboardMusic::none:
        return nullptr;
      case DashboardMusic::ryujin:
        return &ryujin_tune;
      case DashboardMusic::matthias:
        return &matthias_tune;
      case DashboardMusic::wyatt:
        return &wyatt_tune;
      }
      __builtin_unreachable();
    }


    constexpr int rest = -1;
    constexpr int hold = -2;

    /*
     * The MIDI note number of a note like "F#5", or rest, or hold:
     */
    int parse_note(std::string_view name)
    {
      if (name == ".")
        return rest;
      if (name == "-")
        return hold;

      constexpr std::array<int, 7> semitones{9, 11, 0, 2, 4, 5, 7};
      int note = semitones[name[0] - 'A'];
      if (name[1] == '#') {
        ++note;
        name.remove_prefix(1);
      }
      return note + 12 * (name[1] - '0' + 1);
    }


    std::vector<int> parse_notes(const char *notes)
    {
      std::istringstream stream(notes);
      std::vector<int> result;
      for (std::string name; stream >> name;)
        result.push_back(parse_note(name));
      return result;
    }


    using Chord = std::array<int, 4>;

    /*
     * The note of the character @p c of a bass or an arpeggio pattern, or
     * rest:
     */
    int chord_tone(const Chord &chord, char c)
    {
      if (c == '.')
        return rest;
      const int k = c - '0';
      return chord[k % 4] + 12 * (k / 4);
    }


    float frequency(int note)
    {
      return 440.f * std::exp2(static_cast<float>(note - 69) / 12.f);
    }


    /*
     * A voice of an instrument, sounding since time seconds, and released
     * after duration seconds:
     */
    struct Voice {
      float frequency = 0.f;
      float time = 1.e3f;
      float duration = 0.f;

      bool active() const
      {
        return time < duration + 1.f;
      }

      float release(float rate) const
      {
        return time < duration ? 1.f : std::exp((duration - time) * rate);
      }
    };


    /*
     * Two alternating voices, so that a new note does not cut off the
     * decay of the previous one:
     */
    struct Voices {
      std::array<Voice, 2> voices;
      unsigned int current = 0;

      void trigger(float frequency = 0.f, float duration = 0.f)
      {
        current ^= 1;
        voices[current] = {frequency, 0.f, duration};
      }

      template <typename Instrument, typename... Args>
      float sound(const Instrument &instrument, Args... args) const
      {
        float result = 0.f;
        for (const auto &voice : voices)
          if (voice.active())
            result += instrument(voice, args...);
        return result;
      }

      void advance(float dt)
      {
        for (auto &voice : voices)
          if (voice.active())
            voice.time += dt;
      }
    };


    /*
     * The instruments, two-operator FM synthesis in the style of the sound
     * chips of the arcade machines of the eighties:
     */

    constexpr float two_pi = 2.f * std::numbers::pi_v<float>;

    float bass(const Voice &v)
    {
      const float t = v.time;
      const float phase = two_pi * v.frequency * t;
      const float index = 0.6f + 2.5f * std::exp(-15.f * t);
      const float envelope = std::min(t / 0.003f, 1.f) * std::exp(-5.f * t);
      return envelope * v.release(40.f) *
             std::sin(phase + index * std::sin(phase));
    }

    float lead(const Voice &v, float ratio)
    {
      const float t = v.time;
      const float vibrato = std::clamp((t - 0.3f) / 0.3f, 0.f, 1.f) * 0.004f *
                            v.frequency / 5.5f * std::sin(two_pi * 5.5f * t);
      const float phase = two_pi * v.frequency * t + vibrato;
      const float index = 1.f + 1.5f * std::exp(-6.f * t);
      const float envelope =
          std::min(t / 0.01f, 1.f) * (0.75f + 0.25f * std::exp(-4.f * t));
      return envelope * v.release(12.f) *
             std::sin(phase + index * std::sin(ratio * phase));
    }

    float arpeggio(const Voice &v)
    {
      const float t = v.time;
      const float phase = two_pi * v.frequency * t;
      const float index = 1.5f * std::exp(-10.f * t);
      return std::exp(-12.f * t) * v.release(20.f) *
             std::sin(phase + index * std::sin(2.f * phase));
    }

    float kick(const Voice &v)
    {
      const float t = v.time;
      const float sweep = 45.f * t + 110.f / 35.f * (1.f - std::exp(-35.f * t));
      return std::exp(-7.f * t) * std::sin(two_pi * sweep);
    }

    float snare(const Voice &v, float noise)
    {
      const float t = v.time;
      return 0.5f * std::exp(-20.f * t) * noise +
             0.4f * std::exp(-30.f * t) * std::sin(two_pi * 190.f * t);
    }

    float hihat(const Voice &v, float noise)
    {
      return std::exp(-70.f * v.time) * noise;
    }


    /*
     * The synthesizer that renders a tune in an endless loop:
     */
    class Synthesizer
    {
    public:
      static constexpr unsigned int sample_rate = 48000;

      explicit Synthesizer(const Tune &tune)
          : tune_(tune)
          , step_samples_(
                static_cast<unsigned int>(sample_rate * 15.f / tune.tempo))
          , step_seconds_(static_cast<float>(step_samples_) / sample_rate)
          , echo_(3 * step_samples_, 0.f)
      {
        for (const auto pattern :
             {tune.bass, tune.arpeggio, tune.kick, tune.snare, tune.hihat}) {
          (void)pattern;
          Assert(pattern.size() == 16, dealii::ExcInternalError());
        }

        std::vector<int> notes;
        for (const auto &bar : tune.bars) {
          const auto chord = parse_notes(bar.chord);
          const auto lead = parse_notes(bar.lead);
          Assert(chord.size() == 4, dealii::ExcInternalError());
          Assert(lead.size() == 16, dealii::ExcInternalError());
          chords_.push_back({chord[0], chord[1], chord[2], chord[3]});
          notes.insert(notes.end(), lead.begin(), lead.end());
        }

        /* Every note of the lead line, and for how many steps it holds: */
        for (std::size_t i = 0; i < notes.size(); ++i) {
          int steps = 1;
          while (notes[(i + steps) % notes.size()] == hold)
            ++steps;
          melody_.emplace_back(notes[i], steps);
        }
      }

      void render(float *output, unsigned int frames)
      {
        constexpr float dt = 1.f / sample_rate;

        for (unsigned int k = 0; k < frames; ++k) {
          if (countdown_-- == 0) {
            countdown_ = step_samples_ - 1;
            next_step();
          }

          /* A white noise, and a high pass of it for the hihat: */
          random_ = random_ * 1664525u + 1013904223u;
          const float previous = noise_;
          noise_ = static_cast<float>(random_) / 2147483648.f - 1.f;
          const float hiss = 0.5f * (noise_ - previous);

          /* The lead with a dotted eighth echo: */
          const float lead_line = lead_.sound(lead, tune_.lead_ratio);
          float &echo_sample = echo_[echo_position_];
          echo_position_ = (echo_position_ + 1) % echo_.size();
          const float delayed = echo_sample;
          echo_sample = lead_line + 0.4f * delayed;

          const float mix =
              0.45f * bass_.sound(bass) + 0.28f * (lead_line + 0.5f * delayed) +
              0.10f * arpeggio_.sound(arpeggio) + 0.60f * kick_.sound(kick) +
              0.30f * snare_.sound(snare, noise_) +
              0.12f * hihat_.sound(hihat, hiss);
          output[k] = 0.3f * std::tanh(mix);

          for (auto *voices :
               {&bass_, &lead_, &arpeggio_, &kick_, &snare_, &hihat_})
            voices->advance(dt);
        }
      }

    private:
      void next_step()
      {
        const auto i = position_ % 16;
        const auto &chord = chords_[position_ / 16 % chords_.size()];

        if (tune_.kick[i] == 'x')
          kick_.trigger();
        if (tune_.snare[i] == 'x')
          snare_.trigger();
        if (tune_.hihat[i] == 'x')
          hihat_.trigger();

        if (const int note = chord_tone(chord, tune_.bass[i]); note != rest)
          bass_.trigger(frequency(note - 24), 0.8f * step_seconds_);

        if (const int note = chord_tone(chord, tune_.arpeggio[i]); note != rest)
          arpeggio_.trigger(frequency(note - 12), 0.5f * step_seconds_);

        const auto [note, steps] = melody_[position_ % melody_.size()];
        if (note >= 0)
          lead_.trigger(frequency(note), steps * step_seconds_ - 0.02f);

        ++position_;
      }

      const Tune tune_;
      const unsigned int step_samples_;
      const float step_seconds_;

      std::vector<Chord> chords_;
      std::vector<std::pair<int, int>> melody_;

      unsigned int position_ = 0;
      unsigned int countdown_ = 0;

      Voices bass_;
      Voices lead_;
      Voices arpeggio_;
      Voices kick_;
      Voices snare_;
      Voices hihat_;

      std::uint32_t random_ = 1;
      float noise_ = 0.f;

      std::vector<float> echo_;
      std::size_t echo_position_ = 0;
    };


    /*
     * An audio device that plays a tune in its background thread for the
     * lifetime of the object. Silent if there is no audio device.
     */
    class Player
    {
    public:
      explicit Player(const Tune &tune)
          : synthesizer_(tune)
      {
        auto config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_f32;
        config.playback.channels = 1;
        config.sampleRate = Synthesizer::sample_rate;
        config.pUserData = &synthesizer_;
        config.dataCallback = [](ma_device *device,
                                 void *output,
                                 const void * /*input*/,
                                 ma_uint32 frames) {
          static_cast<Synthesizer *>(device->pUserData)
              ->render(static_cast<float *>(output), frames);
        };

        initialized_ = ma_device_init(nullptr, &config, &device_) == MA_SUCCESS;
        if (initialized_)
          ma_device_start(&device_);
      }

      /* Stop the device and end its background thread: */
      ~Player()
      {
        if (initialized_)
          ma_device_uninit(&device_);
      }

      Player(const Player &) = delete;
      Player &operator=(const Player &) = delete;

    private:
      Synthesizer synthesizer_;
      ma_device device_{};
      bool initialized_ = false;
    };


    /*
     * The one Player of the program, replaced whenever the music changes:
     */
    class Jukebox
    {
    public:
      void play(DashboardMusic music)
      {
        std::scoped_lock lock(mutex_);
        if (music == music_)
          return;

        music_ = music;
        player_.reset();
        if (const auto *tune = find_tune(music))
          player_ = std::make_unique<Player>(*tune);
      }

    private:
      std::mutex mutex_;
      DashboardMusic music_ = DashboardMusic::none;
      std::unique_ptr<Player> player_;
    };
  } // namespace


  void play_dashboard_music(DashboardMusic music)
  {
    static Jukebox jukebox;
    jukebox.play(music);
  }
} // namespace ryujin
