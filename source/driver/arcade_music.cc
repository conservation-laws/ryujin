//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/arcade_music.h>

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
#include <numbers>
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
     * The tune: sixteen bars of sixteen steps (sixteenth notes) at 132
     * beats per minute. Every bar has a chord of four notes, the lowest
     * one being the root, and a lead line of sixteen steps, where a note
     * is written as in "F#5", "-" holds the previous note, and "." rests.
     * The bass line, the arpeggio, and the drums follow the same pattern
     * in every bar.
     */

    constexpr float tempo = 132.f;

    struct Bar {
      const char *chord;
      const char *lead;
    };

    /* clang-format off */
    constexpr std::array<Bar, 16> song{{
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

    constexpr int rest = -1;
    constexpr int hold = -2;

    /* Semitones above the root of the chord, two octaves down: */
    constexpr std::array<int, 16> bass_line{
        0, rest, 12, 0, rest, rest, 12, rest, 0, rest, 12, 0, rest, 7, 12, rest};
    /* clang-format on */

    constexpr std::string_view kick_pattern = "x.....x.x.....x.";
    constexpr std::string_view snare_pattern = "....x.......x...";
    constexpr std::string_view hihat_pattern = "x.xxx.x.x.xxx.xx";


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

      void trigger(float frequency, float duration = 0.f)
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

    float lead(const Voice &v)
    {
      const float t = v.time;
      const float vibrato = std::clamp((t - 0.3f) / 0.3f, 0.f, 1.f) * 0.004f *
                            v.frequency / 5.5f * std::sin(two_pi * 5.5f * t);
      const float phase = two_pi * v.frequency * t + vibrato;
      const float index = 1.f + 1.5f * std::exp(-6.f * t);
      const float envelope =
          std::min(t / 0.01f, 1.f) * (0.75f + 0.25f * std::exp(-4.f * t));
      return envelope * v.release(12.f) *
             std::sin(phase + index * std::sin(phase));
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


    class Synthesizer
    {
    public:
      static constexpr unsigned int sample_rate = 48000;

      Synthesizer()
          : step_samples(static_cast<unsigned int>(sample_rate * 15.f / tempo))
          , step_seconds(static_cast<float>(step_samples) / sample_rate)
          , echo(3 * step_samples, 0.f)
      {
        std::vector<int> notes;
        for (const auto &bar : song) {
          const auto chord = parse_notes(bar.chord);
          const auto lead = parse_notes(bar.lead);
          Assert(chord.size() == 4, dealii::ExcInternalError());
          Assert(lead.size() == 16, dealii::ExcInternalError());
          chords.push_back({chord[0], chord[1], chord[2], chord[3]});
          notes.insert(notes.end(), lead.begin(), lead.end());
        }

        /* Every note of the lead line, and for how many steps it holds: */
        for (std::size_t i = 0; i < notes.size(); ++i) {
          int steps = 1;
          while (notes[(i + steps) % notes.size()] == hold)
            ++steps;
          melody.emplace_back(notes[i], steps);
        }
      }

      void render(float *output, unsigned int frames)
      {
        constexpr float dt = 1.f / sample_rate;

        for (unsigned int k = 0; k < frames; ++k) {
          if (countdown-- == 0) {
            countdown = step_samples - 1;
            next_step();
          }

          /* A white noise, and a high pass of it for the hihat: */
          random = random * 1664525u + 1013904223u;
          const float previous = noise;
          noise = static_cast<float>(random) / 2147483648.f - 1.f;
          const float hiss = 0.5f * (noise - previous);

          const float lead_line = lead_voices.sound(lead);
          float &echo_sample = echo[echo_position];
          echo_position = (echo_position + 1) % echo.size();
          const float delayed = echo_sample;
          echo_sample = lead_line + 0.4f * delayed;

          const float mix = 0.45f * bass_voices.sound(bass) +
                            0.28f * (lead_line + 0.5f * delayed) +
                            0.10f * arpeggio_voices.sound(arpeggio) +
                            0.60f * kick_voices.sound(kick) +
                            0.30f * snare_voices.sound(snare, noise) +
                            0.12f * hihat_voices.sound(hihat, hiss);
          output[k] = 0.3f * std::tanh(mix);

          for (auto *voices : {&bass_voices,
                               &lead_voices,
                               &arpeggio_voices,
                               &kick_voices,
                               &snare_voices,
                               &hihat_voices})
            voices->advance(dt);
        }
      }

    private:
      void next_step()
      {
        const auto i = position % 16;
        const auto &chord = chords[position / 16 % chords.size()];

        if (kick_pattern[i] == 'x')
          kick_voices.trigger(0.f);
        if (snare_pattern[i] == 'x')
          snare_voices.trigger(0.f);
        if (hihat_pattern[i] == 'x')
          hihat_voices.trigger(0.f);

        if (bass_line[i] != rest)
          bass_voices.trigger(frequency(chord[0] - 24 + bass_line[i]),
                              0.8f * step_seconds);

        /* Up the chord, alternating between two octaves: */
        arpeggio_voices.trigger(frequency(chord[i % 4] - 12 + 12 * (i / 4 % 2)),
                                0.5f * step_seconds);

        const auto [note, steps] = melody[position % melody.size()];
        if (note >= 0)
          lead_voices.trigger(frequency(note), steps * step_seconds - 0.02f);

        ++position;
      }

      const unsigned int step_samples;
      const float step_seconds;

      std::vector<std::array<int, 4>> chords;
      std::vector<std::pair<int, int>> melody;

      unsigned int position = 0;
      unsigned int countdown = 0;

      Voices bass_voices;
      Voices lead_voices;
      Voices arpeggio_voices;
      Voices kick_voices;
      Voices snare_voices;
      Voices hihat_voices;

      std::uint32_t random = 1;
      float noise = 0.f;

      std::vector<float> echo;
      std::size_t echo_position = 0;
    };


    /*
     * The audio device that plays the synthesizer until the program exits:
     */
    class Player
    {
    public:
      Player()
      {
        auto config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_f32;
        config.playback.channels = 1;
        config.sampleRate = Synthesizer::sample_rate;
        config.pUserData = &synthesizer;
        config.dataCallback = [](ma_device *device,
                                 void *output,
                                 const void * /*input*/,
                                 ma_uint32 frames) {
          static_cast<Synthesizer *>(device->pUserData)
              ->render(static_cast<float *>(output), frames);
        };

        initialized = ma_device_init(nullptr, &config, &device) == MA_SUCCESS;
        if (initialized)
          ma_device_start(&device);
      }

      ~Player()
      {
        if (initialized)
          ma_device_uninit(&device);
      }

      Player(const Player &) = delete;
      Player &operator=(const Player &) = delete;

    private:
      Synthesizer synthesizer;
      ma_device device{};
      bool initialized = false;
    };
  } // namespace


  void play_arcade_music()
  {
    static Player player;
  }
} // namespace ryujin
