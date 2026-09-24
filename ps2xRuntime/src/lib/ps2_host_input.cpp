#include "ps2_host_input.h"
#include "ps2_host_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ps2_host_input
{
    namespace
    {
        using ps2x::iop::PadInputState;
        using ps2x::iop::PadPressure;
        namespace PadButton = ps2x::iop::PadButton;

        struct ButtonBinding
        {
            uint16_t button;
            int gamepadButton;
            int key;
            int alternateKey;
        };

        constexpr std::array<ButtonBinding, 16> kButtons{{
            {PadButton::Up, GAMEPAD_BUTTON_LEFT_FACE_UP, KEY_UP, KEY_NULL},
            {PadButton::Down, GAMEPAD_BUTTON_LEFT_FACE_DOWN, KEY_DOWN, KEY_NULL},
            {PadButton::Left, GAMEPAD_BUTTON_LEFT_FACE_LEFT, KEY_LEFT, KEY_NULL},
            {PadButton::Right, GAMEPAD_BUTTON_LEFT_FACE_RIGHT, KEY_RIGHT, KEY_NULL},
            {PadButton::Cross, GAMEPAD_BUTTON_RIGHT_FACE_DOWN, KEY_X, KEY_SPACE},
            {PadButton::Circle, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT, KEY_C, KEY_BACKSPACE},
            {PadButton::Square, GAMEPAD_BUTTON_RIGHT_FACE_LEFT, KEY_Z, KEY_NULL},
            {PadButton::Triangle, GAMEPAD_BUTTON_RIGHT_FACE_UP, KEY_V, KEY_NULL},
            {PadButton::L1, GAMEPAD_BUTTON_LEFT_TRIGGER_1, KEY_Q, KEY_NULL},
            {PadButton::R1, GAMEPAD_BUTTON_RIGHT_TRIGGER_1, KEY_E, KEY_NULL},
            {PadButton::L2, GAMEPAD_BUTTON_LEFT_TRIGGER_2, KEY_LEFT_SHIFT, KEY_NULL},
            {PadButton::R2, GAMEPAD_BUTTON_RIGHT_TRIGGER_2, KEY_RIGHT_SHIFT, KEY_NULL},
            {PadButton::Start, GAMEPAD_BUTTON_MIDDLE_RIGHT, KEY_ENTER, KEY_NULL},
            {PadButton::Select, GAMEPAD_BUTTON_MIDDLE_LEFT, KEY_TAB, KEY_NULL},
            {PadButton::L3, GAMEPAD_BUTTON_LEFT_THUMB, KEY_F, KEY_NULL},
            {PadButton::R3, GAMEPAD_BUTTON_RIGHT_THUMB, KEY_G, KEY_NULL},
        }};

        constexpr std::array<std::pair<PadPressure, uint16_t>, static_cast<size_t>(PadPressure::Count)> kPressureButtons{{
            {PadPressure::Right, PadButton::Right},
            {PadPressure::Left, PadButton::Left},
            {PadPressure::Up, PadButton::Up},
            {PadPressure::Down, PadButton::Down},
            {PadPressure::Triangle, PadButton::Triangle},
            {PadPressure::Circle, PadButton::Circle},
            {PadPressure::Cross, PadButton::Cross},
            {PadPressure::Square, PadButton::Square},
            {PadPressure::L1, PadButton::L1},
            {PadPressure::R1, PadButton::R1},
            {PadPressure::L2, PadButton::L2},
            {PadPressure::R2, PadButton::R2},
        }};

        struct ScriptStep
        {
            double start = 0.0;
            double end = 0.0;
            uint16_t buttons = 0u;
        };

        constexpr std::array<std::pair<std::string_view, uint16_t>, 16> kButtonNames{{
            {"select", PadButton::Select},
            {"l3", PadButton::L3},
            {"r3", PadButton::R3},
            {"start", PadButton::Start},
            {"up", PadButton::Up},
            {"right", PadButton::Right},
            {"down", PadButton::Down},
            {"left", PadButton::Left},
            {"l2", PadButton::L2},
            {"r2", PadButton::R2},
            {"l1", PadButton::L1},
            {"r1", PadButton::R1},
            {"triangle", PadButton::Triangle},
            {"circle", PadButton::Circle},
            {"cross", PadButton::Cross},
            {"square", PadButton::Square},
        }};

        std::vector<std::string_view> split(std::string_view text, char separator)
        {
            std::vector<std::string_view> parts;
            size_t begin = 0u;
            while (begin <= text.size())
            {
                const size_t end = std::min(text.find(separator, begin), text.size());
                parts.push_back(text.substr(begin, end - begin));
                begin = end + 1u;
            }
            return parts;
        }

        std::optional<ScriptStep> parseStep(std::string_view text)
        {
            constexpr double kDefaultHoldSeconds = 1.0;
            const std::vector<std::string_view> fields = split(text, ':');
            if (fields.size() < 2u || fields.size() > 3u)
                return std::nullopt;
            try
            {
                ScriptStep step;
                step.start = std::stod(std::string(fields[0]));
                step.end = step.start + (fields.size() == 3u ? std::stod(std::string(fields[2])) : kDefaultHoldSeconds);
                for (std::string_view name : split(fields[1], '+'))
                {
                    const auto found = std::find_if(kButtonNames.begin(), kButtonNames.end(), [&](const auto &entry)
                                                    { return entry.first == name; });
                    if (found == kButtonNames.end())
                        return std::nullopt;
                    step.buttons |= found->second;
                }
                return step;
            }
            catch (...)
            {
                return std::nullopt;
            }
        }

        std::vector<ScriptStep> loadScript()
        {
            std::vector<ScriptStep> script;
            const char *value = std::getenv("PS2X_PAD_SCRIPT");
            if (!value || !*value)
                return script;
            for (std::string_view entry : split(value, ','))
            {
                if (const auto step = parseStep(entry))
                    script.push_back(*step);
                else
                    std::cerr << "[input] ignoring pad script step '" << entry << "'" << std::endl;
            }
            std::cerr << "[input] pad script: " << script.size() << " step(s) on port 0" << std::endl;
            return script;
        }

        uint16_t scriptedButtons()
        {
            static const std::vector<ScriptStep> script = loadScript();
            static const auto origin = std::chrono::steady_clock::now();
            if (script.empty())
                return 0u;
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - origin).count();
            uint16_t buttons = 0u;
            for (const ScriptStep &step : script)
            {
                if (now >= step.start && now < step.end)
                    buttons |= step.buttons;
            }
            return buttons;
        }

        std::mutex g_mutex;
        std::array<PadInputState, kPadPorts> g_state = []
        {
            std::array<PadInputState, kPadPorts> state{};
            state[0].connected = true;
            return state;
        }();

        uint8_t axisToByte(float value)
        {
            const float scaled = std::round(127.5f + std::clamp(value, -1.0f, 1.0f) * 127.5f);
            return static_cast<uint8_t>(std::clamp(scaled, 0.0f, 255.0f));
        }

        uint8_t keyAxis(int negative, int positive)
        {
            const bool low = IsKeyDown(negative);
            const bool high = IsKeyDown(positive);
            if (low == high)
                return 0x80u;
            return low ? 0x00u : 0xFFu;
        }

        PadInputState readPort(uint32_t port)
        {
            PadInputState state;
            const int gamepad = static_cast<int>(port);
            const bool gamepadAvailable = IsGamepadAvailable(gamepad);
            const bool keyboard = port == 0u;
            state.connected = gamepadAvailable || keyboard;
            if (!state.connected)
                return state;

            for (const ButtonBinding &binding : kButtons)
            {
                const bool padDown = gamepadAvailable && IsGamepadButtonDown(gamepad, binding.gamepadButton);
                const bool keyDown = keyboard && (IsKeyDown(binding.key) ||
                                                  (binding.alternateKey != KEY_NULL && IsKeyDown(binding.alternateKey)));
                if (padDown || keyDown)
                    state.buttons |= binding.button;
            }
            if (keyboard)
                state.buttons |= scriptedButtons();

            if (gamepadAvailable)
            {
                state.leftX = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_X));
                state.leftY = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_Y));
                state.rightX = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_X));
                state.rightY = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_Y));
            }
            if (keyboard)
            {
                if (const uint8_t x = keyAxis(KEY_A, KEY_D); x != 0x80u)
                    state.leftX = x;
                if (const uint8_t y = keyAxis(KEY_W, KEY_S); y != 0x80u)
                    state.leftY = y;
                if (const uint8_t x = keyAxis(KEY_J, KEY_L); x != 0x80u)
                    state.rightX = x;
                if (const uint8_t y = keyAxis(KEY_I, KEY_K); y != 0x80u)
                    state.rightY = y;
            }

            for (const auto &[pressure, button] : kPressureButtons)
                state.pressure[static_cast<size_t>(pressure)] = (state.buttons & button) != 0u ? 0xFFu : 0x00u;
            return state;
        }
    }

    void sample()
    {
        std::array<PadInputState, kPadPorts> state{};
        for (uint32_t port = 0; port < kPadPorts; ++port)
            state[port] = readPort(port);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_state = state;
    }

    ps2x::iop::PadInputState padState(uint32_t port)
    {
        if (port >= kPadPorts)
            return {};
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_state[port];
    }
}
