#include "ps2_host_input.h"
#include "runtime/ps2_input_options.h"
#include "ps2_host_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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

        enum class BindingKind
        {
            None,
            Keyboard,
            Mouse,
            Shift,
            Control
        };

        struct HostBinding
        {
            BindingKind kind;
            int code;
        };

        struct ButtonBinding
        {
            uint16_t button;
            int gamepadButton;
            const char *setting;
            HostBinding defaultBinding;
        };

        constexpr std::array<ButtonBinding, 16> kButtons{{
            {PadButton::Up, GAMEPAD_BUTTON_LEFT_FACE_UP, "PS2X_BIND_UP", {BindingKind::Keyboard, KEY_UP}},
            {PadButton::Down, GAMEPAD_BUTTON_LEFT_FACE_DOWN, "PS2X_BIND_DOWN", {BindingKind::Keyboard, KEY_DOWN}},
            {PadButton::Left, GAMEPAD_BUTTON_LEFT_FACE_LEFT, "PS2X_BIND_LEFT", {BindingKind::Keyboard, KEY_LEFT}},
            {PadButton::Right, GAMEPAD_BUTTON_LEFT_FACE_RIGHT, "PS2X_BIND_RIGHT", {BindingKind::Keyboard, KEY_RIGHT}},
            {PadButton::Cross, GAMEPAD_BUTTON_RIGHT_FACE_DOWN, "PS2X_BIND_CROSS", {BindingKind::Keyboard, KEY_F}},
            {PadButton::Circle, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT, "PS2X_BIND_CIRCLE", {BindingKind::Mouse, MOUSE_BUTTON_RIGHT}},
            {PadButton::Square, GAMEPAD_BUTTON_RIGHT_FACE_LEFT, "PS2X_BIND_SQUARE", {BindingKind::Mouse, MOUSE_BUTTON_LEFT}},
            {PadButton::Triangle, GAMEPAD_BUTTON_RIGHT_FACE_UP, "PS2X_BIND_TRIANGLE", {BindingKind::Keyboard, KEY_SPACE}},
            {PadButton::L1, GAMEPAD_BUTTON_LEFT_TRIGGER_1, "PS2X_BIND_L1", {BindingKind::Keyboard, KEY_Q}},
            {PadButton::R1, GAMEPAD_BUTTON_RIGHT_TRIGGER_1, "PS2X_BIND_R1", {BindingKind::Keyboard, KEY_LEFT_SHIFT}},
            {PadButton::L2, GAMEPAD_BUTTON_LEFT_TRIGGER_2, "PS2X_BIND_L2", {BindingKind::Keyboard, KEY_LEFT_CONTROL}},
            {PadButton::R2, GAMEPAD_BUTTON_RIGHT_TRIGGER_2, "PS2X_BIND_R2", {BindingKind::Keyboard, KEY_RIGHT_SHIFT}},
            {PadButton::Start, GAMEPAD_BUTTON_MIDDLE_RIGHT, "PS2X_BIND_START", {BindingKind::Keyboard, KEY_ENTER}},
            {PadButton::Select, GAMEPAD_BUTTON_MIDDLE_LEFT, "PS2X_BIND_SELECT", {BindingKind::Keyboard, KEY_TAB}},
            {PadButton::L3, GAMEPAD_BUTTON_LEFT_THUMB, "PS2X_BIND_L3", {BindingKind::Keyboard, KEY_H}},
            {PadButton::R3, GAMEPAD_BUTTON_RIGHT_THUMB, "PS2X_BIND_R3", {BindingKind::Keyboard, KEY_G}},
        }};

        constexpr std::array<std::pair<std::string_view, HostBinding>, 24> kNamedBindings{{
            {"NONE", {BindingKind::None, 0}},
            {"UP", {BindingKind::Keyboard, KEY_UP}},
            {"DOWN", {BindingKind::Keyboard, KEY_DOWN}},
            {"LEFT", {BindingKind::Keyboard, KEY_LEFT}},
            {"RIGHT", {BindingKind::Keyboard, KEY_RIGHT}},
            {"SPACE", {BindingKind::Keyboard, KEY_SPACE}},
            {"BACKSPACE", {BindingKind::Keyboard, KEY_BACKSPACE}},
            {"ENTER", {BindingKind::Keyboard, KEY_ENTER}},
            {"TAB", {BindingKind::Keyboard, KEY_TAB}},
            {"ESCAPE", {BindingKind::Keyboard, KEY_ESCAPE}},
            {"LEFT_SHIFT", {BindingKind::Keyboard, KEY_LEFT_SHIFT}},
            {"RIGHT_SHIFT", {BindingKind::Keyboard, KEY_RIGHT_SHIFT}},
            {"SHIFT", {BindingKind::Shift, 0}},
            {"LEFT_CTRL", {BindingKind::Keyboard, KEY_LEFT_CONTROL}},
            {"RIGHT_CTRL", {BindingKind::Keyboard, KEY_RIGHT_CONTROL}},
            {"LEFT_CONTROL", {BindingKind::Keyboard, KEY_LEFT_CONTROL}},
            {"RIGHT_CONTROL", {BindingKind::Keyboard, KEY_RIGHT_CONTROL}},
            {"CTRL", {BindingKind::Control, 0}},
            {"CONTROL", {BindingKind::Control, 0}},
            {"MOUSE_LEFT", {BindingKind::Mouse, MOUSE_BUTTON_LEFT}},
            {"MOUSE_RIGHT", {BindingKind::Mouse, MOUSE_BUTTON_RIGHT}},
            {"MOUSE_MIDDLE", {BindingKind::Mouse, MOUSE_BUTTON_MIDDLE}},
            {"LMB", {BindingKind::Mouse, MOUSE_BUTTON_LEFT}},
            {"RMB", {BindingKind::Mouse, MOUSE_BUTTON_RIGHT}},
        }};

        std::optional<HostBinding> parseBinding(std::string_view value)
        {
            std::string name(value);
            for (char &ch : name)
            {
                if (ch == ' ' || ch == '-')
                    ch = '_';
                else
                    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            }
            if (name.size() == 1u)
            {
                if (name[0] >= 'A' && name[0] <= 'Z')
                    return HostBinding{BindingKind::Keyboard, KEY_A + name[0] - 'A'};
                if (name[0] >= '0' && name[0] <= '9')
                    return HostBinding{BindingKind::Keyboard, KEY_ZERO + name[0] - '0'};
            }
            for (const auto &[key, binding] : kNamedBindings)
            {
                if (name == key)
                    return binding;
            }
            if (name.size() >= 2u && name[0] == 'F')
            {
                char *end = nullptr;
                const long number = std::strtol(name.c_str() + 1, &end, 10);
                if (*end == '\0' && number >= 1 && number <= 12)
                    return HostBinding{BindingKind::Keyboard, static_cast<int>(KEY_F1 + number - 1)};
            }
            return std::nullopt;
        }

        std::array<HostBinding, kButtons.size()> loadBindings()
        {
            std::array<HostBinding, kButtons.size()> bindings{};
            for (size_t i = 0; i < kButtons.size(); ++i)
            {
                bindings[i] = kButtons[i].defaultBinding;
                if (const char *value = std::getenv(kButtons[i].setting))
                {
                    if (const auto parsed = parseBinding(value))
                        bindings[i] = *parsed;
                    else
                        std::cerr << "[input] invalid " << kButtons[i].setting << "='" << value << "', using default" << std::endl;
                }
            }
            return bindings;
        }

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
            bool fields = false;
            uint16_t buttons = 0u;
            std::array<uint8_t, 4> axes{0x80u, 0x80u, 0x80u, 0x80u};
        };

        struct AxisName
        {
            std::string_view name;
            size_t axis;
            uint8_t value;
        };

        constexpr std::array<AxisName, 8> kAxisNames{{
            {"lleft", 0u, 0x00u},
            {"lright", 0u, 0xFFu},
            {"lup", 1u, 0x00u},
            {"ldown", 1u, 0xFFu},
            {"rleft", 2u, 0x00u},
            {"rright", 2u, 0xFFu},
            {"rup", 3u, 0x00u},
            {"rdown", 3u, 0xFFu},
        }};

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
            constexpr double kDefaultHoldFields = 10.0;
            const std::vector<std::string_view> fields = split(text, ':');
            if (fields.size() < 2u || fields.size() > 3u)
                return std::nullopt;
            try
            {
                ScriptStep step;
                std::string_view start = fields[0];
                step.fields = !start.empty() && start.back() == 'v';
                if (step.fields)
                    start.remove_suffix(1u);
                step.start = std::stod(std::string(start));
                const double defaultHold = step.fields ? kDefaultHoldFields : kDefaultHoldSeconds;
                step.end = step.start + (fields.size() == 3u ? std::stod(std::string(fields[2])) : defaultHold);
                for (std::string_view name : split(fields[1], '+'))
                {
                    const auto axis = std::find_if(kAxisNames.begin(), kAxisNames.end(), [&](const AxisName &entry)
                                                   { return entry.name == name; });
                    if (axis != kAxisNames.end())
                    {
                        step.axes[axis->axis] = axis->value;
                        continue;
                    }
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

        struct ScriptedInput
        {
            uint16_t buttons = 0u;
            std::array<uint8_t, 4> axes{0x80u, 0x80u, 0x80u, 0x80u};
            bool active = false;
        };

        ScriptedInput scriptedInput(uint64_t vsyncTick)
        {
            static const std::vector<ScriptStep> script = loadScript();
            static const auto origin = std::chrono::steady_clock::now();
            ScriptedInput input;
            if (script.empty())
                return input;
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - origin).count();
            for (const ScriptStep &step : script)
            {
                const double clock = step.fields ? static_cast<double>(vsyncTick) : now;
                if (clock >= step.start && clock < step.end)
                {
                    input.buttons |= step.buttons;
                    for (size_t i = 0; i < input.axes.size(); ++i)
                    {
                        if (step.axes[i] != 0x80u)
                        {
                            input.axes[i] = step.axes[i];
                            input.active = true;
                        }
                    }
                }
            }
            input.active = input.active || input.buttons != 0u;
            return input;
        }

        std::mutex g_mutex;
        std::array<PadInputState, kPadPorts> g_state = []
        {
            std::array<PadInputState, kPadPorts> state{};
            state[0].connected = true;
            return state;
        }();

        constexpr uint8_t kAxisNeutral = 0x7Fu;

        uint8_t axisToByte(float value)
        {
            const float clamped = std::clamp(value, -1.0f, 1.0f);
            const float scaled = std::round(static_cast<float>(kAxisNeutral) + clamped * (clamped < 0.0f ? 127.0f : 128.0f));
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

        std::atomic<const std::atomic<uint64_t> *> g_fieldCounter{nullptr};
        std::atomic<uint32_t> g_invertRightDefault{0u};
        std::atomic<bool> g_mouseCameraEnabled{false};
        bool g_mouseLocked = false;
        int g_ignoredMouseButton = -1;
        bool g_skipMouseDelta = false;
        bool g_mouseMoved = false;
        float g_mouseRightX = 0.0f;
        float g_mouseRightY = 0.0f;

        bool isHostBindingDown(const HostBinding &binding)
        {
            switch (binding.kind)
            {
            case BindingKind::Keyboard:
                return IsKeyDown(binding.code);
            case BindingKind::Mouse:
                return g_mouseLocked && binding.code != g_ignoredMouseButton && IsMouseButtonDown(binding.code);
            case BindingKind::Shift:
                return IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
            case BindingKind::Control:
                return IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);
            default:
                return false;
            }
        }

        float positiveSetting(const char *name, float fallback)
        {
            if (const char *value = std::getenv(name))
            {
                char *end = nullptr;
                const float parsed = std::strtof(value, &end);
                if (end != value && *end == '\0' && std::isfinite(parsed) && parsed > 0.0f)
                    return parsed;
            }
            return fallback;
        }

        float mouseSensitivity()
        {
            static const float sensitivity = positiveSetting("PS2X_MOUSE_SENSITIVITY", 1.0f);
            return sensitivity;
        }

        float mouseVerticalSensitivity()
        {
            static const float sensitivity = positiveSetting("PS2X_MOUSE_VERTICAL_SENSITIVITY", 1.5f);
            return sensitivity;
        }

        float mouseStickAxis(float desired)
        {
            if (desired == 0.0f)
                return 0.0f;
            const float magnitude = 0.42f + 0.58f * std::clamp(std::abs(desired), 0.0f, 1.0f);
            return std::copysign(magnitude, desired);
        }

        void sampleMouse()
        {
            g_mouseMoved = false;
            if (!g_mouseCameraEnabled.load(std::memory_order_relaxed))
            {
                if (g_mouseLocked)
                    EnableCursor();
                g_mouseLocked = false;
                g_ignoredMouseButton = -1;
                return;
            }
            if (!IsWindowFocused() || IsKeyPressed(KEY_ESCAPE))
            {
                if (g_mouseLocked)
                    EnableCursor();
                g_mouseLocked = false;
                g_ignoredMouseButton = -1;
                return;
            }
            if (!g_mouseLocked)
            {
                int button = -1;
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
                    button = MOUSE_BUTTON_LEFT;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
                    button = MOUSE_BUTTON_RIGHT;
                else if (IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE))
                    button = MOUSE_BUTTON_MIDDLE;
                if (button != -1)
                {
                    DisableCursor();
                    g_mouseLocked = true;
                    g_ignoredMouseButton = button;
                    g_skipMouseDelta = true;
                }
                return;
            }
            if (g_ignoredMouseButton != -1 && !IsMouseButtonDown(g_ignoredMouseButton))
                g_ignoredMouseButton = -1;
            if (g_skipMouseDelta)
            {
                g_skipMouseDelta = false;
                return;
            }

            const Vector2 delta = GetMouseDelta();
            const float distance = std::hypot(delta.x, delta.y);
            if (distance <= 0.0f)
                return;
            const float frameTime = std::max(GetFrameTime(), 1.0f / 240.0f);
            const float scale = mouseSensitivity() / (frameTime * 900.0f);
            g_mouseRightX = mouseStickAxis(delta.x * scale);
            g_mouseRightY = mouseStickAxis(delta.y * scale * mouseVerticalSensitivity());
            g_mouseMoved = true;
        }

        uint32_t rightStickInvert()
        {
            if (const char *value = std::getenv("PS2X_INVERT_RIGHT_STICK"))
            {
                const std::string_view text(value);
                uint32_t axes = 0u;
                if (text.find_first_of("xX") != std::string_view::npos || text == "1")
                    axes |= kInvertRightX;
                if (text.find_first_of("yY") != std::string_view::npos || text == "1")
                    axes |= kInvertRightY;
                return axes;
            }
            return g_invertRightDefault.load(std::memory_order_relaxed);
        }

        void applyPressure(PadInputState &state)
        {
            for (const auto &[pressure, button] : kPressureButtons)
                state.pressure[static_cast<size_t>(pressure)] = (state.buttons & button) != 0u ? 0xFFu : 0x00u;
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
            state.leftX = kAxisNeutral;
            state.leftY = kAxisNeutral;
            state.rightX = kAxisNeutral;
            state.rightY = kAxisNeutral;

            static const auto hostBindings = loadBindings();
            for (size_t i = 0; i < kButtons.size(); ++i)
            {
                const ButtonBinding &binding = kButtons[i];
                const bool padDown = gamepadAvailable && IsGamepadButtonDown(gamepad, binding.gamepadButton);
                const bool keyDown = keyboard && isHostBindingDown(hostBindings[i]);
                if (padDown || keyDown)
                    state.buttons |= binding.button;
            }
            if (gamepadAvailable)
            {
                state.leftX = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_X));
                state.leftY = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_Y));
                const uint32_t invert = rightStickInvert();
                const float rightX = GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_X);
                const float rightY = GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_Y);
                state.rightX = axisToByte((invert & kInvertRightX) != 0u ? -rightX : rightX);
                state.rightY = axisToByte((invert & kInvertRightY) != 0u ? -rightY : rightY);
            }
            if (keyboard)
            {
                if (const uint8_t x = keyAxis(KEY_A, KEY_D); x != 0x80u)
                    state.leftX = x;
                if (const uint8_t y = keyAxis(KEY_W, KEY_S); y != 0x80u)
                    state.leftY = y;
                if (g_mouseMoved)
                {
                    const uint32_t invert = rightStickInvert();
                    state.rightX = axisToByte((invert & kInvertRightX) != 0u ? -g_mouseRightX : g_mouseRightX);
                    state.rightY = axisToByte((invert & kInvertRightY) != 0u ? -g_mouseRightY : g_mouseRightY);
                }
            }

            applyPressure(state);
            return state;
        }
    }

    void setRightStickInvertDefault(uint32_t axes)
    {
        g_invertRightDefault.store(axes, std::memory_order_relaxed);
    }

    void setMouseCameraEnabled(bool enabled)
    {
        g_mouseCameraEnabled.store(enabled, std::memory_order_relaxed);
    }

    void setFieldCounter(const std::atomic<uint64_t> *counter)
    {
        g_fieldCounter.store(counter);
    }

    void sample()
    {
        sampleMouse();
        std::array<PadInputState, kPadPorts> state{};
        for (uint32_t port = 0; port < kPadPorts; ++port)
            state[port] = readPort(port);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_state = state;
    }

    void releaseMouse()
    {
        if (g_mouseLocked)
            EnableCursor();
        g_mouseLocked = false;
        g_ignoredMouseButton = -1;
        g_mouseMoved = false;
    }

    ps2x::iop::PadInputState padState(uint32_t port)
    {
        if (port >= kPadPorts)
            return {};
        PadInputState state;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            state = g_state[port];
        }
        if (port == 0u)
        {
            const std::atomic<uint64_t> *counter = g_fieldCounter.load();
            const ScriptedInput scripted = scriptedInput(counter ? counter->load() : 0u);
            if (scripted.active)
            {
                state.connected = true;
                state.buttons |= scripted.buttons;
                if (scripted.axes[0] != 0x80u)
                    state.leftX = scripted.axes[0];
                if (scripted.axes[1] != 0x80u)
                    state.leftY = scripted.axes[1];
                if (scripted.axes[2] != 0x80u)
                    state.rightX = scripted.axes[2];
                if (scripted.axes[3] != 0x80u)
                    state.rightY = scripted.axes[3];
                applyPressure(state);
            }
        }
        return state;
    }
}
