#pragma once

#include <cstdint>

namespace ps2_host_input
{
    constexpr uint32_t kInvertRightX = 1u;
    constexpr uint32_t kInvertRightY = 2u;

    struct MouseCameraInput
    {
        float x = 0.0f;
        float y = 0.0f;
        bool active = false;
    };

    void setRightStickInvertDefault(uint32_t axes);
    void setMouseCameraEnabled(bool enabled);
    void setMouseStickEnabled(bool enabled);
    [[nodiscard]] MouseCameraInput consumeMouseCameraInput();
}
