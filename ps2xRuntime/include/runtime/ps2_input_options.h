#pragma once

#include <cstdint>

namespace ps2_host_input
{
    constexpr uint32_t kInvertRightX = 1u;
    constexpr uint32_t kInvertRightY = 2u;

    void setRightStickInvertDefault(uint32_t axes);
    void setMouseCameraEnabled(bool enabled);
}
