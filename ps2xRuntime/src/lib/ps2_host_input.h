#pragma once

#include "ps2x/iop/iop_pad.h"

#include <cstdint>

namespace ps2_host_input
{
    constexpr uint32_t kPadPorts = 2u;

    void sample();
    [[nodiscard]] ps2x::iop::PadInputState padState(uint32_t port);
}
