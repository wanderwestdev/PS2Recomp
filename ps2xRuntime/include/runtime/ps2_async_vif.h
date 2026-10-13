#pragma once

#include "ps2x/state_io.h"

#include <cstdint>
#include <functional>
#include <vector>

class PS2Memory;

namespace ps2x::asyncvif
{
    enum class ItemKind : uint8_t
    {
        GifPath3,
        Vif1,
    };

    struct Item
    {
        ItemKind kind = ItemKind::Vif1;
        std::vector<uint8_t> data;
    };

    void setDefaultEnabled(bool enabled);
    void setHostPacingDefault(bool enabled);
    void setFbrstProvider(std::function<uint32_t()> provider);
    void setGifProcessor(std::function<void(const uint8_t *, uint32_t)> processor);
    bool submitGif(const uint8_t *data, uint32_t sizeBytes);
    bool active(PS2Memory &memory);
    bool onWorker();
    uint32_t currentFbrst();
    std::vector<uint8_t> takeBuffer();
    void enqueue(PS2Memory &memory, std::vector<Item> &&items);
    void sync(PS2Memory &memory);
    // Waits for the VU thread only. The GIF stage keeps drawing what it already has; every
    // later GIF transfer still goes through the VU thread's queue, so the order is kept.
    void syncVu1(PS2Memory &memory);
    void stop(PS2Memory &memory);

    struct Vu1TimingHooks
    {
        std::function<uint64_t()> eeCycle;
        std::function<uint64_t()> vu1Cycles;
        std::function<uint64_t(uint64_t)> consumeEeCycles;
    };

    void setVu1Timing(Vu1TimingHooks hooks);
    void vu1Kick();
    bool vu1Busy(PS2Memory &memory);
    void saveTimingState(ps2x::state::Writer &out);
    void loadTimingState(ps2x::state::Reader &in);
}
