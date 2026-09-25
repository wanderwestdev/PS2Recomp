#pragma once

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
    void setFbrstProvider(std::function<uint32_t()> provider);
    void setGifProcessor(std::function<void(const uint8_t *, uint32_t)> processor);
    bool submitGif(const uint8_t *data, uint32_t sizeBytes);
    bool active(PS2Memory &memory);
    bool onWorker();
    uint32_t currentFbrst();
    std::vector<uint8_t> takeBuffer();
    void enqueue(PS2Memory &memory, std::vector<Item> &&items);
    void sync(PS2Memory &memory);
    void stop(PS2Memory &memory);
}
