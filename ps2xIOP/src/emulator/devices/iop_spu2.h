#pragma once

#include <array>
#include "ps2x/state_io.h"
#include <cstdint>
#include <span>
#include <vector>

namespace ps2x::iop::detail
{
    class IopSpu2
    {
    public:
        static constexpr uint32_t kBase = 0x1F900000u;
        static constexpr uint32_t kEnd = 0x1F900800u;
        static constexpr uint32_t kRamHalfwords = 0x100000u;
        static constexpr uint64_t kCyclesPerSample = 768u;
        static constexpr uint32_t kCores = 2u;
        static constexpr uint32_t kVoices = 24u;

        IopSpu2();

        void reset();

        [[nodiscard]] static bool owns(uint32_t physicalAddress) noexcept
        {
            return physicalAddress >= kBase && physicalAddress < kEnd;
        }

        [[nodiscard]] uint16_t read16(uint32_t physicalAddress, uint64_t cycle);
        void write16(uint32_t physicalAddress, uint16_t value, uint64_t cycle);

        [[nodiscard]] bool autoDma(uint32_t core) const noexcept;
        void dmaStarted(uint32_t core) noexcept;
        void dmaWrite(uint32_t core, const uint8_t *source, size_t bytes, uint64_t cycle);
        void dmaRead(uint32_t core, uint8_t *destination, size_t bytes, uint64_t cycle);

        void advanceTo(uint64_t cycle);

        [[nodiscard]] uint32_t nextAddress(uint32_t core, uint32_t voice) const noexcept { return m_voices[core][voice].nextA; }
        [[nodiscard]] uint32_t loopAddress(uint32_t core, uint32_t voice) const noexcept { return m_voices[core][voice].loopStartA; }
        [[nodiscard]] int32_t envelope(uint32_t core, uint32_t voice) const noexcept { return m_voices[core][voice].envelope; }
        [[nodiscard]] uint32_t endFlags(uint32_t core) const noexcept { return m_endx[core]; }
        [[nodiscard]] uint32_t transferAddress(uint32_t core) const noexcept { return m_transferAddress[core]; }
        [[nodiscard]] std::span<const uint16_t> ram() const noexcept { return m_ram; }

        template <typename Ar>
        void serialize(Ar &ar)
        {
            ps2x::state::fields(ar, m_regs, m_ram, m_voices, m_endx, m_transferAddress, m_cycle);
        }

    private:
        struct Voice
        {
            uint32_t startA = 0u;
            uint32_t loopStartA = 0u;
            uint32_t nextA = 0u;
            uint32_t sampleIndex = 28u;
            int32_t counter = 0;
            uint8_t loopFlags = 0u;
            bool manualLoop = false;
            int32_t envelope = 0;
            uint8_t phase = 0u;
        };

        [[nodiscard]] uint16_t &reg(uint32_t offset) noexcept { return m_regs[(offset & 0x7FEu) >> 1u]; }
        [[nodiscard]] uint16_t reg(uint32_t offset) const noexcept { return m_regs[(offset & 0x7FEu) >> 1u]; }

        void keyOn(uint32_t core, uint32_t voice);
        void keyOff(uint32_t core, uint32_t voice);
        void runVoice(uint32_t core, uint32_t voice, uint64_t samples);
        void stepAddress(uint32_t core, uint32_t voice);
        void stepEnvelope(Voice &voice, uint16_t adsr1, uint16_t adsr2);

        std::array<uint16_t, 0x400> m_regs{};
        std::vector<uint16_t> m_ram;
        std::array<std::array<Voice, kVoices>, kCores> m_voices{};
        std::array<uint32_t, kCores> m_endx{};
        std::array<uint32_t, kCores> m_transferAddress{};
        uint64_t m_cycle = 0u;
    };
}
