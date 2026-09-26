#include "iop_spu2.h"

#include <algorithm>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kCoreStride = 0x400u;
        constexpr uint32_t kSharedBase = 0x760u;
        constexpr uint32_t kVoiceParamsEnd = 0x180u;
        constexpr uint32_t kVoiceAddressBase = 0x1C0u;
        constexpr uint32_t kVoiceAddressStride = 0xCu;
        constexpr uint32_t kVoiceAddressEnd = kVoiceAddressBase + IopSpu2::kVoices * kVoiceAddressStride;
        constexpr uint32_t kKeyOn0 = 0x1A0u;
        constexpr uint32_t kKeyOn1 = 0x1A2u;
        constexpr uint32_t kKeyOff0 = 0x1A4u;
        constexpr uint32_t kKeyOff1 = 0x1A6u;
        constexpr uint32_t kTransferAddressHigh = 0x1A8u;
        constexpr uint32_t kTransferAddressLow = 0x1AAu;
        constexpr uint32_t kTransferData = 0x1ACu;
        constexpr uint32_t kAutoDma = 0x1B0u;
        constexpr uint32_t kEndFlags0 = 0x340u;
        constexpr uint32_t kEndFlags1 = 0x342u;
        constexpr uint32_t kStatus = 0x344u;

        constexpr uint32_t kParamPitch = 2u;
        constexpr uint32_t kParamAdsr1 = 3u;
        constexpr uint32_t kParamAdsr2 = 4u;
        constexpr uint32_t kParamEnvelope = 5u;

        constexpr uint32_t kAddressMask = 0xFFFFFu;
        constexpr uint8_t kLoopEnd = 0x1u;
        constexpr uint8_t kLoopRepeat = 0x2u;
        constexpr uint8_t kLoopStart = 0x4u;
        constexpr uint32_t kSamplesPerBlock = 28u;
        constexpr int32_t kPitchStep = 4096;
        constexpr int32_t kMaxPitch = 0x3FFF;
        constexpr int64_t kEnvelopeMax = 0x7FFFFFFF;
        constexpr uint16_t kStatusDmaReady = 0x80u;

        constexpr uint8_t kPhaseOff = 0u;
        constexpr uint8_t kPhaseAttack = 1u;
        constexpr uint8_t kPhaseDecay = 2u;
        constexpr uint8_t kPhaseSustain = 3u;
        constexpr uint8_t kPhaseSustainEnd = 4u;
        constexpr uint8_t kPhaseRelease = 5u;

        constexpr std::array<int32_t, 8> kInverseExpOffsets{0, 4, 6, 8, 9, 10, 11, 12};

        int64_t envelopeRate(int32_t index)
        {
            const int32_t shift = index >> 2;
            int64_t rate = (index & 3) + 4;
            if (shift < 0)
                rate >>= std::min(-shift, 62);
            else
                rate <<= std::min(shift, 40);
            return std::min<int64_t>(rate, 0x3FFFFFFF);
        }
    }

    IopSpu2::IopSpu2()
        : m_ram(kRamHalfwords)
    {
        reset();
    }

    void IopSpu2::reset()
    {
        m_regs.fill(0u);
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        for (auto &core : m_voices)
            core.fill(Voice{});
        m_endx.fill(0u);
        m_transferAddress.fill(0u);
        m_cycle = 0u;
    }

    uint16_t IopSpu2::read16(uint32_t physicalAddress, uint64_t cycle)
    {
        advanceTo(cycle);
        const uint32_t offset = (physicalAddress - kBase) & 0x7FEu;
        if (offset >= kSharedBase)
            return reg(offset);
        const uint32_t core = offset / kCoreStride;
        const uint32_t local = offset % kCoreStride;
        if (local < kVoiceParamsEnd)
        {
            const uint32_t voice = local >> 4u;
            if (((local & 0xFu) >> 1u) == kParamEnvelope && voice < kVoices)
                return static_cast<uint16_t>(std::max(m_voices[core][voice].envelope, 0) >> 16);
            return reg(offset);
        }
        if (local >= kVoiceAddressBase && local < kVoiceAddressEnd)
        {
            const Voice &voice = m_voices[core][(local - kVoiceAddressBase) / kVoiceAddressStride];
            switch (((local - kVoiceAddressBase) % kVoiceAddressStride) >> 1u)
            {
            case 0:
                return static_cast<uint16_t>(voice.startA >> 16);
            case 1:
                return static_cast<uint16_t>(voice.startA);
            case 2:
                return static_cast<uint16_t>(voice.loopStartA >> 16);
            case 3:
                return static_cast<uint16_t>(voice.loopStartA);
            case 4:
                return static_cast<uint16_t>(voice.nextA >> 16);
            default:
                return static_cast<uint16_t>(voice.nextA);
            }
        }
        switch (local)
        {
        case kTransferAddressHigh:
            return static_cast<uint16_t>(m_transferAddress[core] >> 16);
        case kTransferAddressLow:
            return static_cast<uint16_t>(m_transferAddress[core]);
        case kEndFlags0:
            return static_cast<uint16_t>(m_endx[core]);
        case kEndFlags1:
            return static_cast<uint16_t>(m_endx[core] >> 16);
        default:
            return reg(offset);
        }
    }

    void IopSpu2::write16(uint32_t physicalAddress, uint16_t value, uint64_t cycle)
    {
        advanceTo(cycle);
        const uint32_t offset = (physicalAddress - kBase) & 0x7FEu;
        reg(offset) = value;
        if (offset >= kSharedBase)
            return;
        const uint32_t core = offset / kCoreStride;
        const uint32_t local = offset % kCoreStride;
        if (local < kVoiceParamsEnd)
        {
            const uint32_t voice = local >> 4u;
            if (((local & 0xFu) >> 1u) == kParamEnvelope && voice < kVoices)
                m_voices[core][voice].envelope = static_cast<int32_t>((static_cast<uint32_t>(value) << 16) | value) & 0x7FFFFFFF;
            return;
        }
        if (local >= kVoiceAddressBase && local < kVoiceAddressEnd)
        {
            Voice &voice = m_voices[core][(local - kVoiceAddressBase) / kVoiceAddressStride];
            const uint32_t high = (static_cast<uint32_t>(value) & 0xFu) << 16;
            switch (((local - kVoiceAddressBase) % kVoiceAddressStride) >> 1u)
            {
            case 0:
                voice.startA = high | (voice.startA & 0xFFF8u);
                break;
            case 1:
                voice.startA = (voice.startA & 0xF0000u) | (value & 0xFFF8u);
                break;
            case 2:
                voice.loopStartA = high | (voice.loopStartA & 0xFFF8u);
                voice.manualLoop = true;
                break;
            case 3:
                voice.loopStartA = (voice.loopStartA & 0xF0000u) | (value & 0xFFF8u);
                voice.manualLoop = true;
                break;
            case 4:
                voice.nextA = high | (voice.nextA & 0xFFFFu);
                voice.sampleIndex = kSamplesPerBlock;
                break;
            default:
                voice.nextA = (voice.nextA & 0xF0000u) | value;
                voice.sampleIndex = kSamplesPerBlock;
                break;
            }
            return;
        }
        switch (local)
        {
        case kKeyOn0:
        case kKeyOn1:
        {
            const uint32_t first = local == kKeyOn0 ? 0u : 16u;
            for (uint32_t bit = 0; bit < 16u && first + bit < kVoices; ++bit)
                if ((value >> bit) & 1u)
                    keyOn(core, first + bit);
            break;
        }
        case kKeyOff0:
        case kKeyOff1:
        {
            const uint32_t first = local == kKeyOff0 ? 0u : 16u;
            for (uint32_t bit = 0; bit < 16u && first + bit < kVoices; ++bit)
                if ((value >> bit) & 1u)
                    keyOff(core, first + bit);
            break;
        }
        case kTransferAddressHigh:
            m_transferAddress[core] = ((static_cast<uint32_t>(value) & 0xFu) << 16) | (m_transferAddress[core] & 0xFFFFu);
            break;
        case kTransferAddressLow:
            m_transferAddress[core] = (m_transferAddress[core] & 0xF0000u) | value;
            break;
        case kTransferData:
            m_ram[m_transferAddress[core]] = value;
            m_transferAddress[core] = (m_transferAddress[core] + 1u) & kAddressMask;
            break;
        case kEndFlags0:
            m_endx[core] &= 0xFF0000u;
            break;
        case kEndFlags1:
            m_endx[core] &= 0xFFFFu;
            break;
        default:
            break;
        }
    }

    bool IopSpu2::autoDma(uint32_t core) const noexcept
    {
        return (reg(core * kCoreStride + kAutoDma) & (core + 1u)) != 0u;
    }

    void IopSpu2::dmaStarted(uint32_t core) noexcept
    {
        reg(core * kCoreStride + kStatus) |= kStatusDmaReady;
    }

    void IopSpu2::dmaWrite(uint32_t core, const uint8_t *source, size_t bytes, uint64_t cycle)
    {
        advanceTo(cycle);
        uint32_t address = m_transferAddress[core];
        for (size_t i = 0; i + 2u <= bytes; i += 2u)
        {
            m_ram[address] = static_cast<uint16_t>(source[i] | (source[i + 1u] << 8));
            address = (address + 1u) & kAddressMask;
        }
        m_transferAddress[core] = address;
    }

    void IopSpu2::dmaRead(uint32_t core, uint8_t *destination, size_t bytes, uint64_t cycle)
    {
        advanceTo(cycle);
        uint32_t address = m_transferAddress[core];
        for (size_t i = 0; i + 2u <= bytes; i += 2u)
        {
            destination[i] = static_cast<uint8_t>(m_ram[address]);
            destination[i + 1u] = static_cast<uint8_t>(m_ram[address] >> 8);
            address = (address + 1u) & kAddressMask;
        }
        m_transferAddress[core] = address;
    }

    void IopSpu2::advanceTo(uint64_t cycle)
    {
        if (cycle <= m_cycle)
            return;
        const uint64_t samples = (cycle - m_cycle) / kCyclesPerSample;
        if (samples == 0u)
            return;
        m_cycle += samples * kCyclesPerSample;
        for (uint32_t core = 0; core < kCores; ++core)
            for (uint32_t voice = 0; voice < kVoices; ++voice)
                runVoice(core, voice, samples);
    }

    void IopSpu2::keyOn(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        if (voice.startA & 7u)
            voice.startA = ((voice.startA + 8u) & ~7u) & kAddressMask;
        voice.nextA = voice.startA;
        voice.sampleIndex = kSamplesPerBlock;
        voice.counter = 0;
        voice.loopFlags = 0u;
        voice.manualLoop = false;
        voice.envelope = 0;
        voice.phase = kPhaseAttack;
        m_endx[core] &= ~(1u << index);
    }

    void IopSpu2::keyOff(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        if (voice.phase != kPhaseOff)
            voice.phase = kPhaseRelease;
    }

    void IopSpu2::runVoice(uint32_t core, uint32_t index, uint64_t samples)
    {
        Voice &voice = m_voices[core][index];
        const uint32_t params = core * kCoreStride + index * 0x10u;
        const int32_t pitch = std::min<int32_t>(reg(params + kParamPitch * 2u), kMaxPitch);
        if (voice.phase == kPhaseOff && pitch == 0)
            return;
        const uint16_t adsr1 = reg(params + kParamAdsr1 * 2u);
        const uint16_t adsr2 = reg(params + kParamAdsr2 * 2u);
        for (uint64_t sample = 0; sample < samples; ++sample)
        {
            if (voice.phase != kPhaseOff)
                stepEnvelope(voice, adsr1, adsr2);
            voice.counter += pitch;
            while (voice.counter > 0)
            {
                stepAddress(core, index);
                voice.counter -= kPitchStep;
            }
        }
    }

    void IopSpu2::stepAddress(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        if (voice.sampleIndex >= kSamplesPerBlock)
        {
            const uint32_t block = voice.nextA & ~7u & kAddressMask;
            voice.loopFlags = static_cast<uint8_t>(m_ram[block] >> 8);
            if ((voice.loopFlags & kLoopStart) && !voice.manualLoop)
                voice.loopStartA = block;
            voice.sampleIndex = 0u;
            voice.nextA = (block + 1u) & kAddressMask;
        }
        ++voice.sampleIndex;
        if ((voice.sampleIndex & 3u) != 0u)
            return;
        voice.nextA = (voice.nextA + 1u) & kAddressMask;
        if ((voice.nextA & 7u) != 0u)
            return;
        if (voice.loopFlags & kLoopEnd)
        {
            m_endx[core] |= 1u << index;
            voice.nextA = voice.loopStartA;
            if (!(voice.loopFlags & kLoopRepeat))
            {
                voice.envelope = 0;
                voice.phase = kPhaseOff;
            }
        }
    }

    void IopSpu2::stepEnvelope(Voice &voice, uint16_t adsr1, uint16_t adsr2)
    {
        const bool attackExp = (adsr1 & 0x8000u) != 0u;
        const int32_t attackRate = (adsr1 >> 8) & 0x7F;
        const int32_t decayRate = (adsr1 >> 4) & 0xF;
        const int32_t sustainLevel = adsr1 & 0xF;
        const bool sustainExp = (adsr2 & 0x8000u) != 0u;
        const bool sustainDown = (adsr2 & 0x4000u) != 0u;
        const int32_t sustainRate = (adsr2 >> 6) & 0x7F;
        const bool releaseExp = (adsr2 & 0x20u) != 0u;
        const int32_t releaseRate = adsr2 & 0x1F;
        int64_t value = voice.envelope;
        const int32_t expOffset = kInverseExpOffsets[(value >> 28) & 7];

        switch (voice.phase)
        {
        case kPhaseAttack:
            if (value >= kEnvelopeMax)
            {
                voice.phase = kPhaseDecay;
                break;
            }
            value += envelopeRate((attackRate ^ 0x7F) - ((attackExp && value >= 0x60000000) ? 0x18 : 0x10));
            if (value >= kEnvelopeMax)
            {
                value = kEnvelopeMax;
                voice.phase = kPhaseDecay;
            }
            break;
        case kPhaseDecay:
        {
            value -= envelopeRate(((decayRate ^ 0x1F) * 4) - 0x18 + expOffset);
            const int64_t level = sustainLevel == 0xF ? kEnvelopeMax : (static_cast<int64_t>(sustainLevel) + 1) << 27;
            if (value <= level)
            {
                value = std::max<int64_t>(value, 0);
                voice.phase = kPhaseSustain;
            }
            break;
        }
        case kPhaseSustain:
            if (sustainDown)
            {
                value -= sustainExp ? envelopeRate((sustainRate ^ 0x7F) - 0x1B + expOffset)
                                    : envelopeRate((sustainRate ^ 0x7F) - 0xF);
                if (value <= 0)
                {
                    value = 0;
                    voice.phase = kPhaseSustainEnd;
                }
            }
            else
            {
                value += envelopeRate((sustainRate ^ 0x7F) - ((sustainExp && value >= 0x60000000) ? 0x18 : 0x10));
                if (value >= kEnvelopeMax)
                {
                    value = kEnvelopeMax;
                    voice.phase = kPhaseSustainEnd;
                }
            }
            break;
        case kPhaseSustainEnd:
            value = sustainDown ? 0 : kEnvelopeMax;
            break;
        case kPhaseRelease:
            value -= releaseExp ? envelopeRate(((releaseRate ^ 0x1F) * 4) - 0x18 + expOffset)
                                : envelopeRate(((releaseRate ^ 0x1F) * 4) - 0xC);
            if (value <= 0)
            {
                value = 0;
                voice.phase = kPhaseOff;
            }
            break;
        default:
            value = 0;
            break;
        }
        voice.envelope = static_cast<int32_t>(value);
    }
}
