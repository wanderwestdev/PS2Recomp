#include "iop_spu2.h"

#include <algorithm>
#include <cmath>
#include <numbers>

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

        int32_t volume(uint16_t value)
        {
            if (value & 0x8000u)
                return 0;
            return static_cast<int16_t>(value << 1u);
        }

        int32_t scale(int32_t sample, int32_t gain)
        {
            return static_cast<int32_t>((static_cast<int64_t>(sample) * gain) >> 15);
        }

        int32_t saturate(int32_t sample)
        {
            return std::clamp(sample, -32768, 32767);
        }

        const std::array<std::array<int16_t, 4>, 256> &interpolationWeights()
        {
            static const auto weights = []
            {
                std::array<double, 512> kernel{};
                double total = 0.0;
                for (size_t index = 0u; index < kernel.size(); ++index)
                {
                    const double position = static_cast<double>(index) + 0.5;
                    const double window = 1.0 + (std::cos(std::numbers::pi * position / 511.5) - 1.0) * 0.5
                                              + (std::cos(std::numbers::pi * position / 255.75) - 1.0) * 0.08;
                    const double value = std::sin(std::numbers::pi * position / 500.0) * window / position;
                    kernel[511u - index] = value;
                    total += value;
                }
                const double normalization = 32640.0 * 128.0 / total;
                std::array<std::array<int16_t, 4>, 256> result{};
                for (size_t phase = 0u; phase < result.size(); ++phase)
                {
                    const std::array<size_t, 4> indices{255u - phase, 511u - phase, phase + 256u, phase};
                    double sum = 0.0;
                    for (const size_t index : indices)
                        sum += kernel[index] * normalization;
                    const double correction = (sum - 32640.0) / 4.0;
                    for (size_t tap = 0u; tap < indices.size(); ++tap)
                        result[phase][tap] = static_cast<int16_t>(std::lround(kernel[indices[tap]] * normalization - correction));
                }
                return result;
            }();
            return weights;
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
        for (auto &core : m_decoders)
            core.fill(Decoder{});
        m_interpolation = {};
        m_audioFrames = 0u;
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
        {
            for (uint32_t sharedCore = 0u; sharedCore < kCores; ++sharedCore)
            {
                const uint32_t shared = kSharedBase + sharedCore * 0x28u;
                if (offset == shared || offset == shared + 2u)
                    reg(offset + 0x10u) = static_cast<uint16_t>(volume(value));
            }
            return;
        }
        const uint32_t core = offset / kCoreStride;
        const uint32_t local = offset % kCoreStride;
        if (local < kVoiceParamsEnd)
        {
            const uint32_t voice = local >> 4u;
            if ((local & 0xFu) <= 2u && voice < kVoices)
                reg(offset + 0xCu) = static_cast<uint16_t>(volume(value));
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
        for (uint64_t sample = 0; sample < samples; ++sample)
        {
            const auto core0 = mixCore(0u, {});
            const auto output = mixCore(1u, core0);
            m_audioBuffer[m_audioFrames * 2u] = static_cast<int16_t>(output[0]);
            m_audioBuffer[m_audioFrames * 2u + 1u] = static_cast<int16_t>(output[1]);
            if (++m_audioFrames == m_audioBuffer.size() / 2u)
                flushAudio();
        }
    }

    void IopSpu2::flushAudio()
    {
        if (m_audioFrames != 0u && m_audioSink)
            m_audioSink(std::span<const int16_t>(m_audioBuffer.data(), m_audioFrames * 2u));
        m_audioFrames = 0u;
    }

    std::array<int32_t, 2> IopSpu2::mixCore(uint32_t core, const std::array<int32_t, 2> &external)
    {
        const uint32_t base = core * kCoreStride;
        const uint32_t leftMask = reg(base + 0x188u) | (static_cast<uint32_t>(reg(base + 0x18Au)) << 16u);
        const uint32_t rightMask = reg(base + 0x190u) | (static_cast<uint32_t>(reg(base + 0x192u)) << 16u);
        std::array<int32_t, 2> output{};
        for (uint32_t index = 0u; index < kVoices; ++index)
        {
            const int32_t sample = runVoice(core, index);
            if (sample == 0)
                continue;
            const uint32_t params = base + index * 0x10u;
            const int32_t left = volume(reg(params));
            const int32_t right = volume(reg(params + 2u));
            if (leftMask & (1u << index))
                output[0] += scale(sample, left);
            if (rightMask & (1u << index))
                output[1] += scale(sample, right);
        }
        const uint16_t mixer = reg(base + 0x198u);
        const uint32_t shared = kSharedBase + core * 0x28u;
        for (uint32_t channel = 0u; channel < 2u; ++channel)
        {
            if (!(mixer & (0x800u >> channel)))
                output[channel] = 0;
            if (mixer & (0x80u >> channel))
                output[channel] += scale(external[channel], static_cast<int16_t>(reg(shared + 8u + channel * 2u)));
            const int32_t master = volume(reg(shared + channel * 2u));
            output[channel] = scale(saturate(output[channel]), master);
            if (!(reg(base + 0x19Au) & 0x4000u))
                output[channel] = 0;
        }
        return output;
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
        m_decoders[core][index] = {};
        m_interpolation[core][index] = {};
        m_endx[core] &= ~(1u << index);
    }

    void IopSpu2::keyOff(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        if (voice.phase != kPhaseOff)
            voice.phase = kPhaseRelease;
    }

    int32_t IopSpu2::runVoice(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        const uint32_t params = core * kCoreStride + index * 0x10u;
        const int32_t pitch = std::min<int32_t>(reg(params + kParamPitch * 2u), kMaxPitch);
        if (voice.phase == kPhaseOff && pitch == 0)
            return 0;
        const uint16_t adsr1 = reg(params + kParamAdsr1 * 2u);
        const uint16_t adsr2 = reg(params + kParamAdsr2 * 2u);
        if (voice.phase != kPhaseOff)
            stepEnvelope(voice, adsr1, adsr2);
        voice.counter += pitch;
        while (voice.counter > 0)
        {
            stepAddress(core, index);
            voice.counter -= kPitchStep;
        }
        if (voice.phase == kPhaseOff || voice.envelope == 0)
            return 0;
        const int32_t fraction = voice.counter + kPitchStep;
        const auto &weights = interpolationWeights()[std::min(fraction >> 4, 255)];
        const auto &history = m_interpolation[core][index];
        int32_t pcm = 0;
        for (size_t tap = 0u; tap < history.size(); ++tap)
            pcm += scale(history[tap], weights[tap]);
        return scale(saturate(pcm), voice.envelope >> 16);
    }

    void IopSpu2::decodeBlock(uint32_t core, uint32_t index, uint32_t address)
    {
        Decoder &decoder = m_decoders[core][index];
        constexpr std::array<int32_t, 5> first{0, 60, 115, 98, 122};
        constexpr std::array<int32_t, 5> second{0, 0, -52, -55, -60};
        const uint16_t header = m_ram[address];
        const uint32_t predictor = (header >> 4u) & 0xFu;
        const int32_t c1 = predictor < first.size() ? first[predictor] : 0;
        const int32_t c2 = predictor < second.size() ? second[predictor] : 0;
        const uint32_t shift = header & 0xFu;
        for (uint32_t sample = 0u; sample < kSamplesPerBlock; ++sample)
        {
            const uint16_t packed = m_ram[(address + 1u + sample / 4u) & kAddressMask];
            const int32_t nibble = (packed >> ((sample % 4u) * 4u)) & 0xFu;
            const int32_t signedNibble = nibble >= 8 ? nibble - 16 : nibble;
            const int32_t raw = (signedNibble * 4096) >> shift;
            const int32_t filtered = raw + ((decoder.history1 * c1 + decoder.history2 * c2 + 32) >> 6);
            const int32_t decoded = saturate(filtered);
            decoder.pcm[sample] = static_cast<int16_t>(decoded);
            decoder.history2 = decoder.history1;
            decoder.history1 = decoded;
        }
    }

    void IopSpu2::stepAddress(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        if (voice.sampleIndex >= kSamplesPerBlock)
        {
            const uint32_t block = voice.nextA & ~7u & kAddressMask;
            if (voice.phase != kPhaseOff)
                decodeBlock(core, index, block);
            voice.loopFlags = static_cast<uint8_t>(m_ram[block] >> 8);
            if ((voice.loopFlags & kLoopStart) && !voice.manualLoop)
                voice.loopStartA = block;
            voice.sampleIndex = 0u;
            voice.nextA = (block + 1u) & kAddressMask;
        }
        if (voice.phase != kPhaseOff)
        {
            Decoder &decoder = m_decoders[core][index];
            decoder.previous = decoder.current;
            decoder.current = decoder.pcm[voice.sampleIndex];
            auto &history = m_interpolation[core][index];
            history[0] = history[1];
            history[1] = history[2];
            history[2] = history[3];
            history[3] = static_cast<int16_t>(decoder.current);
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
