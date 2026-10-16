#include "runtime/ps2_audio.h"
#include "runtime/ps2_memory.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
    std::vector<uint8_t> buildWavFromPcm(const int16_t *pcm, size_t sampleCount, uint32_t sampleRate)
    {
        const uint32_t dataSize = static_cast<uint32_t>(sampleCount * 2);
        const uint32_t fileSize = 36 + dataSize;
        std::vector<uint8_t> wav(8 + fileSize);

        uint8_t *p = wav.data();
        p[0] = 'R';
        p[1] = 'I';
        p[2] = 'F';
        p[3] = 'F';
        p[4] = static_cast<uint8_t>(fileSize);
        p[5] = static_cast<uint8_t>(fileSize >> 8);
        p[6] = static_cast<uint8_t>(fileSize >> 16);
        p[7] = static_cast<uint8_t>(fileSize >> 24);
        p[8] = 'W';
        p[9] = 'A';
        p[10] = 'V';
        p[11] = 'E';
        p[12] = 'f';
        p[13] = 'm';
        p[14] = 't';
        p[15] = ' ';
        p[16] = 16;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 1;
        p[21] = 0;
        p[22] = 1;
        p[23] = 0;
        p[24] = static_cast<uint8_t>(sampleRate);
        p[25] = static_cast<uint8_t>(sampleRate >> 8);
        p[26] = static_cast<uint8_t>(sampleRate >> 16);
        p[27] = static_cast<uint8_t>(sampleRate >> 24);
        const uint32_t byteRate = sampleRate * 2;
        p[28] = static_cast<uint8_t>(byteRate);
        p[29] = static_cast<uint8_t>(byteRate >> 8);
        p[30] = static_cast<uint8_t>(byteRate >> 16);
        p[31] = static_cast<uint8_t>(byteRate >> 24);
        p[32] = 2;
        p[33] = 0;
        p[34] = 16;
        p[35] = 0;
        p[36] = 'd';
        p[37] = 'a';
        p[38] = 't';
        p[39] = 'a';
        p[40] = static_cast<uint8_t>(dataSize);
        p[41] = static_cast<uint8_t>(dataSize >> 8);
        p[42] = static_cast<uint8_t>(dataSize >> 16);
        p[43] = static_cast<uint8_t>(dataSize >> 24);
        if (dataSize != 0u)
            std::memcpy(p + 44, pcm, dataSize);
        return wav;
    }
}

namespace ps2_vag
{
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate);
}

struct PS2AudioBackend::Impl
{
    struct TrackedSound
    {
        Sound snd;
        uint32_t sampleKey;
    };
    std::vector<TrackedSound> activeSounds;
    // Stereo samples at 48 kHz: 96 per millisecond. Playback starts with about 85 ms queued, so one
    // or two late frames (17-33 ms each) drain the queue without an audible gap; 43 ms was too
    // little with the device pulling 10.7 ms blocks. The ring holds about 340 ms, and an overflow
    // trims back to the same 85 ms.
    static constexpr size_t kRingSamples = 32768u;
    std::array<int16_t, kRingSamples> ring{};
    size_t read = 0u;
    size_t queued = 0u;
    static constexpr size_t kPrimeSamples = 8192u;
    // After an underrun, playback resumes with about 21 ms queued instead of the start-up 85 ms:
    // the hitch that caused it is already over, and waiting for 85 ms only made every cut longer.
    static constexpr size_t kRePrimeSamples = 2048u;
    static constexpr int32_t kFadeFrames = 128;
    // Gap concealment: while the queue is empty, replay the last kGrainFrames of output in a loop
    // (crossfaded at each wrap) and fade it out over kConcealFrames, instead of going silent at once.
    // A short game hitch then sounds like a brief smear rather than a cut.
    static constexpr int32_t kGrainFrames = 960;      // 20 ms
    static constexpr int32_t kConcealFrames = 14400;  // 300 ms
    std::array<int16_t, kGrainFrames * 2> history{};
    int32_t historyPos = 0;
    int32_t concealPos = 0;
    int32_t concealHead = 0;
    bool conceal = true; // PS2X_AUDIO_CONCEAL=0: old behavior (85 ms re-prime, fade to silence)
    // Rate control: audio is produced per emulated field, so a few seconds of slow emulation (VU1
    // spikes, 57 fields/s in Dirge's fight) drain the queue faster than it fills. Only when the queue
    // is about to run dry does playback stretch the sound (up to 6% longer, pitch kept: see the
    // splices below), so a short slowdown costs a slight tempo change instead of a cut; otherwise
    // the rate is exactly the game's. PS2X_AUDIO_STRETCH=0: fixed rate always.
    bool stretch = true;
    // Slowing down keeps the pitch: instead of reading the ring slower, playback replays the last
    // kSpliceBack frames it played (crossfaded in and out over kSpliceFade), which adds
    // kSpliceBack - 2 * kSpliceFade frames of sound at the original pitch, as PCSX2's time
    // stretch does. Splices are spaced so the tempo drops by at most 6%.
    static constexpr int32_t kSpliceBack = 720;  // 15 ms
    static constexpr int32_t kSpliceFade = 96;   // 2 ms
    int32_t splicePhase = 0;                     // 0 none, 1 fade to replay, 2 replay, 3 fade back
    int32_t spliceStep = 0;
    int32_t sinceSplice = 0;
    double slowWanted = 0.0;                     // fraction of extra time wanted (0..0.06)
    double readStep = 1.0;
    double readFrac = 0.0;
    bool primed = false;
    int32_t fadePosition = 0;
    int32_t gapRemaining = 0;
    std::array<int16_t, 2> lastOutput{};
    std::array<int16_t, 2> fadeSource{};
    std::array<int16_t, 2> gapSource{};
    std::mutex pcmMutex;
    AudioStream stream{};
    bool streamReady = false;
    std::FILE *dump = nullptr;
    std::FILE *outDump = nullptr;
    uint32_t dumpBytes = 0u;
    bool statsEnabled = false;
    bool everPrimed = false;
    uint64_t producedFrames = 0u;
    uint64_t requestedFrames = 0u;
    uint64_t missingFrames = 0u;
    uint64_t underruns = 0u;
    uint64_t droppedFrames = 0u;
    uint64_t slowedFrames = 0u;  // output frames played below the game's rate (rate control)
    std::chrono::steady_clock::time_point statsStart = std::chrono::steady_clock::now();
    static Impl *streamOwner;

    static void audioCallback(void *buffer, unsigned int frames)
    {
        auto *output = static_cast<int16_t *>(buffer);
        const size_t requested = static_cast<size_t>(frames) * 2u;
        std::fill_n(output, requested, int16_t{0});
        Impl *owner = streamOwner;
        if (!owner)
            return;
        std::lock_guard<std::mutex> lock(owner->pcmMutex);
        owner->requestedFrames += frames;
        if (owner->stretch && owner->primed)
        {
            // Aim at the start-up queue; the step moves a little per block (10.7 ms), so the pitch
            // glides rather than jumps.
            // The original pitch is kept whenever possible: the rate is exactly 1.0 unless the
            // queue is below 40% of the start-up 85 ms (a cut is near), and only then slows, by
            // up to 6%. A queue that grows past 2.5x drains at 1% faster (the ring would trim it).
            const double queued = static_cast<double>(owner->queued);
            double want = 1.0;
            owner->slowWanted = 0.0;
            if (queued < kPrimeSamples * 0.4)
                owner->slowWanted = std::min(0.06, 0.1 * (kPrimeSamples * 0.4 - queued) / (kPrimeSamples * 0.4) + 0.02);
            else if (queued > kPrimeSamples * 2.5)
                want = 1.01;
            owner->readStep += (want - owner->readStep) * 0.2;
            if (std::abs(owner->readStep - 1.0) < 0.0005 && want == 1.0)
                owner->readStep = 1.0;
        }
        else
        {
            owner->readStep = 1.0;
            owner->slowWanted = 0.0;
        }
        if (owner->slowWanted > 0.0 || owner->splicePhase != 0)
            owner->slowedFrames += frames;
        for (size_t frame = 0u; frame < frames; ++frame)
        {
            if (!owner->primed && owner->queued >= (owner->everPrimed && owner->conceal ? kRePrimeSamples : kPrimeSamples))
            {
                owner->primed = true;
                owner->everPrimed = true;
                owner->fadePosition = 0;
                owner->fadeSource = owner->lastOutput;
            }
            if (owner->primed && owner->queued < 2u)
            {
                owner->primed = false;
                ++owner->underruns;
                owner->gapRemaining = kFadeFrames;
                owner->gapSource = owner->lastOutput;
                owner->concealPos = 0;
                owner->concealHead = 0;
            }
            if (owner->primed)
            {
                owner->fadePosition = std::min(owner->fadePosition + 1, kFadeFrames);
                const int32_t fade = owner->fadePosition;
                const double frac = owner->queued >= 4u ? owner->readFrac : 0.0;
                // Start a splice when the queue is short: spaced so it adds slowWanted of time.
                if (owner->splicePhase == 0)
                {
                    ++owner->sinceSplice;
                    const int32_t added = kSpliceBack - 2 * kSpliceFade;
                    if (owner->slowWanted > 0.0 && owner->sinceSplice >= static_cast<int32_t>(added / owner->slowWanted) - kSpliceBack)
                    {
                        owner->splicePhase = 1;
                        owner->spliceStep = 0;
                        owner->sinceSplice = 0;
                    }
                }
                const int32_t phase = owner->splicePhase;
                const int32_t k = owner->spliceStep;
                const int32_t back = (owner->historyPos + kGrainFrames - kSpliceBack) % kGrainFrames;
                for (size_t channel = 0u; channel < 2u; ++channel)
                {
                    const int32_t a = owner->ring[(owner->read + channel) % kRingSamples];
                    const int32_t b = owner->queued >= 4u ? owner->ring[(owner->read + 2u + channel) % kRingSamples] : a;
                    int32_t sample = static_cast<int32_t>(a + (b - a) * frac);
                    const int32_t replay = owner->history[back * 2 + channel];
                    if (phase == 1)
                        sample = (sample * (kSpliceFade - k) + replay * k) / kSpliceFade;
                    else if (phase == 2)
                        sample = replay;
                    else if (phase == 3)
                        sample = (replay * (kSpliceFade - k) + sample * k) / kSpliceFade;
                    owner->lastOutput[channel] = static_cast<int16_t>((sample * fade + owner->fadeSource[channel] * (kFadeFrames - fade)) / kFadeFrames);
                    owner->history[owner->historyPos * 2 + channel] = owner->lastOutput[channel];
                }
                owner->historyPos = (owner->historyPos + 1) % kGrainFrames;
                if (phase != 0)
                {
                    const int32_t length = phase == 2 ? kSpliceBack - 2 * kSpliceFade : kSpliceFade;
                    if (++owner->spliceStep >= length)
                    {
                        owner->splicePhase = phase == 3 ? 0 : phase + 1;
                        owner->spliceStep = 0;
                    }
                }
                // While replaying, the ring is not read: that is the time the splice adds.
                if (phase != 2)
                    owner->readFrac += owner->readStep;
                while (owner->readFrac >= 1.0 && owner->queued >= 2u)
                {
                    owner->readFrac -= 1.0;
                    owner->read = (owner->read + 2u) % kRingSamples;
                    owner->queued -= 2u;
                }
                if (owner->readFrac >= 1.0)
                    owner->readFrac = 0.0;
            }
            else if (owner->everPrimed && !owner->conceal)
            {
                ++owner->missingFrames;
                owner->gapRemaining = std::max(owner->gapRemaining - 1, 0);
                for (size_t channel = 0u; channel < 2u; ++channel)
                    owner->lastOutput[channel] = static_cast<int16_t>(owner->gapSource[channel] * owner->gapRemaining / kFadeFrames);
            }
            else if (owner->everPrimed)
            {
                ++owner->missingFrames;
                // Loop the last 20 ms (oldest sample first) with a crossfaded loop point: over the
                // grain's last kFadeFrames the start of the grain fades in, then the playhead jumps
                // back to kFadeFrames, which continues that faded-in signal without a step. Entry
                // fades from the last live sample; the whole loop fades out over 300 ms.
                const int32_t pos = owner->concealPos;
                const int32_t head = owner->concealHead;
                const int32_t gain = std::max(kConcealFrames - pos, 0);
                for (size_t channel = 0u; channel < 2u; ++channel)
                {
                    const auto at = [&](int32_t i)
                    { return static_cast<int32_t>(owner->history[((owner->historyPos + i) % kGrainFrames) * 2 + channel]); };
                    int32_t sample = at(head);
                    const int32_t loopStart = kGrainFrames - kFadeFrames;
                    if (head >= loopStart)
                    {
                        const int32_t w = head - loopStart;
                        sample = (sample * (kFadeFrames - w) + at(head - loopStart) * w) / kFadeFrames;
                    }
                    if (pos < kFadeFrames)
                        sample = (sample * pos + owner->gapSource[channel] * (kFadeFrames - pos)) / kFadeFrames;
                    owner->lastOutput[channel] = static_cast<int16_t>(static_cast<int64_t>(sample) * gain / kConcealFrames);
                }
                owner->concealHead = head + 1 >= kGrainFrames ? kFadeFrames : head + 1;
                owner->concealPos = std::min(pos + 1, kConcealFrames);
            }
            output[frame * 2u] = owner->lastOutput[0];
            output[frame * 2u + 1u] = owner->lastOutput[1];
        }
        // PS2X_AUDIO_OUT_DUMP: raw 48 kHz stereo s16 of what the device plays (debugging only;
        // writes from the audio callback).
        if (owner->outDump)
            std::fwrite(output, sizeof(int16_t), requested, owner->outDump);
    }

    void finishDump()
    {
        if (!dump)
            return;
        auto header = buildWavFromPcm(nullptr, 0u, 48000u);
        header[22] = 2;
        header[28] = 0;
        header[29] = 0xEE;
        header[30] = 2;
        header[32] = 4;
        const uint32_t length = dumpBytes + 36u;
        for (uint32_t byte = 0u; byte < 4u; ++byte)
        {
            header[4u + byte] = static_cast<uint8_t>(length >> (byte * 8u));
            header[40u + byte] = static_cast<uint8_t>(dumpBytes >> (byte * 8u));
        }
        std::fseek(dump, 0, SEEK_SET);
        std::fwrite(header.data(), 1u, header.size(), dump);
        std::fclose(dump);
        dump = nullptr;
    }
};

PS2AudioBackend::Impl *PS2AudioBackend::Impl::streamOwner = nullptr;

PS2AudioBackend::PS2AudioBackend() : m_impl(std::make_unique<Impl>())
{
}

PS2AudioBackend::~PS2AudioBackend()
{
    if (m_impl)
    {
        stopAll();
        setAudioReady(false);
    }
}

void PS2AudioBackend::setAudioReady(bool ready)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (ready == m_audioReady)
        return;
    m_audioReady = ready;
#if !defined(PLATFORM_VITA)
    if (ready)
    {
        const char *audioStats = std::getenv("PS2X_AUDIO_STATS");
        m_impl->statsEnabled = audioStats && *audioStats && *audioStats != '0';
        m_impl->statsStart = std::chrono::steady_clock::now();
        const char *audioMute = std::getenv("PS2X_AUDIO_MUTE");
        if (audioMute && *audioMute && *audioMute != '0')
            SetMasterVolume(0.0f);
        const char *concealValue = std::getenv("PS2X_AUDIO_CONCEAL");
        m_impl->conceal = !(concealValue && *concealValue == '0');
        const char *stretchValue = std::getenv("PS2X_AUDIO_STRETCH");
        m_impl->stretch = !(stretchValue && *stretchValue == '0');
        if (const char *outPath = std::getenv("PS2X_AUDIO_OUT_DUMP"); outPath && *outPath)
            m_impl->outDump = std::fopen(outPath, "wb");
        SetAudioStreamBufferSizeDefault(512);
        m_impl->stream = LoadAudioStream(48000, 16, 2);
        m_impl->streamReady = m_impl->stream.buffer != nullptr;
        if (m_impl->streamReady)
        {
            Impl::streamOwner = m_impl.get();
            SetAudioStreamCallback(m_impl->stream, &Impl::audioCallback);
            PlayAudioStream(m_impl->stream);
        }
        const char *path = std::getenv("PS2X_AUDIO_DUMP");
        if (path && *path)
        {
            m_impl->dump = std::fopen(path, "wb");
            if (m_impl->dump)
            {
                std::array<uint8_t, 44> header{};
                std::fwrite(header.data(), 1u, header.size(), m_impl->dump);
                m_impl->dumpBytes = 0u;
            }
        }
    }
    else
    {
        if (m_impl->streamReady)
        {
            StopAudioStream(m_impl->stream);
            UnloadAudioStream(m_impl->stream);
            m_impl->streamReady = false;
            Impl::streamOwner = nullptr;
        }
        if (m_impl->outDump)
        {
            std::fclose(m_impl->outDump);
            m_impl->outDump = nullptr;
        }
        m_impl->finishDump();
        resetPcm();
    }
#endif
}

void PS2AudioBackend::resetPcm()
{
    std::lock_guard<std::mutex> lock(m_impl->pcmMutex);
    m_impl->read = 0u;
    m_impl->queued = 0u;
    m_impl->primed = false;
    m_impl->fadePosition = 0;
    m_impl->gapRemaining = 0;
    m_impl->lastOutput = {};
    m_impl->fadeSource = {};
    m_impl->gapSource = {};
    m_impl->history = {};
    m_impl->historyPos = 0;
    m_impl->concealPos = 0;
    m_impl->concealHead = 0;
    m_impl->everPrimed = false;
    m_impl->producedFrames = 0u;
    m_impl->requestedFrames = 0u;
    m_impl->missingFrames = 0u;
    m_impl->underruns = 0u;
    m_impl->droppedFrames = 0u;
    m_impl->statsStart = std::chrono::steady_clock::now();
}

void PS2AudioBackend::submitPcm(std::span<const int16_t> samples)
{
    std::lock_guard<std::mutex> lock(m_impl->pcmMutex);
    if (m_impl->dump && samples.size_bytes() <= 0xFFFFFF00u - m_impl->dumpBytes)
    {
        const size_t written = std::fwrite(samples.data(), sizeof(int16_t), samples.size(), m_impl->dump);
        m_impl->dumpBytes += static_cast<uint32_t>(written * sizeof(int16_t));
    }
    if (!m_impl->streamReady)
        return;
    m_impl->producedFrames += samples.size() / 2u;
    if (samples.size() > Impl::kRingSamples)
        samples = samples.last(Impl::kRingSamples);
    if (m_impl->queued + samples.size() > Impl::kRingSamples)
    {
        const size_t drop = std::min(m_impl->queued, std::max(m_impl->queued + samples.size() - Impl::kRingSamples,
                                                          m_impl->queued > Impl::kPrimeSamples ? m_impl->queued - Impl::kPrimeSamples : 0u));
        m_impl->read = (m_impl->read + drop) % Impl::kRingSamples;
        m_impl->queued -= drop;
        m_impl->droppedFrames += drop / 2u;
        m_impl->fadePosition = 0;
        m_impl->fadeSource = m_impl->lastOutput;
    }
    const size_t write = (m_impl->read + m_impl->queued) % Impl::kRingSamples;
    for (size_t i = 0u; i < samples.size(); ++i)
        m_impl->ring[(write + i) % Impl::kRingSamples] = samples[i];
    m_impl->queued += samples.size();
    if (m_impl->statsEnabled)
    {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - m_impl->statsStart).count();
        if (elapsed >= 2.0)
        {
            std::fprintf(stderr, "[audio] %.0f produced/s, %.0f requested/s, %.2f ms queued, %llu missing frames, %llu underruns, %llu dropped frames, %llu slowed frames\n",
                         m_impl->producedFrames / elapsed, m_impl->requestedFrames / elapsed,
                         m_impl->queued / 96.0, static_cast<unsigned long long>(m_impl->missingFrames),
                         static_cast<unsigned long long>(m_impl->underruns), static_cast<unsigned long long>(m_impl->droppedFrames),
                         static_cast<unsigned long long>(m_impl->slowedFrames));
            m_impl->slowedFrames = 0u;
            m_impl->producedFrames = 0u;
            m_impl->requestedFrames = 0u;
            m_impl->missingFrames = 0u;
            m_impl->underruns = 0u;
            m_impl->droppedFrames = 0u;
            m_impl->statsStart = now;
        }
    }
}

void PS2AudioBackend::onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
{
    if (!rdram || sizeBytes < 48)
        return;

    const uint32_t physAddr = srcAddr & PS2_RAM_MASK;
    if (physAddr + sizeBytes > PS2_RAM_SIZE)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(rdram + physAddr, sizeBytes, pcm, sampleRate))
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = std::move(sample);
    m_mostRecentSampleKey = physAddr;
}

void PS2AudioBackend::onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr)
{
    if (!data || sizeBytes < 48)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(data, sizeBytes, pcm, sampleRate))
        return;

    const uint32_t physAddr = keyAddr & PS2_RAM_MASK;
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = sample;
    m_mostRecentSampleKey = physAddr;
    m_loadOrderSamples.push_back(std::move(sample));
    m_loadOrderSampleKeys.push_back(physAddr);
    constexpr size_t kMaxLoadOrderSamples = 32;
    if (m_loadOrderSamples.size() > kMaxLoadOrderSamples)
    {
        m_loadOrderSamples.erase(m_loadOrderSamples.begin());
        m_loadOrderSampleKeys.erase(m_loadOrderSampleKeys.begin());
    }
}

namespace
{
    constexpr uint32_t LIBSD_CMD_SET_VOICE = 0x8010u;
}

void PS2AudioBackend::onSoundCommand(uint32_t sid, uint32_t rpcNum,
                                     const uint8_t *sendBuf, uint32_t sendSize,
                                     uint8_t *recvBuf, uint32_t recvSize)
{
    if (sid != 0x80000701u)
        return;

    if ((rpcNum == LIBSD_CMD_SET_VOICE || (rpcNum & 0xFF00u) == 0x8100u) &&
        sendBuf && sendSize >= 20)
    {
        uint32_t sampleAddr = 0;
        uint32_t voiceIndex = 0xFFFFFFFFu;
        for (int vo = 4; vo >= 0 && voiceIndex == 0xFFFFFFFFu; vo -= 4)
        {
            if (vo < static_cast<int>(sendSize))
            {
                uint32_t v = 0;
                std::memcpy(&v, sendBuf + vo, sizeof(v));
                if (v < 24u)
                    voiceIndex = v;
            }
        }

        constexpr uint32_t kMinPlausibleAddr = 0x1000u;
        for (int off = 12; off <= 24 && sampleAddr == 0; off += 4)
        {
            if (sendSize >= static_cast<uint32_t>(off + 4))
            {
                uint32_t cand = 0;
                std::memcpy(&cand, sendBuf + off, sizeof(cand));
                if (cand >= kMinPlausibleAddr && (cand <= PS2_RAM_MASK || (cand & ~PS2_RAM_MASK) == 0))
                    sampleAddr = cand;
            }
        }
        if (sampleAddr == 0)
            sampleAddr = m_mostRecentSampleKey;

        float pitch = 1.0f;
        if (sendSize >= 12)
        {
            uint16_t pitchHalf = 0;
            std::memcpy(&pitchHalf, sendBuf + 8, sizeof(pitchHalf));
            if (pitchHalf != 0)
                pitch = 4096.0f / static_cast<float>(pitchHalf);
        }
        play(sampleAddr, pitch, 1.0f, voiceIndex);
    }
}

void PS2AudioBackend::play(uint32_t sampleAddr, float pitch, float volume, uint32_t voiceIndex)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample *sampleToPlay = nullptr;
    uint32_t sampleKey = 0;

    auto it = m_sampleBank.find(sampleAddr & PS2_RAM_MASK);
    if (it != m_sampleBank.end())
    {
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    else if (voiceIndex != 0xFFFFFFFFu &&
             voiceIndex < m_loadOrderSamples.size() &&
             voiceIndex < m_loadOrderSampleKeys.size())
    {
        sampleToPlay = &m_loadOrderSamples[voiceIndex];
        sampleKey = m_loadOrderSampleKeys[voiceIndex];
    }
    else
    {
        it = m_sampleBank.find(m_mostRecentSampleKey);
        if (it == m_sampleBank.end())
            return;
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    if (!sampleToPlay || sampleToPlay->pcm.empty())
        return;

    const bool isBgm = (sampleToPlay->pcm.size() > static_cast<size_t>(sampleToPlay->sampleRate * 5));
    playDecodedSample(sampleKey, *sampleToPlay, pitch, volume, isBgm);
}

void PS2AudioBackend::pruneFinishedSounds()
{
#if defined(PLATFORM_VITA)
    return;
#else
    auto &sounds = m_impl->activeSounds;
    auto it = sounds.begin();
    while (it != sounds.end())
    {
        if (!IsSoundPlaying(it->snd))
        {
            UnloadSound(it->snd);
            it = sounds.erase(it);
        }
        else
        {
            ++it;
        }
    }
#endif
}

void PS2AudioBackend::playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                                        bool isBgm)
{
#if defined(PLATFORM_VITA)
    (void)sampleKey;
    (void)sample;
    (void)pitch;
    (void)volume;
    (void)isBgm;
    return;
#else
    if (!m_audioReady || sample.pcm.empty())
        return;

    pruneFinishedSounds();

    for (const auto &t : m_impl->activeSounds)
    {
        if (t.sampleKey == sampleKey && IsSoundPlaying(t.snd))
            return;
    }

    auto &sounds = m_impl->activeSounds;
    if (isBgm)
    {
        for (auto it = sounds.begin(); it != sounds.end();)
        {
            if (IsSoundPlaying(it->snd))
            {
                StopSound(it->snd);
                UnloadSound(it->snd);
                it = sounds.erase(it);
            }
            else
                ++it;
        }
    }

    constexpr int kMaxConcurrentSounds = 4;
    while (static_cast<int>(sounds.size()) >= kMaxConcurrentSounds)
    {
        StopSound(sounds.front().snd);
        UnloadSound(sounds.front().snd);
        sounds.erase(sounds.begin());
    }

    std::vector<uint8_t> wav = buildWavFromPcm(sample.pcm.data(), sample.pcm.size(), sample.sampleRate);
    Wave wave = LoadWaveFromMemory(".wav", wav.data(), static_cast<int>(wav.size()));
    if (wave.frameCount <= 0)
        return;
    Sound snd = LoadSoundFromWave(wave);
    UnloadWave(wave);
    SetSoundPitch(snd, pitch);
    SetSoundVolume(snd, volume);
    m_impl->activeSounds.push_back({snd, sampleKey});
    PlaySound(snd);
#endif
}

void PS2AudioBackend::stop(uint32_t voiceId)
{
    (void)voiceId;
}

void PS2AudioBackend::stopAll()
{
    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(PLATFORM_VITA)
    return;
#else
    for (auto &t : m_impl->activeSounds)
    {
        StopSound(t.snd);
        UnloadSound(t.snd);
    }
    m_impl->activeSounds.clear();
#endif
}
