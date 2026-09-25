#pragma once

#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#ifndef VU1_FORCE_INLINE
#if defined(_MSC_VER)
#define VU1_FORCE_INLINE __forceinline
#else
#define VU1_FORCE_INLINE inline __attribute__((always_inline))
#endif
#endif

namespace vu1_impl
{
    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    inline int freeSlot(uint32_t validMask, uint32_t capacity)
    {
        const uint32_t freeMask = ~validMask & ((1u << capacity) - 1u);
        return freeMask == 0u ? -1 : std::countr_zero(freeMask);
    }
}

using namespace vu1_impl;

VU1_FORCE_INLINE float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

VU1_FORCE_INLINE float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
    {
        bits &= 0x80000000u;
    }
    else if (exponent == 0xFFu)
    {
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

VU1_FORCE_INLINE float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

VU1_FORCE_INLINE uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

VU1_FORCE_INLINE int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

VU1_FORCE_INLINE void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

VU1_FORCE_INLINE void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

VU1_FORCE_INLINE void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

VU1_FORCE_INLINE void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        long double exactResult = 0.0L;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

inline bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<long double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const long double q = operand(m_state.q);
    const long double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0L
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0L
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

VU1_FORCE_INLINE uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const long double magnitude = std::fabs(exactResult);
    const long double maximum = static_cast<long double>(std::numeric_limits<float>::max());
    const long double minimum = static_cast<long double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0L)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

inline uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        const long double exactProduct = static_cast<long double>(left) * static_cast<long double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

VU1_FORCE_INLINE void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    const int slot = freeSlot(m_flagValid, kMaxFlagEntries);
    if (slot < 0)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }

    FlagPipelineEntry *entry = &m_flagPipeline[static_cast<size_t>(slot)];
    *entry = {};
    entry->valid = true;
    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
    m_flagValid |= 1u << slot;
    noteReadyCycle(entry->readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDest(dst, result, dest);
}

VU1_FORCE_INLINE void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDestAcc(result, dest);
}

VU1_FORCE_INLINE void VU1Interpreter::queueFsset(uint16_t immediate)
{
    for (uint32_t mask = m_flagValid; mask != 0u; mask &= mask - 1u)
    {
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(std::countr_zero(mask))];
        if (entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    const int slot = freeSlot(m_flagValid, kMaxFlagEntries);
    if (slot < 0)
    {
        reportReservedInstruction(false, 0xFFFFFFFEu);
        return;
    }
    FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(slot)];
    entry = {};
    entry.valid = true;
    entry.issueCycle = m_cycle;
    entry.readyCycle = m_cycle + kFmacLatency;
    entry.status = static_cast<uint32_t>(immediate) & 0xFC0u;
    entry.writesSticky = true;
    m_flagValid |= 1u << slot;
    noteReadyCycle(entry.readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    const int slot = freeSlot(m_flagValid, kMaxFlagEntries);
    if (slot < 0)
    {
        reportReservedInstruction(true, 0xFFFFFFFDu);
        return;
    }
    FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(slot)];
    entry = {};
    entry.valid = true;
    entry.issueCycle = m_cycle;
    entry.readyCycle = m_cycle + kFmacLatency;
    entry.clip = m_workingClip;
    entry.writesClip = true;
    m_flagValid |= 1u << slot;
    noteReadyCycle(entry.readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    for (uint32_t mask = m_flagValid; mask != 0u; mask &= mask - 1u)
    {
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(std::countr_zero(mask))];
        if (entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    const int slot = freeSlot(m_flagValid, kMaxFlagEntries);
    if (slot < 0)
    {
        reportReservedInstruction(false, 0xFFFFFFFAu);
        return;
    }
    FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(slot)];
    entry = {};
    entry.valid = true;
    entry.issueCycle = m_cycle;
    entry.readyCycle = m_cycle + kFmacLatency;
    entry.clip = m_workingClip;
    entry.writesClip = true;
    m_flagValid |= 1u << slot;
    noteReadyCycle(entry.readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
    noteReadyCycle(m_fdiv.readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            noteReadyCycle(entry.readyCycle);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

VU1_FORCE_INLINE void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    const int slot = freeSlot(m_storeValid, kMaxPendingStores);
    if (slot < 0)
    {
        reportReservedInstruction(false, 0xFFFFFFFCu);
        return;
    }
    PendingStore &store = m_storePipeline[static_cast<size_t>(slot)];
    store.valid = true;
    store.readyCycle = m_cycle + 1u;
    store.address = address;
    store.laneMask = laneMask;
    std::copy(words, words + 4, store.words.begin());
    m_storeValid |= 1u << slot;
    noteReadyCycle(store.readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    const int slot = freeSlot(m_vfWriteValid, kMaxPendingVfWrites);
    if (slot < 0)
    {
        reportReservedInstruction(false, 0xFFFFFFF7u);
        return;
    }
    PendingVfWrite &write = m_vfWritePipeline[static_cast<size_t>(slot)];
    write = {};
    write.valid = true;
    write.readyCycle = m_cycle + latency;
    write.sequence = ++m_nextWriteSequence;
    write.reg = reg;
    write.laneMask = laneMask;
    std::copy(value, value + 4, write.value.begin());
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((laneMask & laneForComponent(component)) != 0u)
            m_vfLatestWrite[reg][component] = write.sequence;
    }
    m_vfWriteValid |= 1u << slot;
    noteReadyCycle(write.readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    const int slot = freeSlot(m_viWriteValid, kMaxPendingViWrites);
    if (slot < 0)
    {
        reportReservedInstruction(false, 0xFFFFFFF6u);
        return;
    }
    PendingViWrite &write = m_viWritePipeline[static_cast<size_t>(slot)];
    write = {};
    write.valid = true;
    write.readyCycle = m_cycle + latency;
    write.sequence = ++m_nextWriteSequence;
    write.reg = reg;
    write.value = value;
    m_viLatestWrite[reg] = write.sequence;
    m_viWriteValid |= 1u << slot;
    noteReadyCycle(write.readyCycle);
}

VU1_FORCE_INLINE void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    const int slot = freeSlot(m_accWriteValid, kMaxPendingAccWrites);
    if (slot < 0)
    {
        reportReservedInstruction(true, 0xFFFFFFF5u);
        return;
    }
    PendingAccWrite &write = m_accWritePipeline[static_cast<size_t>(slot)];
    write = {};
    write.valid = true;
    write.readyCycle = m_cycle + latency;
    write.sequence = ++m_nextWriteSequence;
    write.laneMask = laneMask;
    std::copy(value, value + 4, write.value.begin());
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((laneMask & laneForComponent(component)) != 0u)
            m_accLatestWrite[component] = write.sequence;
    }
    m_accWriteValid |= 1u << slot;
    noteReadyCycle(write.readyCycle);
}

inline void VU1Interpreter::commitReadyPipelines()
{
    if (m_cycle < m_nextCommitCycle)
        return;

    uint64_t next = std::numeric_limits<uint64_t>::max();

    for (uint32_t mask = m_flagValid; mask != 0u; mask &= mask - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(std::countr_zero(mask));
        FlagPipelineEntry &entry = m_flagPipeline[index];
        if (entry.readyCycle > m_cycle)
        {
            next = std::min(next, entry.readyCycle);
            continue;
        }

        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry = {};
        m_flagValid &= ~(1u << index);
    }

    if (m_fdiv.valid)
    {
        if (m_fdiv.readyCycle <= m_cycle)
        {
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }
        else
        {
            next = std::min(next, m_fdiv.readyCycle);
        }
    }

    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
            continue;
        if (entry.readyCycle <= m_cycle)
        {
            m_state.p = entry.value;
            entry = {};
        }
        else
        {
            next = std::min(next, entry.readyCycle);
        }
    }

    for (uint32_t mask = m_storeValid; mask != 0u; mask &= mask - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(std::countr_zero(mask));
        PendingStore &store = m_storePipeline[index];
        if (store.readyCycle > m_cycle)
        {
            next = std::min(next, store.readyCycle);
            continue;
        }
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((store.laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = store.words[component];
            }
            std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
        }
        store = {};
        m_storeValid &= ~(1u << index);
    }

    for (uint32_t mask = m_vfWriteValid; mask != 0u; mask &= mask - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(std::countr_zero(mask));
        PendingVfWrite &write = m_vfWritePipeline[index];
        if (write.readyCycle > m_cycle)
        {
            next = std::min(next, write.readyCycle);
            continue;
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_vfLatestWrite[write.reg][component] == write.sequence)
            {
                m_state.vf[write.reg][component] = write.value[component];
            }
        }
        write.valid = false;
        m_vfWriteValid &= ~(1u << index);
    }

    for (uint32_t mask = m_viWriteValid; mask != 0u; mask &= mask - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(std::countr_zero(mask));
        PendingViWrite &write = m_viWritePipeline[index];
        if (write.readyCycle > m_cycle)
        {
            next = std::min(next, write.readyCycle);
            continue;
        }
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
        write.valid = false;
        m_viWriteValid &= ~(1u << index);
    }

    for (uint32_t mask = m_accWriteValid; mask != 0u; mask &= mask - 1u)
    {
        const uint32_t index = static_cast<uint32_t>(std::countr_zero(mask));
        PendingAccWrite &write = m_accWritePipeline[index];
        if (write.readyCycle > m_cycle)
        {
            next = std::min(next, write.readyCycle);
            continue;
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_accLatestWrite[component] == write.sequence)
            {
                m_state.acc[component] = write.value[component];
            }
        }
        write.valid = false;
        m_accWriteValid &= ~(1u << index);
    }

    m_nextCommitCycle = next;
}

inline void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        const uint32_t firstSource = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) % m_activeVuDataSize;
        if (firstSource + 16u <= m_activeVuDataSize)
        {
            std::memcpy(m_xgkick.packet.data() + m_xgkick.copiedBytes, m_activeVuData + firstSource, 16u);
        }
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
            {
                const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes + i) % m_activeVuDataSize;
                m_xgkick.packet[m_xgkick.copiedBytes + i] = m_activeVuData[source];
            }
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u)
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
            {
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

inline void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    m_xgkick.active = false;
}

inline void VU1Interpreter::resetXgkick()
{
    m_xgkick.sourceAddress = 0u;
    m_xgkick.totalBytes = 0u;
    m_xgkick.copiedBytes = 0u;
    m_xgkick.currentTagEnd = 0u;
    m_xgkick.cycleCredit = 0u;
    m_xgkick.issueCycle = 0u;
    m_xgkick.active = false;
    m_xgkick.currentTagEop = false;
}

inline void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    resetXgkick();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

VU1_FORCE_INLINE void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    commitReadyPipelines();
    progressXgkick();
}

VU1_FORCE_INLINE void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
    {
        if (!m_xgkick.active && m_nextCommitCycle > m_cycle + 1u)
        {
            const uint64_t quietUntil = std::min(targetCycle, m_nextCommitCycle) - 1u;
            if (quietUntil > m_cycle)
                m_cycle = quietUntil;
        }
        advanceOneCycle();
    }
}

VU1_FORCE_INLINE uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    const InstructionUsage *usages[2] = {
        &decoded.upperUsage,
        &decoded.lowerUsage};
    for (const InstructionUsage *usage : usages)
    {
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((access.lanes & laneForComponent(component)) != 0u)
                    ready = std::max(ready, m_vfReady[access.reg][component]);
            }
        }
        for (uint32_t mask = usage->viRead & 0xFFFEu; mask != 0u; mask &= mask - 1u)
            ready = std::max(ready, m_viReady[static_cast<size_t>(std::countr_zero(mask))]);
        if (usage->accRead != 0u)
        {
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((usage->accRead & laneForComponent(component)) != 0u)
                    ready = std::max(ready, m_accReady[component]);
            }
        }
    }

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

VU1_FORCE_INLINE void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((lowerWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[lowerWrite.reg][component] = m_cycle + latency;
        }
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[upperWrite.reg][component] = m_cycle + latency;
        }
    }

    for (uint32_t mask = decoded.lowerUsage.viWrite & 0xFFFEu; mask != 0u; mask &= mask - 1u)
        m_viReady[static_cast<size_t>(std::countr_zero(mask))] = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency);
    if (decoded.upperUsage.accWrite != 0u)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
                m_accReady[component] = m_cycle + kAccForwardLatency;
        }
    }
}


namespace vu1_impl
{
    inline int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
inline void VU1Interpreter::execUpper(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    float *vd = m_state.vf[fd];
    float normalizedVs[4];
    float normalizedVt[4];
    float normalizedAcc[4];
    for (uint32_t component = 0; component < 4u; ++component)
    {
        normalizedVs[component] = normalizeOperand(m_state.vf[fs][component]);
        normalizedVt[component] = normalizeOperand(m_state.vf[ft][component]);
        normalizedAcc[component] = normalizeOperand(m_state.acc[component]);
    }
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = acc[0] - vs[1] * vt[2];
        result[1] = acc[1] - vs[2] * vt[0];
        result[2] = acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyFmacDest(vd, result, dest);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1D: // ABS
            for (int c = 0; c < 4; c++)
                result[c] = std::fabs(vs[c]);
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1F: // CLIP
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

            const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };

            uint32_t flags = 0u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x00000000u))
                flags |= 0x01u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x80000000u))
                flags |= 0x02u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x00000000u))
                flags |= 0x04u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x80000000u))
                flags |= 0x08u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x00000000u))
                flags |= 0x10u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x80000000u))
                flags |= 0x20u;
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacDestAcc(result, dest);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}

namespace vu1_impl
{
    inline float vuEatan(float value)
    {
        constexpr float coefficients[] = {
            0.999999344348907f,
            -0.333298563957214f,
            0.199465364217758f,
            -0.13085337519646f,
            0.096420042216778f,
            -0.055909886956215f,
            0.021861229091883f,
            -0.004054057877511f};
        constexpr float quarterPi = 0.785398185253143f;

        const float squared = value * value;
        float polynomial = coefficients[7];
        for (int index = 6; index >= 0; --index)
            polynomial = coefficients[index] + squared * polynomial;
        return quarterPi + value * polynomial;
    }

    inline float vuEsin(float value)
    {
        constexpr float coefficients[] = {
            1.0f,
            -0.166666567325592f,
            0.008333025500178f,
            -0.000198074136279f,
            0.000002601886990f};

        const float squared = value * value;
        float polynomial = coefficients[4];
        for (int index = 3; index >= 0; --index)
            polynomial = coefficients[index] + squared * polynomial;
        return value * polynomial;
    }

    inline float vuEexp(float value)
    {
        constexpr float coefficients[] = {
            0.249998688697815f,
            0.031257584691048f,
            0.002591371303424f,
            0.000171562001924f,
            0.000005430199963f,
            0.000000690600018f};

        float polynomial = coefficients[5];
        for (int index = 4; index >= 0; --index)
            polynomial = coefficients[index] + value * polynomial;
        polynomial = 1.0f + value * polynomial;
        polynomial *= polynomial;
        polynomial *= polynomial;
        return polynomial != 0.0f ? 1.0f / polynomial : std::numeric_limits<float>::max();
    }
}

// ============================================================================
// Lower instructions
// ============================================================================
inline void VU1Interpreter::execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr)
{
    (void)upperInstr;
    if (instr == 0x00000000 || instr == 0x8000033C) // NOP
        return;

    uint8_t opHi = (instr >> 25) & 0x7F;
    const uint32_t pcMask = microAddressMask();

    // The lower instruction encoding uses bits 31:25 for the primary opcode
    switch (opHi)
    {
    case 0x00: // LQ (Load Quadword from VU data memory)
    {
        uint8_t it = FT(instr);  // VF destination
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            float tmp[4];
            std::memcpy(tmp, vuData + addr, 16);
            applyDest(m_state.vf[it], tmp, dest);
        }
        return;
    }
    case 0x01: // SQ (Store Quadword to VU data memory)
    {
        uint8_t is = FS(instr);  // VF source
        uint8_t it = VIT(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[it] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            uint32_t words[4]{};
            std::memcpy(words, m_state.vf[is], sizeof(words));
            queueStore(addr, words, dest);
        }
        return;
    }
    case 0x04: // ILW (Integer Load Word from VU data memory)
    {
        uint8_t it = VIT(instr); // VI destination
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            int comp = 0;
            if (dest & 0x8)
                comp = 0;
            else if (dest & 0x4)
                comp = 1;
            else if (dest & 0x2)
                comp = 2;
            else
                comp = 3;
            uint32_t v;
            std::memcpy(&v, vuData + addr + comp * 4, 4);
            if (it != 0)
                m_state.vi[it] = (int32_t)(int16_t)(v & 0xFFFF);
        }
        return;
    }
    case 0x05: // ISW (Integer Store Word to VU data memory)
    {
        uint8_t it = VIT(instr); // VI source
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            const uint32_t val = static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[it] & 0xFFFF));
            const uint32_t words[4] = {val, val, val, val};
            queueStore(addr, words, dest);
        }
        return;
    }
    case 0x08: // IADDIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] + imm);
        return;
    }
    case 0x09: // ISUBIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] - imm);
        return;
    }
    case 0x10: // FCEQ
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & 0xFFFFFF) == imm24) ? 1 : 0;
        return;
    }
    case 0x11: // FCSET
    {
        queueFcset(instr & 0xFFFFFFu);
        return;
    }
    case 0x12: // FCAND
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & imm24) != 0) ? 1 : 0;
        return;
    }
    case 0x13: // FCOR
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip | imm24) == 0xFFFFFF) ? 1 : 0;
        return;
    }
    case 0x14: // FSEQ
    {
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = ((m_state.status & 0xFFFu) == imm12) ? 1 : 0;
        return;
    }
    case 0x15: // FSSET
    {
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        queueFsset(imm12);
        return;
    }
    case 0x16: // FSAND
    {
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>((m_state.status & 0xFFFu) & imm12);
        return;
    }
    case 0x17: // FSOR
    {
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>((m_state.status & 0xFFFu) | imm12);
        return;
    }
    case 0x18: // FMEQ
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = ((m_state.mac & 0xFFFF) == (uint32_t)(uint16_t)m_state.vi[is]) ? 1 : 0;
        return;
    }
    case 0x1A: // FMAND
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac & (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    case 0x1B: // FMOR
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac | (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    case 0x1C: // FCGET
    {
        const uint8_t it = VIT(instr);
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>(m_state.clip & 0x0FFFu);
        return;
    }
    case 0x20: // B (unconditional branch)
    {
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x21: // BAL (Branch and link)
    {
        uint8_t it = VIT(instr);
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x24: // JR
    {
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)readBranchVi(is) * 8u) & pcMask;
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x25: // JALR
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)readBranchVi(is) * 8u) & pcMask;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x28: // IBEQ
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) == (int16_t)readBranchVi(it))
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x29: // IBNE
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) != (int16_t)readBranchVi(it))
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2C: // IBLTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) < 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2D: // IBGTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) > 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2E: // IBLEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) <= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2F: // IBGEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) >= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }

    case 0x40: // Lower1 / lower special. Bit31 set; low 6 bits select integer or special op.
    {
        const uint8_t funct = instr & 0x3Fu;
        const uint8_t vfT = FT(instr);
        const uint8_t vfS = FS(instr);
        const uint8_t viT = VIT(instr);
        const uint8_t viS = VIS(instr);
        const uint8_t viD = VID(instr);
        const uint8_t dest = (instr >> 21) & 0xF;

        switch (funct)
        {
        case 0x30: // IADD
            if (viD != 0)
                m_state.vi[viD] = (int16_t)(m_state.vi[viS] + m_state.vi[viT]);
            return;
        case 0x31: // ISUB
            if (viD != 0)
                m_state.vi[viD] = (int16_t)(m_state.vi[viS] - m_state.vi[viT]);
            return;
        case 0x32: // IADDI
        {
            int16_t imm5 = (int16_t)((int32_t)((instr >> 6) & 0x1F) << 27 >> 27);
            if (viT != 0)
                m_state.vi[viT] = (int16_t)(m_state.vi[viS] + imm5);
            return;
        }
        case 0x34: // IAND
            if (viD != 0)
                m_state.vi[viD] = m_state.vi[viS] & m_state.vi[viT];
            return;
        case 0x35: // IOR
            if (viD != 0)
                m_state.vi[viD] = m_state.vi[viS] | m_state.vi[viT];
            return;

        case 0x3C:
        case 0x3D:
        case 0x3E:
        case 0x3F: // Lower1 special. Dobie decodes this as (instr & 3) | ((instr >> 4) & 0x7C).
        {
            const uint8_t funct2 = (uint8_t)((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
            switch (funct2)
            {
            case 0x30: // MOVE
            {
                float tmp[4];
                std::memcpy(tmp, m_state.vf[vfS], 16);
                applyDest(m_state.vf[vfT], tmp, dest);
                return;
            }
            case 0x31: // MR32 (rotate right by 32 bits = shift xyzw -> yzwx)
            {
                float tmp[4] = {m_state.vf[vfS][1], m_state.vf[vfS][2], m_state.vf[vfS][3], m_state.vf[vfS][0]};
                applyDest(m_state.vf[vfT], tmp, dest);
                return;
            }
            case 0x34: // LQI (Load Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[vfT], tmp, dest);
                }
                if (viS != 0)
                    m_state.vi[viS] = (int16_t)(m_state.vi[viS] + 1);
                return;
            }
            case 0x35: // SQI (Store Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, m_state.vf[vfS], sizeof(words));
                    queueStore(addr, words, dest);
                }
                if (viT != 0)
                    m_state.vi[viT] = (int16_t)(m_state.vi[viT] + 1);
                return;
            }
            case 0x36: // LQD (Load Quadword, pre-decrement)
            {
                if (viS != 0)
                    m_state.vi[viS] = (int16_t)(m_state.vi[viS] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[vfT], tmp, dest);
                }
                return;
            }
            case 0x37: // SQD (Store Quadword, pre-decrement)
            {
                if (viT != 0)
                    m_state.vi[viT] = (int16_t)(m_state.vi[viT] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, m_state.vf[vfS], sizeof(words));
                    queueStore(addr, words, dest);
                }
                return;
            }
            case 0x38: // DIV
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                const float num = normalizeOperand(m_state.vf[vfS][fsf]);
                const float den = normalizeOperand(m_state.vf[vfT][ftf]);
                uint32_t statusDi = 0u;
                float result = 0.0f;
                if (den == 0.0f)
                {
                    statusDi = num == 0.0f ? 0x10u : 0x20u;
                    result = std::signbit(num) != std::signbit(den)
                                 ? -std::numeric_limits<float>::max()
                                 : std::numeric_limits<float>::max();
                }
                else
                {
                    result = num / den;
                }
                uint32_t ignoredFlags = 0u;
                result = normalizeResult(result, ignoredFlags);
                queueQ(result, 7u, statusDi);
                return;
            }
            case 0x39: // SQRT
            {
                int ftf = (instr >> 23) & 0x3;
                const float val = normalizeOperand(m_state.vf[vfT][ftf]);
                queueQ(std::sqrt(std::fabs(val)), 7u,
                       val < 0.0f ? 0x10u : 0u);
                return;
            }
            case 0x3A: // RSQRT
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                const float num = normalizeOperand(m_state.vf[vfS][fsf]);
                const float radicand = normalizeOperand(m_state.vf[vfT][ftf]);
                const float den = std::sqrt(std::fabs(radicand));
                uint32_t statusDi = radicand < 0.0f ? 0x10u : 0u;
                float result = 0.0f;
                if (den != 0.0f)
                    result = num / den;
                else
                {
                    statusDi = num == 0.0f ? 0x10u : 0x20u;
                    result = std::signbit(num)
                                 ? -std::numeric_limits<float>::max()
                                 : std::numeric_limits<float>::max();
                }
                uint32_t ignoredFlags = 0u;
                result = normalizeResult(result, ignoredFlags);
                queueQ(result, 13u, statusDi);
                return;
            }
            case 0x3B: // WAITQ
                return;
            case 0x3C: // MTIR (Move To Integer Register)
            {
                // MTIR encodes a two-bit fsf component selector in bits
                // 22:21. It is not a four-bit destination mask.
                const uint32_t comp = (instr >> 21) & 0x3u;
                uint32_t fval;
                std::memcpy(&fval, &m_state.vf[vfS][comp], 4);
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(int16_t)(fval & 0xFFFF);
                return;
            }
            case 0x3D: // MFIR (Move From Integer Register)
            {
                float result[4];
                int32_t val = (int32_t)(int16_t)(m_state.vi[viS] & 0xFFFF);
                std::memcpy(&result[0], &val, 4);
                result[1] = result[0];
                result[2] = result[0];
                result[3] = result[0];
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x3E: // ILWR - integer load word from address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    int comp = 0;
                    if (dest & 0x8)
                        comp = 0;
                    else if (dest & 0x4)
                        comp = 1;
                    else if (dest & 0x2)
                        comp = 2;
                    else
                        comp = 3;
                    uint32_t v;
                    std::memcpy(&v, vuData + addr + comp * 4, 4);
                    if (viT != 0)
                        m_state.vi[viT] = (int32_t)(int16_t)(v & 0xFFFF);
                }
                return;
            }
            case 0x3F: // ISWR - integer store word to address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    const uint32_t val =
                        static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viT] & 0xFFFF));
                    const uint32_t words[4] = {val, val, val, val};
                    queueStore(addr, words, dest);
                }
                return;
            }
            case 0x40: // RNEXT
            {
                const uint32_t x = (m_state.r >> 4) & 1u;
                const uint32_t y = (m_state.r >> 22) & 1u;
                m_state.r = ((m_state.r << 1) ^ x ^ y) & 0x007FFFFFu;
                m_state.r |= 0x3F800000u;
                float value = 0.0f;
                std::memcpy(&value, &m_state.r, sizeof(value));
                const float result[4] = {value, value, value, value};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x41: // RGET
            {
                float value = 0.0f;
                std::memcpy(&value, &m_state.r, sizeof(value));
                const float result[4] = {value, value, value, value};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x42: // RINIT
            {
                const uint32_t component = (instr >> 21) & 3u;
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[vfS][component], sizeof(bits));
                m_state.r = 0x3F800000u | (bits & 0x007FFFFFu);
                return;
            }
            case 0x43: // RXOR
            {
                const uint32_t component = (instr >> 21) & 3u;
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[vfS][component], sizeof(bits));
                m_state.r = 0x3F800000u | ((m_state.r ^ bits) & 0x007FFFFFu);
                return;
            }
            case 0x64: // MFP (Move From P register)
            {
                float result[4] = {m_state.p, m_state.p, m_state.p, m_state.p};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x68: // XTOP - move current VIF1 TOP into VI register
            {
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(m_state.top & 0x3FFu);
                return;
            }
            case 0x69: // XITOP - move current VIF1 ITOP into VI register
            {
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(m_state.itop & 0x3FFu);
                return;
            }
            case 0x6C: // XGKICK - send GIF packet from VU1 data memory
                startXgkick(static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viS])));
                return;
            case 0x70: // ESADD
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                queueP(x * x + y * y + z * z, 11u);
                return;
            }
            case 0x71: // ERSADD
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                const float sum = x * x + y * y + z * z;
                queueP(sum != 0.0f ? 1.0f / sum : sum, 18u);
                return;
            }
            case 0x72: // ELENG
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                queueP(std::sqrt(x * x + y * y + z * z), 18u);
                return;
            }
            case 0x73: // ERLENG
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                const float len = std::sqrt(x * x + y * y + z * z);
                queueP(len != 0.0f ? 1.0f / len : len, 24u);
                return;
            }
            case 0x74: // EATANxy
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                queueP(x != 0.0f ? vuEatan(y / x) : 0.0f, 54u);
                return;
            }
            case 0x75: // EATANxz
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                queueP(x != 0.0f ? vuEatan(z / x) : 0.0f, 54u);
                return;
            }
            case 0x76: // ESUM
            {
                float sum = 0.0f;
                for (uint32_t component = 0; component < 4u; ++component)
                    sum += normalizeOperand(m_state.vf[vfS][component]);
                queueP(sum, 12u);
                return;
            }
            case 0x77: // ERSQRT
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                float result = value;
                if (result >= 0.0f)
                {
                    result = std::sqrt(result);
                    if (result != 0.0f)
                        result = 1.0f / result;
                }
                queueP(result, 18u);
                return;
            }
            case 0x78: // ESQRT
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                queueP(value >= 0.0f ? std::sqrt(value) : value, 12u);
                return;
            }
            case 0x79: // ESIN
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                queueP(vuEsin(value), 29u);
                return;
            }
            case 0x7A: // ERCPR
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                queueP(value != 0.0f ? 1.0f / value : value, 12u);
                return;
            }
            case 0x7B: // WAITP
                return;
            case 0x7C: // EATAN
            {
                const uint32_t component = (instr >> 21) & 3u;
                queueP(vuEatan(normalizeOperand(m_state.vf[vfS][component])), 54u);
                return;
            }
            case 0x7D: // EEXP
            {
                const uint32_t component = (instr >> 21) & 3u;
                queueP(vuEexp(normalizeOperand(m_state.vf[vfS][component])), 44u);
                return;
            }
            default:
                reportReservedInstruction(false, instr);
                return;
            }
        }
        default:
            reportReservedInstruction(false, instr);
            return;
        }
    }
    default:
        reportReservedInstruction(false, instr);
        break;
    }
}

struct Vu1RunArgs
{
    uint8_t *vuData = nullptr;
    uint32_t dataSize = 0;
    GS *gs = nullptr;
    PS2Memory *memory = nullptr;
    uint32_t codeSize = 0;
    uint64_t budgetEnd = 0;
};

struct VU1InterpreterAccess
{
    enum class StepResult : uint8_t
    {
        Continue,
        Ended,
        Budget,
        Fallback,
    };

    using CompiledProgram = StepResult (*)(VU1Interpreter &vu, const Vu1RunArgs &args);
    using CompiledBlock = StepResult (*)(VU1Interpreter &vu, const Vu1RunArgs &args);

    template <uint64_t ImageHash>
    static StepResult runImage(VU1Interpreter &vu, const Vu1RunArgs &args);

    template <uint64_t ImageHash, uint32_t Block>
    static StepResult runBlock(VU1Interpreter &vu, const Vu1RunArgs &args);

    static VU1Interpreter::DecodedInstructionPair decodePair(const VU1Interpreter &vu, const uint8_t *vuCode, uint32_t pc)
    {
        return vu.decodeInstructionPair(vuCode, pc);
    }

    static std::string pairInitializer(const VU1Interpreter &vu, const uint8_t *vuCode, uint32_t pc, bool &reserved);

    static uint32_t pc(const VU1Interpreter &vu)
    {
        return vu.m_state.pc;
    }

    static VU1_FORCE_INLINE StepResult generatedStep(VU1Interpreter &vu,
                                                     const VU1Interpreter::DecodedInstructionPair &decoded,
                                                     const Vu1RunArgs &args)
    {
        if (vu.m_cycle >= args.budgetEnd || vu.m_stopRequested)
            return StepResult::Budget;
        return stepPairOutOfLine(vu, decoded, args);
    }

    static StepResult stepPairOutOfLine(VU1Interpreter &vu,
                                        const VU1Interpreter::DecodedInstructionPair &decoded,
                                        const Vu1RunArgs &args);

    static StepResult runTable(VU1Interpreter &vu, const Vu1RunArgs &args, const CompiledBlock *table, uint32_t entries)
    {
        for (;;)
        {
            const uint32_t pc = vu.m_state.pc;
            const uint32_t index = pc / 8u;
            if ((pc & 7u) != 0u || index >= entries || table[index] == nullptr)
                return StepResult::Fallback;
            const StepResult result = table[index](vu, args);
            if (result != StepResult::Continue)
                return result;
        }
    }

    static VU1_FORCE_INLINE StepResult stepPair(VU1Interpreter &vu,
                                                const VU1Interpreter::DecodedInstructionPair &decoded,
                                                uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory,
                                                uint32_t codeSize, uint64_t budgetEnd)
    {
        bool ended = false;
            uint64_t readyCycle = vu.calculatePairReadyCycle(decoded);
            while (readyCycle > vu.m_cycle)
            {
                if (readyCycle >= budgetEnd)
                {
                    vu.advanceTo(budgetEnd);
                    break;
                }
                vu.advanceTo(readyCycle);
                readyCycle = vu.calculatePairReadyCycle(decoded);
            }
            if (vu.m_cycle >= budgetEnd)
                return StepResult::Budget;

            uint8_t writtenVi = 0u;
            int32_t oldVi = 0;
            if (const uint32_t viWrites = decoded.lowerUsage.viWrite & 0xFFFEu; viWrites != 0u)
            {
                writtenVi = static_cast<uint8_t>(std::countr_zero(viWrites));
                oldVi = vu.m_state.vi[writtenVi];
            }

            const VU1Interpreter::VfAccess upperWrite = decoded.upperUsage.vfWrite;
            const VU1Interpreter::VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
            const bool hasUpperWrite = upperWrite.reg != 0u;
            const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
            const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
            float oldUpperVf[4]{};
            float newUpperVf[4]{};
            float oldLowerVf[4]{};
            float newLowerVf[4]{};
            float oldAcc[4]{};
            float newAcc[4]{};
            if (hasUpperWrite)
                std::memcpy(oldUpperVf, vu.m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
            if (hasDistinctLowerWrite)
                std::memcpy(oldLowerVf, vu.m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
            if (decoded.upperUsage.accWrite != 0u)
                std::memcpy(oldAcc, vu.m_state.acc, sizeof(oldAcc));

            if (decoded.iBit)
            {
                vu.execUpper(decoded.upper);
                float immediate = 0.0f;
                std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
                vu.m_state.i = vu.normalizeOperand(immediate);
            }
            else if (decoded.upperVfShadowReg != 0u)
            {
                float oldVf[4]{};
                float upperVf[4]{};
                std::memcpy(oldVf,
                            vu.m_state.vf[decoded.upperVfShadowReg],
                            sizeof(oldVf));
                vu.execUpper(decoded.upper);
                std::memcpy(upperVf,
                            vu.m_state.vf[decoded.upperVfShadowReg],
                            sizeof(upperVf));
                std::memcpy(vu.m_state.vf[decoded.upperVfShadowReg],
                            oldVf,
                            sizeof(oldVf));
                vu.execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
                std::memcpy(vu.m_state.vf[decoded.upperVfShadowReg],
                            upperVf,
                            sizeof(upperVf));
            }
            else
            {
                vu.execUpper(decoded.upper);
                vu.execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            }

            vu.m_viBranchBackupValid = false;

            if (hasUpperWrite)
            {
                std::memcpy(newUpperVf, vu.m_state.vf[upperWrite.reg], sizeof(newUpperVf));
                std::memcpy(vu.m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
                const uint32_t latency =
                    decoded.upperUsage.vfLatency != 0u
                        ? decoded.upperUsage.vfLatency
                        : decoded.upperUsage.latency;
                vu.queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
            }
            if (hasDistinctLowerWrite)
            {
                std::memcpy(newLowerVf, vu.m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
                std::memcpy(vu.m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
                const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                             ? decoded.lowerUsage.vfLatency
                                             : decoded.lowerUsage.latency;
                vu.queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
            }
            if (decoded.upperUsage.accWrite != 0u)
            {
                std::memcpy(newAcc, vu.m_state.acc, sizeof(newAcc));
                std::memcpy(vu.m_state.acc, oldAcc, sizeof(oldAcc));
                // ACC is forwarded to the next upper instruction. Its arithmetic
                // flags still use the normal four-cycle FMAC timeline.
                vu.queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                              VU1Interpreter::kAccForwardLatency);
            }
            if (writtenVi != 0u)
            {
                const int32_t newVi = vu.m_state.vi[writtenVi];
                vu.m_state.vi[writtenVi] = oldVi;
                const uint32_t latency =
                    decoded.lowerUsage.viLatency != 0u
                        ? decoded.lowerUsage.viLatency
                        : decoded.lowerUsage.latency;
                vu.queueViWrite(writtenVi, newVi, latency);
            }

            vu.markPairWrites(decoded);
            if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
                vu.recordViWriteForBranch(writtenVi, oldVi);

            vu.m_state.vf[0][0] = 0.0f;
            vu.m_state.vf[0][1] = 0.0f;
            vu.m_state.vf[0][2] = 0.0f;
            vu.m_state.vf[0][3] = 1.0f;
            vu.m_state.vi[0] = 0;

            uint32_t nextPc = vu.m_state.pc + 8u;
            if (nextPc >= codeSize)
                nextPc = 0u;
            vu.m_state.pc = nextPc;

            if (vu.m_state.branchPending)
            {
                if (vu.m_state.branchDelay == 0u)
                {
                    vu.m_state.pc = vu.m_state.branchTarget & vu.microAddressMask();
                    vu.m_state.branchPending = false;
                }
                else
                {
                    --vu.m_state.branchDelay;
                }
            }

            const bool dHalt = decoded.dBit && vu.m_state.dBitEnabled;
            const bool tHalt = decoded.tBit && vu.m_state.tBitEnabled;
            const bool haltBit = dHalt || tHalt;
            const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == VU1Interpreter::PipelineBranch;

            if (vu.m_state.haltAfterDelaySlot)
            {
                vu.m_state.stoppedByD = vu.m_pendingHaltD;
                vu.m_state.stoppedByT = vu.m_pendingHaltT;
                ended = true;
            }
            else if (vu.m_state.ebit)
                ended = true;
            else if (haltBit && !haltBranch)
            {
                vu.m_state.stoppedByD = dHalt;
                vu.m_state.stoppedByT = tHalt;
                ended = true;
            }
            else if (decoded.eBit)
                vu.m_state.ebit = true;
            else if (haltBranch)
            {
                vu.m_state.haltAfterDelaySlot = true;
                vu.m_pendingHaltD = dHalt;
                vu.m_pendingHaltT = tHalt;
            }

        vu.advanceOneCycle();
        return ended ? StepResult::Ended : StepResult::Continue;
    }
};

void registerVu1CompiledProgram(uint64_t imageHash, VU1InterpreterAccess::CompiledProgram program);
