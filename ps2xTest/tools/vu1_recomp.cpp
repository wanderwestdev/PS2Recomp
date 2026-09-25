#include "ps2_vu1_impl.inl"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    using A = VU1InterpreterAccess;

    constexpr uint32_t kCodeSize = 16u * 1024u;
    constexpr uint32_t kDefaultBlockPairs = 128u;

    uint64_t fnv1a64(const uint8_t *data, size_t size)
    {
        uint64_t hash = 0xCBF29CE484222325ull;
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= data[i];
            hash *= 0x100000001B3ull;
        }
        return hash;
    }

    std::string hex64(uint64_t value)
    {
        char text[32];
        std::snprintf(text, sizeof(text), "0x%016" PRIX64 "ull", value);
        return text;
    }

    std::string hx(uint32_t value)
    {
        char text[16];
        std::snprintf(text, sizeof(text), "0x%Xu", value);
        return text;
    }

    std::string hexPcId(uint32_t index)
    {
        char text[16];
        std::snprintf(text, sizeof(text), "%04X", index * 8u);
        return text;
    }

    std::string label(uint32_t pc)
    {
        char text[16];
        std::snprintf(text, sizeof(text), "pc_%04X", pc);
        return text;
    }

    std::string num(int64_t value)
    {
        return std::to_string(value);
    }

    uint32_t laneBit(uint32_t component)
    {
        return 1u << (3u - component);
    }

    uint32_t nextPc(uint32_t pc)
    {
        return pc + 8u >= kCodeSize ? 0u : pc + 8u;
    }

    uint8_t fDest(uint32_t i) { return static_cast<uint8_t>((i >> 21) & 0xFu); }
    uint8_t fFt(uint32_t i) { return static_cast<uint8_t>((i >> 16) & 0x1Fu); }
    uint8_t fFs(uint32_t i) { return static_cast<uint8_t>((i >> 11) & 0x1Fu); }
    uint8_t fFd(uint32_t i) { return static_cast<uint8_t>((i >> 6) & 0x1Fu); }
    uint8_t fVit(uint32_t i) { return static_cast<uint8_t>((i >> 16) & 0xFu); }
    uint8_t fVis(uint32_t i) { return static_cast<uint8_t>((i >> 11) & 0xFu); }
    uint8_t fVid(uint32_t i) { return static_cast<uint8_t>((i >> 6) & 0xFu); }
    int32_t fImm11(uint32_t i) { return static_cast<int32_t>(i << 21) >> 21; }

    std::string vf(uint32_t reg) { return "vu.m_state.vf[" + num(reg) + "]"; }
    std::string vi(uint32_t reg) { return "vu.m_state.vi[" + num(reg) + "]"; }

    struct UpperEmit
    {
        bool nop = true;
        bool generic = false;
        std::string value;
        std::string target;
        std::string statement;
        uint32_t dest = 0u;
        int writeReg = -1;
        bool readsQ = false;
        bool isFmac = false;
        bool normalize = false;
        uint32_t reg = 0u;
    };

    UpperEmit upperEmit(uint32_t upper)
    {
        UpperEmit e;
        const uint32_t op = upper & 0x3Fu;
        const uint32_t dest = fDest(upper);
        const uint32_t ft = fFt(upper);
        const uint32_t fs = fFs(upper);
        const uint32_t fd = fFd(upper);
        const std::string s = "nv(ctx, " + num(fs) + "u)";
        const auto bcExpr = [&](uint32_t bc) { return "nbc(ctx, " + num(ft) + "u, " + num(bc) + "u)"; };
        const std::string tVec = "nv(ctx, " + num(ft) + "u)";
        const std::string zero = "_mm_setzero_ps()";

        const auto fmac = [&](const char *kind, const std::string &t, bool usesAcc, bool toAcc, bool readsQ)
        {
            if (dest == 0u)
                return;
            e.nop = false;
            e.dest = dest;
            e.readsQ = readsQ;
            e.isFmac = true;
            e.value = std::string("fmac<FK::") + kind + ">(vu, ctx, cyc, " + hx(dest) + ", " + s + ", " + t + ", " +
                      (usesAcc ? "nacc(ctx)" : zero) + ", un)";
            if (toAcc)
            {
                e.reg = 32u;
                e.writeReg = -2;
            }
            else
            {
                e.reg = fd;
                e.writeReg = static_cast<int>(fd);
            }
        };
        const auto unary = [&](const std::string &value, uint32_t reg, bool normalize = false)
        {
            if (dest == 0u)
                return;
            e.nop = false;
            e.dest = dest;
            e.value = value;
            e.reg = reg;
            e.normalize = normalize;
            e.writeReg = static_cast<int>(reg);
        };

        if (op < 0x3Cu)
        {
            if (op <= 0x1Bu)
            {
                const uint32_t bc = op & 3u;
                switch (op >> 2)
                {
                case 0: fmac("Add", bcExpr(bc), false, false, false); break;
                case 1: fmac("Sub", bcExpr(bc), false, false, false); break;
                case 2: fmac("Madd", bcExpr(bc), true, false, false); break;
                case 3: fmac("Msub", bcExpr(bc), true, false, false); break;
                case 4: unary("_mm_max_ps(" + s + ", " + bcExpr(bc) + ")", fd); break;
                case 5: unary("_mm_min_ps(" + s + ", " + bcExpr(bc) + ")", fd); break;
                case 6: fmac("Mul", bcExpr(bc), false, false, false); break;
                }
                return e;
            }
            switch (op)
            {
            case 0x1C: fmac("Mul", "qN(vu)", false, false, true); break;
            case 0x1D: unary("_mm_max_ps(" + s + ", iN(vu))", fd); break;
            case 0x1E: fmac("Mul", "iN(vu)", false, false, false); break;
            case 0x1F: unary("_mm_min_ps(" + s + ", iN(vu))", fd); break;
            case 0x20: fmac("Add", "qN(vu)", false, false, true); break;
            case 0x21: fmac("Madd", "qN(vu)", true, false, true); break;
            case 0x22: fmac("Add", "iN(vu)", false, false, false); break;
            case 0x23: fmac("Madd", "iN(vu)", true, false, false); break;
            case 0x24: fmac("Sub", "qN(vu)", false, false, true); break;
            case 0x25: fmac("Msub", "qN(vu)", true, false, true); break;
            case 0x26: fmac("Sub", "iN(vu)", false, false, false); break;
            case 0x27: fmac("Msub", "iN(vu)", true, false, false); break;
            case 0x28: fmac("Add", tVec, false, false, false); break;
            case 0x29: fmac("Madd", tVec, true, false, false); break;
            case 0x2A: fmac("Mul", tVec, false, false, false); break;
            case 0x2B: unary("_mm_max_ps(" + s + ", " + tVec + ")", fd); break;
            case 0x2C: fmac("Sub", tVec, false, false, false); break;
            case 0x2D: fmac("Msub", tVec, true, false, false); break;
            case 0x2E: fmac("Opmsub", tVec, true, false, false); break;
            case 0x2F: unary("_mm_min_ps(" + s + ", " + tVec + ")", fd); break;
            default:
                e.nop = false;
                e.generic = true;
                break;
            }
            return e;
        }

        const uint32_t special = (upper & 3u) | ((upper >> 4) & 0x7Cu);
        if (special <= 0x0Fu || (special >= 0x18u && special <= 0x1Bu))
        {
            const uint32_t bc = special & 3u;
            switch (special >> 2)
            {
            case 0: fmac("Add", bcExpr(bc), false, true, false); break;
            case 1: fmac("Sub", bcExpr(bc), false, true, false); break;
            case 2: fmac("Madd", bcExpr(bc), true, true, false); break;
            case 3: fmac("Msub", bcExpr(bc), true, true, false); break;
            case 6: fmac("Mul", bcExpr(bc), false, true, false); break;
            }
            return e;
        }
        static constexpr uint32_t kShift[4] = {0u, 4u, 12u, 15u};
        switch (special)
        {
        case 0x10: case 0x11: case 0x12: case 0x13:
            unary("itof<" + num(kShift[special & 3u]) + "u>(_mm_loadu_ps(" + vf(fs) + "))", ft);
            break;
        case 0x14: case 0x15: case 0x16: case 0x17:
            unary("ftoi<" + num(kShift[special & 3u]) + "u>(" + s + ")", ft, true);
            break;
        case 0x1C: fmac("Mul", "qN(vu)", false, true, true); break;
        case 0x1D: unary("absN(" + s + ")", ft); break;
        case 0x1E: fmac("Mul", "iN(vu)", false, true, false); break;
        case 0x1F:
            e.nop = false;
            e.statement = "clipOp(vu, ctx, cyc, " + num(fs) + "u, " + num(ft) + "u);";
            break;
        case 0x20: fmac("Add", "qN(vu)", false, true, true); break;
        case 0x21: fmac("Madd", "qN(vu)", true, true, true); break;
        case 0x22: fmac("Add", "iN(vu)", false, true, false); break;
        case 0x23: fmac("Madd", "iN(vu)", true, true, false); break;
        case 0x24: fmac("Sub", "qN(vu)", false, true, true); break;
        case 0x25: fmac("Msub", "qN(vu)", true, true, true); break;
        case 0x26: fmac("Sub", "iN(vu)", false, true, false); break;
        case 0x27: fmac("Msub", "iN(vu)", true, true, false); break;
        case 0x28: fmac("Add", tVec, false, true, false); break;
        case 0x29: fmac("Madd", tVec, true, true, false); break;
        case 0x2A: fmac("Mul", tVec, false, true, false); break;
        case 0x2C: fmac("Sub", tVec, false, true, false); break;
        case 0x2D: fmac("Msub", tVec, true, true, false); break;
        case 0x2E: fmac("Opmula", tVec, false, true, false); break;
        case 0x2F:
        case 0x30:
            break;
        default:
            e.nop = false;
            e.generic = true;
            break;
        }
        return e;
    }

    struct LowerEmit
    {
        std::string code;
        bool writesVf0 = false;
        bool generic = false;
    };

    uint32_t firstComponent(uint32_t dest)
    {
        if (dest & 0x8u)
            return 0u;
        if (dest & 0x4u)
            return 1u;
        if (dest & 0x2u)
            return 2u;
        return 3u;
    }

    LowerEmit lowerEmit(uint32_t pc, uint32_t lower, uint32_t upper)
    {
        LowerEmit e;
        if (lower == 0u || lower == 0x8000033Cu)
            return e;
        const uint32_t opHi = (lower >> 25) & 0x7Fu;
        const uint32_t dest = fDest(lower);
        const uint32_t ft = fFt(lower);
        const uint32_t fs = fFs(lower);
        const uint32_t it = fVit(lower);
        const uint32_t is = fVis(lower);
        const uint32_t id = fVid(lower);
        const int32_t imm = fImm11(lower);
        const std::string dst = hx(dest);
        const auto generic = [&]()
        {
            e.generic = true;
            e.writesVf0 = true;
            e.code = "genericLower(vu, ctx, cyc, " + hx(lower) + ", " + hx(upper) + "); syncAllNorm(vu, ctx);";
        };
        const auto viSet = [&](uint32_t reg, const std::string &value)
        {
            if (reg != 0u)
                e.code += vi(reg) + " = static_cast<int16_t>(" + value + ");";
        };
        const auto loadVf = [&](uint32_t reg, const std::string &address)
        {
            if (dest == 0u)
                return;
            e.code += "{ const __m128 lv = ldq(mem, " + address + "); writeVf(vu, ctx, " + num(reg) + "u, lv, vnorm(lv), " + dst + "); }";
            e.writesVf0 = e.writesVf0 || reg == 0u;
        };
        const auto storeVf = [&](uint32_t reg, const std::string &address)
        {
            if (dest != 0u)
                e.code += "stq(vu, ctx, cyc, " + address + ", _mm_loadu_ps(" + vf(reg) + "), " + dst + ");";
        };
        const uint32_t imm12 = (((lower >> 21) & 1u) << 11) | (lower & 0x7FFu);
        const uint32_t imm24 = lower & 0xFFFFFFu;
        const std::string status = "(vu.m_state.status & 0xFFFu)";

        switch (opHi)
        {
        case 0x00:
            loadVf(ft, "qaddr(" + vi(is) + ", " + num(imm) + ")");
            return e;
        case 0x01:
            storeVf(fs, "qaddr(" + vi(it) + ", " + num(imm) + ")");
            return e;
        case 0x04:
            if (it != 0u)
                viSet(it, "ldw(mem, qaddr(" + vi(is) + ", " + num(imm) + ") + " + num(firstComponent(dest) * 4u) + "u) & 0xFFFFu");
            return e;
        case 0x05:
            if (dest != 0u)
                e.code = "stw(vu, ctx, cyc, qaddr(" + vi(is) + ", " + num(imm) + "), " + vi(it) + ", " + dst + ");";
            return e;
        case 0x08:
        case 0x09:
        {
            const int16_t imm15 = static_cast<int16_t>(static_cast<uint32_t>(static_cast<int16_t>(lower & 0x7FFu)) | ((lower >> 10) & 0x7800u));
            viSet(it, vi(is) + (opHi == 0x08u ? " + " : " - ") + num(imm15));
            return e;
        }
        case 0x10:
            e.code = "commitClip(vu, ctx, cyc); " + vi(1) + " = (vu.m_state.clip & 0xFFFFFFu) == " + hx(imm24) + " ? 1 : 0;";
            return e;
        case 0x11:
            e.code = "queueFcsetAt(vu, ctx, cyc, " + hx(imm24) + ");";
            return e;
        case 0x12:
            e.code = "commitClip(vu, ctx, cyc); " + vi(1) + " = (vu.m_state.clip & " + hx(imm24) + ") != 0u ? 1 : 0;";
            return e;
        case 0x13:
            e.code = "commitClip(vu, ctx, cyc); " + vi(1) + " = (vu.m_state.clip | " + hx(imm24) + ") == 0xFFFFFFu ? 1 : 0;";
            return e;
        case 0x14:
            if (it != 0u)
                e.code = "commitStatus(vu, ctx, cyc); " + vi(it) + " = " + status + " == " + hx(imm12) + " ? 1 : 0;";
            return e;
        case 0x15:
            e.code = "queueFssetAt(vu, ctx, cyc, " + hx(imm12) + ");";
            return e;
        case 0x16:
            if (it != 0u)
            {
                e.code = "commitStatus(vu, ctx, cyc); ";
                viSet(it, status + " & " + hx(imm12));
            }
            return e;
        case 0x17:
            if (it != 0u)
            {
                e.code = "commitStatus(vu, ctx, cyc); ";
                viSet(it, status + " | " + hx(imm12));
            }
            return e;
        case 0x18:
            if (it != 0u)
                e.code = "commitStatus(vu, ctx, cyc); " + vi(it) + " = (vu.m_state.mac & 0xFFFFu) == static_cast<uint32_t>(static_cast<uint16_t>(" + vi(is) + ")) ? 1 : 0;";
            return e;
        case 0x1A:
        case 0x1B:
            if (it != 0u)
            {
                e.code = "commitStatus(vu, ctx, cyc); ";
                viSet(it, std::string("vu.m_state.mac ") + (opHi == 0x1Au ? "&" : "|") + " static_cast<uint32_t>(static_cast<uint16_t>(" + vi(is) + "))");
            }
            return e;
        case 0x1C:
            if (it != 0u)
            {
                e.code = "commitClip(vu, ctx, cyc); ";
                viSet(it, "vu.m_state.clip & 0x0FFFu");
            }
            return e;
        case 0x20:
            return e;
        case 0x21:
            viSet(it, num((pc + 16u) / 8u));
            return e;
        case 0x24:
            e.code = "jt = (static_cast<uint32_t>(static_cast<uint16_t>(vu.readBranchVi(" + num(is) + "u))) * 8u) & 0x3FFFu;";
            return e;
        case 0x25:
            e.code = "jt = (static_cast<uint32_t>(static_cast<uint16_t>(vu.readBranchVi(" + num(is) + "u))) * 8u) & 0x3FFFu;";
            viSet(it, num((pc + 16u) / 8u));
            return e;
        case 0x28:
        case 0x29:
            e.code = "tk = static_cast<int16_t>(vu.readBranchVi(" + num(is) + "u)) " + (opHi == 0x28u ? "==" : "!=") +
                     " static_cast<int16_t>(vu.readBranchVi(" + num(it) + "u));";
            return e;
        case 0x2C:
        case 0x2D:
        case 0x2E:
        case 0x2F:
        {
            static const char *kCompare[4] = {"< 0", "> 0", "<= 0", ">= 0"};
            e.code = "tk = static_cast<int16_t>(vu.readBranchVi(" + num(is) + "u)) " + kCompare[opHi - 0x2Cu] + ";";
            return e;
        }
        case 0x40:
            break;
        default:
            generic();
            return e;
        }

        const uint32_t funct = lower & 0x3Fu;
        switch (funct)
        {
        case 0x30:
            viSet(id, vi(is) + " + " + vi(it));
            return e;
        case 0x31:
            viSet(id, vi(is) + " - " + vi(it));
            return e;
        case 0x32:
        {
            const int32_t imm5 = static_cast<int32_t>(((lower >> 6) & 0x1Fu) << 27) >> 27;
            viSet(it, vi(is) + " + " + num(imm5));
            return e;
        }
        case 0x34:
            viSet(id, vi(is) + " & " + vi(it));
            return e;
        case 0x35:
            viSet(id, vi(is) + " | " + vi(it));
            return e;
        default:
            break;
        }
        if (funct < 0x3Cu)
        {
            generic();
            return e;
        }

        const uint32_t special = (lower & 3u) | ((lower >> 4) & 0x7Cu);
        const uint32_t fsf = (lower >> 21) & 3u;
        const uint32_t ftf = (lower >> 23) & 3u;
        switch (special)
        {
        case 0x30:
            if (dest != 0u)
            {
                e.code = "writeVf(vu, ctx, " + num(ft) + "u, _mm_loadu_ps(" + vf(fs) + "), nv(ctx, " + num(fs) + "u), " + dst + ");";
                e.writesVf0 = ft == 0u;
            }
            return e;
        case 0x31:
            if (dest != 0u)
            {
                e.code = "{ const __m128 mr = _mm_loadu_ps(" + vf(fs) + "); const __m128 mn = nv(ctx, " + num(fs) + "u); writeVf(vu, ctx, " +
                         num(ft) + "u, _mm_shuffle_ps(mr, mr, _MM_SHUFFLE(0, 3, 2, 1)), _mm_shuffle_ps(mn, mn, _MM_SHUFFLE(0, 3, 2, 1)), " + dst + "); }";
                e.writesVf0 = ft == 0u;
            }
            return e;
        case 0x34:
            loadVf(ft, "qaddr(" + vi(is) + ", 0)");
            viSet(is, vi(is) + " + 1");
            return e;
        case 0x35:
            storeVf(fs, "qaddr(" + vi(it) + ", 0)");
            viSet(it, vi(it) + " + 1");
            return e;
        case 0x36:
            viSet(is, vi(is) + " - 1");
            loadVf(ft, "qaddr(" + vi(is) + ", 0)");
            return e;
        case 0x37:
            viSet(it, vi(it) + " - 1");
            storeVf(fs, "qaddr(" + vi(it) + ", 0)");
            return e;
        case 0x38:
            e.code = "divOp(vu, ctx, cyc, " + num(fs) + "u, " + num(fsf) + "u, " + num(ft) + "u, " + num(ftf) + "u);";
            return e;
        case 0x39:
            e.code = "sqrtOp(vu, ctx, cyc, " + num(ft) + "u, " + num(ftf) + "u);";
            return e;
        case 0x3A:
            e.code = "rsqrtOp(vu, ctx, cyc, " + num(fs) + "u, " + num(fsf) + "u, " + num(ft) + "u, " + num(ftf) + "u);";
            return e;
        case 0x3B:
            return e;
        case 0x3C:
            if (it != 0u)
                viSet(it, "std::bit_cast<uint32_t>(" + vf(fs) + "[" + num(fsf) + "]) & 0xFFFFu");
            return e;
        case 0x3D:
            if (dest != 0u)
            {
                e.code = "{ const __m128 mv = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int16_t>(" + vi(is) +
                         " & 0xFFFF))); writeVf(vu, ctx, " + num(ft) + "u, mv, vnorm(mv), " + dst + "); }";
                e.writesVf0 = ft == 0u;
            }
            return e;
        case 0x3E:
            if (it != 0u)
                viSet(it, "ldw(mem, qaddr(" + vi(is) + ", 0) + " + num(firstComponent(dest) * 4u) + "u) & 0xFFFFu");
            return e;
        case 0x3F:
            if (dest != 0u)
                e.code = "stw(vu, ctx, cyc, qaddr(" + vi(is) + ", 0), " + vi(it) + ", " + dst + ");";
            return e;
        case 0x64:
            if (dest != 0u)
            {
                e.code = "commitEfu(vu, cyc); { const __m128 pv = _mm_set1_ps(vu.m_state.p); writeVf(vu, ctx, " + num(ft) + "u, pv, vnorm(pv), " + dst + "); }";
                e.writesVf0 = ft == 0u;
            }
            return e;
        case 0x68:
            viSet(it, "vu.m_state.top & 0x3FFu");
            return e;
        case 0x69:
            viSet(it, "vu.m_state.itop & 0x3FFu");
            return e;
        case 0x6C:
            e.code = "xgkickStart(vu, ctx, cyc, " + vi(is) + ");";
            return e;
        case 0x7B:
            return e;
        default:
            generic();
            return e;
        }
    }

    struct Pair
    {
        uint32_t pc = 0;
        A::Decoded d{};
        bool reserved = false;
        bool branch = false;
        bool dynamic = false;
        bool conditional = false;
        uint32_t target = 0;
        bool fallback = false;
    };

    struct Known
    {
        uint32_t pos = 0;
        uint64_t bound = 0;
        std::string id;
        std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>> vf;
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> vi;
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> acc;
        bool div = false;
        uint32_t divPos = 0;
        uint32_t divLat = 0;
        std::map<uint32_t, std::vector<std::string>> checks;
        std::set<std::string> checked;
        bool fdivChecked = false;
        uint32_t fdivK = 0;
    };

    std::string segLabel(uint32_t pc)
    {
        char text[16];
        std::snprintf(text, sizeof(text), "seg_%04X", pc);
        return text;
    }

    struct Emitter
    {
        std::vector<Pair> pairs;
        uint32_t first = 0;
        uint32_t last = 0;
        bool usesTk = false;
        bool usesJt = false;
        std::set<std::string> exits;
        std::vector<bool> isTarget;
        std::set<uint32_t> segs;

        bool inBlock(uint32_t pc) const
        {
            const uint32_t index = pc / 8u;
            return index >= first && index < last;
        }

        std::string jumpTo(uint32_t pc)
        {
            if (inBlock(pc))
                return "goto " + (segs.count(pc / 8u) ? segLabel(pc) : label(pc)) + ";";
            exits.insert("exit_continue");
            return "{ epc = " + hx(pc) + "; goto exit_continue; }";
        }

        bool heavy(const Pair &p) const
        {
            if (p.fallback || p.reserved)
                return true;
            if (!p.d.iBit)
            {
                const unsigned pipe = A::pipeline(p.d.lowerUsage);
                if (pipe == A::kPipeXgkick || pipe == A::kPipeEfu || p.d.lowerUsage.waitP)
                    return true;
                if (lowerEmit(p.pc, p.d.lower, p.d.upper).generic)
                    return true;
            }
            return upperEmit(p.d.upper).generic;
        }

        bool breaker(uint32_t index) const
        {
            const Pair &p = pairs[index];
            if (heavy(p))
                return true;
            if (p.branch || p.d.eBit)
                return heavy(pairs[nextPc(p.pc) / 8u]);
            return false;
        }

        bool isStart(uint32_t index) const
        {
            if (index == first || isTarget[index])
                return true;
            if (index >= 1u && (breaker(index - 1u) || pairs[index - 1u].branch || pairs[index - 1u].d.eBit))
                return true;
            return index >= 2u && pairs[index - 2u].branch;
        }

        uint32_t segmentPositions(uint32_t start) const
        {
            if (breaker(start))
                return 0u;
            uint32_t positions = 0u;
            for (uint32_t index = start;;)
            {
                if (index >= last || (index != start && isStart(index)) || breaker(index))
                    return positions;
                const Pair &p = pairs[index];
                if (p.branch || p.d.eBit)
                    return positions + 2u;
                ++positions;
                const uint32_t next = nextPc(p.pc) / 8u;
                if (next != index + 1u)
                    return positions;
                index = next;
            }
        }

        void planSegments()
        {
            segs.clear();
            for (uint32_t index = first; index < last; ++index)
                if (isStart(index) && segmentPositions(index) >= 3u)
                    segs.insert(index);
        }

        std::string knownStall(const Pair &p, Known &k)
        {
            const auto &d = p.d;
            std::vector<std::string> terms;
            uint32_t extra = 0u;
            const auto internal = [&](uint32_t writerPos, uint32_t latency)
            {
                const uint32_t distance = k.pos - writerPos;
                if (distance >= latency)
                    return;
                terms.push_back("s" + k.id + "_" + num(writerPos) + " + " + num(latency) + "u");
                extra = std::max(extra, latency - distance);
            };
            const auto external = [&](const std::string &expr)
            {
                if (k.checked.insert(expr).second)
                    k.checks[k.pos].push_back(expr);
            };
            std::set<std::pair<uint32_t, uint32_t>> lanes;
            for (const auto *usage : {&d.upperUsage, &d.lowerUsage})
                for (uint32_t index = 0; index < usage->vfReadCount; ++index)
                {
                    const auto access = usage->vfRead[index];
                    if (access.reg == 0u)
                        continue;
                    for (uint32_t component = 0; component < 4u; ++component)
                        if (access.lanes & laneBit(component))
                            lanes.insert({access.reg, component});
                }
            for (const auto &lane : lanes)
            {
                const auto it = k.vf.find(lane);
                if (it != k.vf.end())
                    internal(it->second.first, it->second.second);
                else
                    external("vu.m_vfReady[" + num(lane.first) + "][" + num(lane.second) + "]");
            }
            const uint32_t viReads = (d.upperUsage.viRead | d.lowerUsage.viRead) & 0xFFFEu;
            for (uint32_t reg = 1; reg < 16u; ++reg)
            {
                if (!(viReads & (1u << reg)))
                    continue;
                const auto it = k.vi.find(reg);
                if (it != k.vi.end())
                    internal(it->second.first, it->second.second);
                else
                    external("vu.m_viReady[" + num(reg) + "]");
            }
            const uint32_t accReads = d.upperUsage.accRead | d.lowerUsage.accRead;
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if (!(accReads & laneBit(component)))
                    continue;
                const auto it = k.acc.find(component);
                if (it != k.acc.end())
                    internal(it->second.first, it->second.second);
                else
                    external("vu.m_accReady[" + num(component) + "]");
            }
            if (!d.iBit && (A::pipeline(d.lowerUsage) == A::kPipeFdiv || d.lowerUsage.waitQ))
            {
                if (k.div)
                    internal(k.divPos, k.divLat);
                else if (!k.fdivChecked)
                {
                    k.fdivChecked = true;
                    k.fdivK = k.pos;
                }
            }
            k.bound += 1u + extra;
            if (terms.empty())
                return "";
            std::string out = "        { uint64_t r = cyc;";
            for (const std::string &term : terms)
                out += " r = rmax(r, " + term + ");";
            out += " cyc = r; }\n";
            return out;
        }

        void recordWrites(const Pair &p, Known &k)
        {
            const auto &d = p.d;
            const auto lw = d.lowerUsage.vfWrite;
            if (lw.reg != 0u && d.suppressedLowerVf != lw.reg)
                for (uint32_t component = 0; component < 4u; ++component)
                    if (lw.lanes & laneBit(component))
                        k.vf[{lw.reg, component}] = {k.pos, d.lowerUsage.vfLatency != 0u ? d.lowerUsage.vfLatency : d.lowerUsage.latency};
            const auto uw = d.upperUsage.vfWrite;
            if (uw.reg != 0u)
                for (uint32_t component = 0; component < 4u; ++component)
                    if (uw.lanes & laneBit(component))
                        k.vf[{uw.reg, component}] = {k.pos, d.upperUsage.vfLatency != 0u ? d.upperUsage.vfLatency : d.upperUsage.latency};
            const uint32_t viWrites = d.lowerUsage.viWrite & 0xFFFEu;
            for (uint32_t reg = 1; reg < 16u; ++reg)
                if (viWrites & (1u << reg))
                    k.vi[reg] = {k.pos, d.lowerUsage.viLatency != 0u ? d.lowerUsage.viLatency : d.lowerUsage.latency};
            for (uint32_t component = 0; component < 4u; ++component)
                if (d.upperUsage.accWrite & laneBit(component))
                    k.acc[component] = {k.pos, 1u};
            if (!d.iBit && A::pipeline(d.lowerUsage) == A::kPipeFdiv && !d.lowerUsage.waitQ && d.lowerUsage.latency != 0u)
            {
                k.div = true;
                k.divPos = k.pos;
                k.divLat = d.lowerUsage.latency;
            }
            ++k.pos;
        }

        std::string stall(const Pair &p, const std::string &stub)
        {
            const auto &d = p.d;
            std::vector<std::pair<uint32_t, uint32_t>> vfLanes;
            const auto addReads = [&](const auto &usage)
            {
                for (uint32_t index = 0; index < usage.vfReadCount; ++index)
                {
                    const auto access = usage.vfRead[index];
                    if (access.reg == 0u)
                        continue;
                    for (uint32_t component = 0; component < 4u; ++component)
                    {
                        const auto item = std::make_pair(static_cast<uint32_t>(access.reg), component);
                        if ((access.lanes & laneBit(component)) != 0u &&
                            std::find(vfLanes.begin(), vfLanes.end(), item) == vfLanes.end())
                            vfLanes.push_back(item);
                    }
                }
            };
            addReads(d.upperUsage);
            addReads(d.lowerUsage);
            const uint32_t viReads = (d.upperUsage.viRead | d.lowerUsage.viRead) & 0xFFFEu;
            const uint32_t accReads = d.upperUsage.accRead | d.lowerUsage.accRead;
            const unsigned lowerPipe = A::pipeline(d.lowerUsage);
            const bool fdiv = lowerPipe == A::kPipeFdiv || d.lowerUsage.waitQ;
            const bool efu = lowerPipe == A::kPipeEfu;
            const bool waitP = d.lowerUsage.waitP;
            const bool xgkick = lowerPipe == A::kPipeXgkick;
            if (vfLanes.empty() && viReads == 0u && accReads == 0u && !fdiv && !efu && !waitP && !xgkick)
                return "";
            std::string out = "        { uint64_t r = cyc;";
            for (const auto &[reg, component] : vfLanes)
                out += " r = rmax(r, vu.m_vfReady[" + num(reg) + "][" + num(component) + "]);";
            for (uint32_t reg = 1; reg < 16u; ++reg)
                if (viReads & (1u << reg))
                    out += " r = rmax(r, vu.m_viReady[" + num(reg) + "]);";
            for (uint32_t component = 0; component < 4u; ++component)
                if (accReads & laneBit(component))
                    out += " r = rmax(r, vu.m_accReady[" + num(component) + "]);";
            if (fdiv)
                out += " if (vu.m_fdiv.valid) r = rmax(r, vu.m_fdiv.readyCycle);";
            if (efu)
                out += " r = rmax(r, vu.m_efuResourceReady);";
            if (waitP)
                out += " if (vu.m_efu[0].valid) r = rmax(r, vu.m_efu[0].readyCycle); if (vu.m_efu[1].valid) r = rmax(r, vu.m_efu[1].readyCycle);";
            if (xgkick)
                out += " if (!xgkickStall(vu, ctx, cyc, r, budget)) " + stub + " }\n";
            else
                out += " if (r > cyc) { if (r >= budget) { cyc = budget; " + stub + " } cyc = r; } }\n";
            return out;
        }

        std::string body(const Pair &p, const std::string &stub, Known *known = nullptr)
        {
            const auto &d = p.d;
            std::string out;
            if (known)
            {
                out += knownStall(p, *known);
                out += "        s" + known->id + "_" + num(known->pos) + " = cyc;\n";
            }
            else
            {
                out += "        if (cyc >= budget) " + stub + "\n";
                out += stall(p, stub);
            }
            out += "        {\n";
            const uint32_t viWrites = d.lowerUsage.viWrite & 0xFFFEu;
            const uint32_t writtenVi = viWrites != 0u ? static_cast<uint32_t>(std::countr_zero(viWrites)) : 0u;
            const bool recordBackup = writtenVi != 0u && d.lowerUsage.delaysNextBranchRead;
            if (recordBackup)
                out += "            const int32_t oldVi = " + vi(writtenVi) + ";\n";
            const UpperEmit up = upperEmit(d.upper);
            const LowerEmit low = d.iBit ? LowerEmit{} : lowerEmit(p.pc, d.lower, d.upper);
            if (up.readsQ || up.generic)
                out += "            syncQ(vu, ctx, cyc);\n";
            std::string lowerCode = low.code.empty() ? "" : "            " + low.code + "\n";
            const auto upperStore = [&]()
            {
                if (up.nop)
                    return std::string();
                if (up.generic)
                    return "            genericUpper(vu, ctx, cyc, " + hx(d.upper) + "); syncAllNorm(vu, ctx);\n";
                if (!up.statement.empty())
                    return "            " + up.statement + "\n";
                if (up.isFmac)
                    return "            { __m128 un; const __m128 uv = " + up.value + "; writeVf(vu, ctx, " + num(up.reg) + "u, uv, un, " + hx(up.dest) + "); }\n";
                return "            { const __m128 uv = " + up.value + "; writeVf(vu, ctx, " + num(up.reg) + "u, uv, " +
                       (up.normalize ? std::string("vnorm(uv)") : std::string("uv")) + ", " + hx(up.dest) + "); }\n";
            };
            if (d.iBit)
            {
                uint32_t bits = d.lower;
                const uint32_t exponent = (bits >> 23) & 0xFFu;
                if (exponent == 0u)
                    bits &= 0x80000000u;
                else if (exponent == 0xFFu)
                    bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
                out += upperStore();
                out += "            vu.m_state.i = std::bit_cast<float>(" + hx(bits) + ");\n";
            }
            else if (d.upperVfShadowReg != 0u)
            {
                const std::string reg = vf(d.upperVfShadowReg);
                if (up.generic)
                {
                    out += "            float shadowOld[4]; std::memcpy(shadowOld, " + reg + ", 16u);\n";
                    out += "            genericUpper(vu, ctx, cyc, " + hx(d.upper) + ");\n";
                    out += "            float shadowNew[4]; std::memcpy(shadowNew, " + reg + ", 16u); std::memcpy(" + reg + ", shadowOld, 16u);\n";
                    out += lowerCode;
                    out += "            std::memcpy(" + reg + ", shadowNew, 16u); syncAllNorm(vu, ctx);\n";
                }
                else
                {
                    out += "            const __m128 shadowOld = _mm_loadu_ps(" + reg + ");\n";
                    out += "            const __m128 shadowOldN = nv(ctx, " + num(d.upperVfShadowReg) + "u);\n";
                    if (up.isFmac)
                        out += "            __m128 un; const __m128 uv = " + up.value + ";\n";
                    else
                        out += "            const __m128 uv = " + up.value + "; const __m128 un = " +
                               (up.normalize ? std::string("vnorm(uv)") : std::string("uv")) + ";\n";
                    out += lowerCode;
                    out += "            writeVfFull(vu, ctx, " + num(d.upperVfShadowReg) + "u, blendDest(shadowOld, uv, " + hx(up.dest) +
                           "), blendDest(shadowOldN, un, " + hx(up.dest) + "));\n";
                }
            }
            else
            {
                out += upperStore();
                out += lowerCode;
            }
            out += "            vu.m_viBranchBackupValid = false;\n";
            const auto markVf = [&](uint32_t reg, uint32_t lanes, uint32_t latency)
            {
                for (uint32_t component = 0; component < 4u; ++component)
                    if (lanes & laneBit(component))
                        out += "            vu.m_vfReady[" + num(reg) + "][" + num(component) + "] = cyc + " + num(latency) + "u;\n";
            };
            const auto lw = d.lowerUsage.vfWrite;
            if (lw.reg != 0u && d.suppressedLowerVf != lw.reg)
                markVf(lw.reg, lw.lanes, d.lowerUsage.vfLatency != 0u ? d.lowerUsage.vfLatency : d.lowerUsage.latency);
            const auto uw = d.upperUsage.vfWrite;
            if (uw.reg != 0u)
                markVf(uw.reg, uw.lanes, d.upperUsage.vfLatency != 0u ? d.upperUsage.vfLatency : d.upperUsage.latency);
            for (uint32_t reg = 1; reg < 16u; ++reg)
                if (viWrites & (1u << reg))
                    out += "            vu.m_viReady[" + num(reg) + "] = cyc + " +
                           num(d.lowerUsage.viLatency != 0u ? d.lowerUsage.viLatency : d.lowerUsage.latency) + "u;\n";
            for (uint32_t component = 0; component < 4u; ++component)
                if (d.upperUsage.accWrite & laneBit(component))
                    out += "            vu.m_accReady[" + num(component) + "] = cyc + 1u;\n";
            if (recordBackup)
                out += "            vu.m_viBranchBackupValue = oldVi; vu.m_viBranchBackupReg = " + num(writtenVi) +
                       "u; vu.m_viBranchBackupValid = true;\n";
            if (up.writeReg == 0 || up.generic || low.writesVf0)
                out += "            resetVf0(vu, ctx);\n";
            out += "        }\n";
            out += "        ++cyc;\n";
            if (known)
                recordWrites(p, *known);
            return out;
        }

        std::string resolution(const Pair &p)
        {
            const uint32_t next = nextPc(p.pc);
            const uint32_t after = nextPc(next);
            if (p.dynamic)
            {
                exits.insert("exit_continue");
                return "        { epc = jt; goto exit_continue; }\n";
            }
            if (p.conditional)
                return "        if (tk) " + jumpTo(p.target) + "\n        " + jumpTo(after) + "\n";
            return "        " + jumpTo(p.target) + "\n";
        }

        std::string emitSegment(uint32_t start)
        {
            Known k;
            k.id = hexPcId(start);
            std::string code;
            std::string tail;
            for (uint32_t index = start;;)
            {
                if (index >= last)
                {
                    tail = "        " + jumpTo(index * 8u) + "\n";
                    break;
                }
                if ((index != start && isStart(index)) || breaker(index))
                {
                    tail = "        " + jumpTo(index * 8u) + "\n";
                    if (breaker(index))
                        tail = "        goto " + label(index * 8u) + ";\n";
                    break;
                }
                const Pair &p = pairs[index];
                code += "        // " + label(p.pc) + "\n";
                code += body(p, "", &k);
                const uint32_t next = nextPc(p.pc);
                if (p.branch)
                {
                    if (p.conditional)
                        usesTk = true;
                    if (p.dynamic)
                        usesJt = true;
                    code += body(pairs[next / 8u], "", &k);
                    tail = resolution(p);
                    break;
                }
                if (p.d.eBit)
                {
                    code += body(pairs[next / 8u], "", &k);
                    exits.insert("exit_ended");
                    tail = "        { epc = " + hx(nextPc(next)) + "; goto exit_ended; }\n";
                    break;
                }
                if (next / 8u != index + 1u)
                {
                    tail = "        " + jumpTo(next) + "\n";
                    break;
                }
                index = next / 8u;
            }
            std::string out = segLabel(start * 8u) + ":\n";
            const std::string fallback = " goto " + label(start * 8u) + ";\n";
            out += "        if (cyc + " + num(k.bound) + "u > budget)" + fallback;
            for (const auto &[position, exprs] : k.checks)
            {
                std::string max = exprs[0];
                for (size_t i = 1; i < exprs.size(); ++i)
                    max = "rmax(" + max + ", " + exprs[i] + ")";
                out += "        if (" + max + " > cyc + " + num(position) + "u)" + fallback;
            }
            if (k.fdivChecked)
                out += "        if (vu.m_fdiv.valid && vu.m_fdiv.readyCycle > cyc + " + num(k.fdivK) + "u)" + fallback;
            out += "    {\n        uint64_t ";
            for (uint32_t i = 0; i < k.pos; ++i)
                out += (i ? ", s" : "s") + k.id + "_" + num(i) + " = 0u";
            out += ";\n";
            out += code;
            out += tail;
            out += "    }\n";
            return out;
        }

        std::string emitPair(uint32_t index)
        {
            const Pair &p = pairs[index];
            std::string out = label(p.pc) + ":\n";
            if (p.fallback)
            {
                exits.insert("exit_fallback");
                out += "        { epc = " + hx(p.pc) + "; goto exit_fallback; }\n";
                return out;
            }
            exits.insert("exit_budget");
            const std::string stub = "{ epc = " + hx(p.pc) + "; goto exit_budget; }";
            out += body(p, stub);
            const uint32_t next = nextPc(p.pc);
            if (p.branch)
            {
                const Pair &slot = pairs[next / 8u];
                std::string state = "{ vu.m_state.branchPending = true; vu.m_state.branchTarget = " +
                                    (p.dynamic ? std::string("jt") : hx(p.target)) + "; vu.m_state.branchDelay = 0u; }";
                if (p.conditional)
                {
                    usesTk = true;
                    state = "if (tk) " + state;
                }
                if (p.dynamic)
                    usesJt = true;
                const std::string slotStub = "{ epc = " + hx(next) + "; " + state + " goto exit_budget; }";
                out += "        // delay slot\n";
                out += body(slot, slotStub);
                out += resolution(p);
                return out;
            }
            if (p.d.eBit)
            {
                const Pair &slot = pairs[next / 8u];
                const std::string slotStub = "{ epc = " + hx(next) + "; vu.m_state.ebit = true; goto exit_budget; }";
                out += "        // end delay slot\n";
                out += body(slot, slotStub);
                exits.insert("exit_ended");
                out += "        { epc = " + hx(nextPc(next)) + "; goto exit_ended; }\n";
                return out;
            }
            if (next / 8u != index + 1u || index + 1u >= last || segs.count(index + 1u))
                out += "        " + jumpTo(next) + "\n";
            return out;
        }
    };

    bool plainPair(const Pair &p)
    {
        return !p.reserved && !p.branch && !p.d.eBit && !p.d.dBit && !p.d.tBit;
    }

    bool writeImage(const std::filesystem::path &outDir, uint64_t hash, const std::vector<uint8_t> &code, uint32_t maxBlockPairs,
                    std::map<std::string, uint32_t> &stats)
    {
        VU1Interpreter vu(VU1Interpreter::Unit::VU1);
        uint32_t used = 0u;
        for (uint32_t pc = 0; pc + 8u <= code.size(); pc += 8u)
            if (std::any_of(code.begin() + pc, code.begin() + pc + 8u, [](uint8_t b) { return b != 0u; }))
                used = pc / 8u + 1u;
        const uint32_t total = kCodeSize / 8u;
        const uint32_t count = std::min(total, used + 2u);

        Emitter emitter;
        emitter.pairs.resize(total);
        for (uint32_t index = 0; index < total; ++index)
        {
            Pair &p = emitter.pairs[index];
            p.pc = index * 8u;
            p.d = A::decodePair(vu, code.data(), p.pc);
            p.reserved = p.d.upperUsage.reserved || p.d.lowerUsage.reserved;
            p.branch = !p.d.iBit && A::pipeline(p.d.lowerUsage) == A::kPipeBranch;
            if (p.branch)
            {
                const uint32_t opHi = (p.d.lower >> 25) & 0x7Fu;
                p.dynamic = opHi == 0x24u || opHi == 0x25u;
                p.conditional = opHi >= 0x28u;
                p.target = (p.pc + 8u + static_cast<uint32_t>(fImm11(p.d.lower) * 8)) & 0x3FFFu;
            }
        }
        for (uint32_t index = 0; index < count; ++index)
        {
            Pair &p = emitter.pairs[index];
            const Pair &next = emitter.pairs[nextPc(p.pc) / 8u];
            p.fallback = p.reserved || p.d.dBit || p.d.tBit;
            if ((p.branch || p.d.eBit) && !plainPair(next))
                p.fallback = true;
            if (p.branch && p.d.eBit)
                p.fallback = true;
            const uint32_t op = p.d.upper & 0x3Fu;
            const uint32_t special = (p.d.upper & 3u) | ((p.d.upper >> 4) & 0x7Cu);
            if ((op == 0x2Eu || (op >= 0x3Cu && special == 0x2Eu)) && (fDest(p.d.upper) & 0xEu) != 0xEu && fDest(p.d.upper) != 0u)
                p.fallback = true;
            if (p.fallback)
                ++stats["fallback pairs"];
            if (!p.d.iBit && lowerEmit(p.pc, p.d.lower, p.d.upper).generic)
                ++stats["generic lower"];
            if (upperEmit(p.d.upper).generic)
                ++stats["generic upper"];
        }

        emitter.isTarget.assign(total + 1u, false);
        for (uint32_t index = 0; index < count; ++index)
        {
            const Pair &p = emitter.pairs[index];
            if (p.branch && !p.dynamic)
                emitter.isTarget[p.target / 8u] = true;
        }

        std::vector<bool> spanned(count + 1u, false);
        for (uint32_t index = 0; index < count; ++index)
        {
            const Pair &p = emitter.pairs[index];
            if (!p.branch || p.dynamic)
                continue;
            const uint32_t target = p.target / 8u;
            if (target <= index)
                for (uint32_t cut = target + 1u; cut <= std::min(count, index + 1u); ++cut)
                    spanned[cut] = true;
        }
        std::vector<std::pair<uint32_t, uint32_t>> blocks;
        for (uint32_t start = 0; start < count;)
        {
            uint32_t end = std::min(count, start + maxBlockPairs);
            uint32_t cut = end;
            while (cut < count && cut > start + maxBlockPairs / 4u && spanned[cut])
                --cut;
            if (cut < count && spanned[cut])
                cut = end;
            blocks.emplace_back(start, cut);
            start = cut;
        }

        std::ostringstream out;
        out << "#include \"ps2_vu1_impl.inl\"\n\n";
        std::vector<uint32_t> blockOf(count, 0u);
        for (uint32_t block = 0; block < blocks.size(); ++block)
        {
            const auto [start, end] = blocks[block];
            emitter.first = start;
            emitter.last = end;
            emitter.usesTk = false;
            emitter.usesJt = false;
            emitter.exits.clear();
            emitter.planSegments();
            std::string code;
            for (uint32_t index = start; index < end; ++index)
            {
                blockOf[index] = block;
                code += emitter.emitPair(index);
            }
            for (const uint32_t segment : emitter.segs)
                code += emitter.emitSegment(segment);
            stats["known segments"] += static_cast<uint32_t>(emitter.segs.size());
            out << "template <>\nVU1InterpreterAccess::StepResult VU1InterpreterAccess::fastBlock<" << hex64(hash) << ", "
                << block << "u>(VU1Interpreter &vu, Vu1FastCtx &ctx)\n{\n";
            out << "    uint64_t cyc = ctx.cycle;\n    const uint64_t budget = ctx.budget;\n    const uint8_t *const mem = ctx.mem;\n";
            out << "    uint32_t epc = 0u;\n";
            if (emitter.usesTk)
                out << "    bool tk = false;\n";
            if (emitter.usesJt)
                out << "    uint32_t jt = 0u;\n";
            out << "    (void)mem;\n";
            out << "    switch (vu.m_state.pc)\n    {\n";
            for (uint32_t index = start; index < end; ++index)
                out << "    case " << hx(index * 8u) << ": goto " << (emitter.segs.count(index) ? segLabel(index * 8u) : label(index * 8u)) << ";\n";
            out << "    default: return StepResult::Fallback;\n    }\n";
            out << code;
            const std::pair<const char *, const char *> exitKinds[] = {
                {"exit_budget", "Budget"}, {"exit_continue", "Continue"}, {"exit_ended", "Ended"}, {"exit_fallback", "Fallback"}};
            for (const auto &[name, result] : exitKinds)
                if (emitter.exits.count(name))
                    out << name << ":\n    ctx.cycle = cyc;\n    vu.m_state.pc = epc;\n    return StepResult::" << result << ";\n";
            out << "}\n\n";
        }

        out << "template <>\nVU1InterpreterAccess::StepResult VU1InterpreterAccess::runImage<" << hex64(hash)
            << ">(VU1Interpreter &vu, const Vu1RunArgs &args)\n{\n    static constexpr FastBlock table[] = {\n";
        for (uint32_t index = 0; index < count; ++index)
            out << "        &fastBlock<" << hex64(hash) << ", " << blockOf[index] << "u>,\n";
        out << "    };\n    return runFast(vu, args, table, " << count << "u);\n}\n";

        char name[64];
        std::snprintf(name, sizeof(name), "vu1_%016" PRIx64 ".cpp", hash);
        std::ofstream file(outDir / name, std::ios::binary);
        file << out.str();
        std::cout << name << ": " << count << " pairs, " << blocks.size() << " blocks\n";
        return static_cast<bool>(file);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: vu1_recomp <capture-dir> <out-dir> [max-pairs-per-block]\n";
        return 2;
    }
    const std::filesystem::path captureDir = argv[1];
    const std::filesystem::path outDir = argv[2];
    const uint32_t blockPairs = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : kDefaultBlockPairs;
    std::filesystem::create_directories(outDir);

    std::vector<std::filesystem::path> images;
    for (const auto &entry : std::filesystem::directory_iterator(captureDir))
    {
        const std::string name = entry.path().filename().string();
        if (name.rfind("vu1_", 0) == 0 && entry.path().extension() == ".bin")
            images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());

    std::vector<uint64_t> hashes;
    std::map<std::string, uint32_t> stats;
    for (const auto &path : images)
    {
        std::ifstream file(path, std::ios::binary);
        std::vector<uint8_t> code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (code.size() != kCodeSize)
        {
            std::cerr << path << ": expected " << kCodeSize << " bytes\n";
            return 1;
        }
        const uint64_t hash = fnv1a64(code.data(), code.size());
        if (!writeImage(outDir, hash, code, blockPairs, stats))
            return 1;
        hashes.push_back(hash);
    }
    for (const auto &[name, value] : stats)
        std::cout << name << ": " << value << "\n";

    std::ostringstream reg;
    reg << "#include \"ps2_vu1_impl.inl\"\n\n";
    for (const uint64_t hash : hashes)
        reg << "template <>\nVU1InterpreterAccess::StepResult VU1InterpreterAccess::runImage<" << hex64(hash)
            << ">(VU1Interpreter &, const Vu1RunArgs &);\n";
    reg << "\nvoid registerGeneratedVu1Programs()\n{\n";
    for (const uint64_t hash : hashes)
        reg << "    registerVu1CompiledProgram(" << hex64(hash) << ", &VU1InterpreterAccess::runImage<" << hex64(hash) << ">);\n";
    reg << "}\n";
    std::ofstream file(outDir / "vu1_programs.cpp", std::ios::binary);
    file << reg.str();
    return file ? 0 : 1;
}
