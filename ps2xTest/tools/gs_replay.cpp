#include "gs_golden.h"
#include "gs_reference_backend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_threaded_backend.h"
#include "gs_sampler.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <xmmintrin.h>

namespace GSRaster
{
    std::array<uint64_t, 8> &FallbackCounters();
}

namespace
{
    std::unique_ptr<GSRasterBackend> makeBackend(const std::string &name)
    {
        if (name == "ref")
            return std::make_unique<GSReferenceBackend>();
        if (name == "cpu")
            return std::make_unique<GSCpuBackend>();
        if (name == "thread")
            return std::make_unique<GSThreadedBackend>(std::make_unique<GSCpuBackend>());
        return nullptr;
    }

    void writeDraw(std::FILE *file, const GSPrimitiveBatch &batch)
    {
        const GSDrawState &st = batch.state;
        const GSContext &c = st.context;
        const GSPrimReg &p = st.prim;
        std::fprintf(file,
                     "D prim=%u iip=%d tme=%d fge=%d abe=%d aa1=%d fst=%d ctxt=%d fix=%d"
                     " frame=%x/%u/%02x/%08x zbuf=%x/%02x/%d"
                     " tex0=%x/%u/%02x/%ux%u/tcc%u/tfx%u/cbp%x/cpsm%02x/csm%u/csa%u/cld%u"
                     " tex1=%llx clamp=%llx alpha=%llx test=%llx fba=%llx scissor=%u-%u,%u-%u ofs=%u,%u"
                     " texa=%u/%d/%u pabe=%d dthe=%llx colclamp=%llx",
                     static_cast<unsigned>(p.type), p.iip, p.tme, p.fge, p.abe, p.aa1, p.fst, p.ctxt, p.fix,
                     c.frame.fbp, c.frame.fbw, c.frame.psm, c.frame.fbmsk, c.zbuf.zbp, c.zbuf.psm, c.zbuf.zmask,
                     c.tex0.tbp0, c.tex0.tbw, c.tex0.psm, 1u << c.tex0.tw, 1u << c.tex0.th, c.tex0.tcc, c.tex0.tfx,
                     c.tex0.cbp, c.tex0.cpsm, c.tex0.csm, c.tex0.csa, c.tex0.cld,
                     static_cast<unsigned long long>(c.tex1), static_cast<unsigned long long>(c.clamp),
                     static_cast<unsigned long long>(c.alpha), static_cast<unsigned long long>(c.test),
                     static_cast<unsigned long long>(c.fba), c.scissor.x0, c.scissor.x1, c.scissor.y0, c.scissor.y1,
                     c.xyoffset.ofx, c.xyoffset.ofy, st.texa.ta0, st.texa.aem, st.texa.ta1, st.pabe,
                     static_cast<unsigned long long>(st.dthe), static_cast<unsigned long long>(st.colclamp));
        for (uint32_t i = 0; i < batch.vertexCount; ++i)
        {
            const GSVertex &v = batch.vertices[i];
            std::fprintf(file, " | %.2f,%.2f z=%.0f rgba=%02x%02x%02x%02x uv=%.2f,%.2f st=%g,%g q=%g",
                         v.x, v.y, v.z, v.r, v.g, v.b, v.a, v.u / 16.0, v.v / 16.0, v.s, v.t, v.q);
        }
        std::fputc('\n', file);
    }

    int writeText(const GSGolden::Recording &recording, const std::string &path)
    {
        std::FILE *file = std::fopen(path.c_str(), "w");
        if (!file)
            return 1;
        auto writeField = [&](const GSRecording::Checkpoint &cp)
        {
            std::fprintf(file,
                         "V tick=%llu pmode=%llx smode2=%llx dispfb1=%llx display1=%llx dispfb2=%llx display2=%llx bgcolor=%llx\n",
                         static_cast<unsigned long long>(cp.tick),
                         static_cast<unsigned long long>(cp.display[0]), static_cast<unsigned long long>(cp.display[1]),
                         static_cast<unsigned long long>(cp.display[2]), static_cast<unsigned long long>(cp.display[3]),
                         static_cast<unsigned long long>(cp.display[4]), static_cast<unsigned long long>(cp.display[5]),
                         static_cast<unsigned long long>(cp.display[6]));
        };
        writeField(recording.initial);
        for (const GSRecording::Record &record : recording.records)
        {
            switch (record.op)
            {
            case GSRecording::Op::Submit:
            {
                GSPrimitiveBatch batch{};
                std::memcpy(&batch, record.payload.data(), sizeof(batch));
                writeDraw(file, batch);
                break;
            }
            case GSRecording::Op::BeginTransfer:
            {
                GSTransferCommand cmd{};
                std::memcpy(&cmd, record.payload.data(), sizeof(cmd));
                const GSBitBltBuf &b = cmd.bitbltbuf;
                std::fprintf(file, "T dir=%u src=%x/%u/%02x dst=%x/%u/%02x ss=%u,%u ds=%u,%u size=%ux%u\n",
                             cmd.direction, b.sbp, b.sbw, b.spsm, b.dbp, b.dbw, b.dpsm, cmd.trxpos.ssax,
                             cmd.trxpos.ssay, cmd.trxpos.dsax, cmd.trxpos.dsay, cmd.trxreg.rrw, cmd.trxreg.rrh);
                break;
            }
            case GSRecording::Op::Upload:
                std::fprintf(file, "I %u ", static_cast<unsigned>(record.payload.size()));
                for (uint8_t byte : record.payload)
                    std::fprintf(file, "%02x", byte);
                std::fputc('\n', file);
                break;
            case GSRecording::Op::Field:
            {
                GSRecording::Checkpoint cp;
                if (GSRecording::ParseCheckpoint(record, cp))
                    writeField(cp);
                break;
            }
            default:
                break;
            }
        }
        std::fclose(file);
        return 0;
    }
}

namespace
{
    std::string profileKey(const GSRecording::Record &record)
    {
        if (record.op != GSRecording::Op::Submit)
            return "op" + std::to_string(static_cast<int>(record.op));
        GSPrimitiveBatch batch{};
        std::memcpy(&batch, record.payload.data(), sizeof(batch));
        const GSContext &c = batch.state.context;
        const GSPrimReg &p = batch.state.prim;
        char key[256];
        std::snprintf(key, sizeof(key), "prim%u tme%d fst%d lin%d tpsm%02x abe%d fpsm%02x test%05llx alpha%02llx fge%d iip%d",
                      static_cast<unsigned>(p.type), p.tme, p.fst, batch.state.linearFilter, p.tme ? c.tex0.psm : 0xFF, p.abe,
                      c.frame.psm, static_cast<unsigned long long>(c.test & 0x7FFFF),
                      static_cast<unsigned long long>(c.alpha & 0xFF), p.fge, p.iip);
        return key;
    }

    void profile(const GSGolden::Recording &recording, GSRasterBackend &backend)
    {
        std::vector<uint8_t> vram = recording.initial.vram;
        backend.Initialize(vram.data(), static_cast<uint32_t>(vram.size()));
        if (auto *access = dynamic_cast<GSBackendStateAccess *>(&backend))
            access->ImportState(recording.initial.state);
        std::map<std::string, std::pair<double, size_t>> totals;
        double all = 0.0;
        for (const GSRecording::Record &record : recording.records)
        {
            const auto start = std::chrono::steady_clock::now();
            GSRecording::Apply(backend, record);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            auto &entry = totals[profileKey(record)];
            entry.first += seconds;
            entry.second += 1u;
            all += seconds;
        }
        std::vector<std::pair<double, std::string>> sorted;
        for (const auto &kv : totals)
            sorted.push_back({kv.second.first, kv.first + " x" + std::to_string(kv.second.second)});
        std::sort(sorted.rbegin(), sorted.rend());
        std::printf("total %.3f s\n", all);
        const auto &fallbacks = GSRaster::FallbackCounters();
        std::printf("fallbacks: spriteFill %llu spriteSetup %llu spriteTex %llu spriteOverlap %llu spriteCache %llu triTex %llu triOverlap %llu triCache %llu\n",
                    (unsigned long long)fallbacks[0], (unsigned long long)fallbacks[1], (unsigned long long)fallbacks[2],
                    (unsigned long long)fallbacks[3], (unsigned long long)fallbacks[4], (unsigned long long)fallbacks[5],
                    (unsigned long long)fallbacks[6], (unsigned long long)fallbacks[7]);
        for (size_t i = 0; i < sorted.size() && i < 25; ++i)
            std::printf("%8.1f ms %5.1f%%  %s\n", sorted[i].first * 1000.0, 100.0 * sorted[i].first / all, sorted[i].second.c_str());
    }
}

namespace GSRaster
{
    long ExactLround(float value);
}

namespace
{
    int checkLround()
    {
        const unsigned modes[] = {0x1F80u, 0xFFC0u, 0x7F80u};
        const unsigned saved = _mm_getcsr();
        int failures = 0;
        for (unsigned mode : modes)
        {
            _mm_setcsr(mode);
            uint64_t checked = 0u;
            for (uint32_t bits = 0u;; ++bits)
            {
                float value;
                std::memcpy(&value, &bits, 4u);
                if (std::fabs(value) <= 1024.0f)
                {
                    ++checked;
                    if (GSRaster::ExactLround(value) != std::lround(value))
                    {
                        if (failures++ < 10)
                            std::printf("mismatch mode %x value %a: exact %ld lround %ld\n", mode, value,
                                        GSRaster::ExactLround(value), std::lround(value));
                    }
                }
                if (bits == 0xFFFFFFFFu)
                    break;
            }
            std::printf("mode %x: %llu floats checked\n", mode, static_cast<unsigned long long>(checked));
        }
        _mm_setcsr(saved);
        std::printf("%s\n", failures ? "LROUND MISMATCH" : "lround equivalent");
        return failures ? 1 : 0;
    }
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--check-lround")
        return checkLround();
    if (argc < 2)
    {
        std::cerr << "usage: gs_replay <recording.gsr> [--backend cpu|ref] [--repeat N] [--no-compare]"
                     " [--lockstep] [--text out.txt]\n";
        return 2;
    }
    const std::string path = argv[1];
    if (const char *csr = std::getenv("GS_REPLAY_MXCSR"))
        _mm_setcsr(static_cast<unsigned>(std::strtoul(csr, nullptr, 16)));
    std::string backendName = "cpu";
    int repeat = 1;
    bool compare = true;
    bool lockstep = false;
    bool doProfile = false;
    bool doSample = false;
    std::string textPath;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc)
            backendName = argv[++i];
        else if (arg == "--repeat" && i + 1 < argc)
            repeat = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--no-compare")
            compare = false;
        else if (arg == "--lockstep")
            lockstep = true;
        else if (arg == "--profile")
            doProfile = true;
        else if (arg == "--sample")
            doSample = true;
        else if (arg == "--text" && i + 1 < argc)
            textPath = argv[++i];
    }

    GSGolden::Recording recording;
    std::string error;
    if (!GSGolden::Load(path, recording, error))
    {
        std::cerr << "load failed: " << error << "\n";
        return 1;
    }
    std::cout << path << ": " << recording.records.size() << " records, " << recording.submits << " draws, "
              << recording.checkpoints << " checkpoints\n";

    if (!textPath.empty())
        return writeText(recording, textPath);

    if (doProfile)
    {
        auto backend = makeBackend(backendName);
        profile(recording, *backend);
        return 0;
    }

    if (lockstep)
    {
        auto reference = makeBackend("ref");
        auto candidate = makeBackend(backendName);
        if (!candidate)
        {
            std::cerr << "unknown backend " << backendName << "\n";
            return 2;
        }
        const GSGolden::Report report = GSGolden::Lockstep(recording, *reference, *candidate);
        std::cout << (report.ok ? "lockstep OK" : "lockstep MISMATCH: " + report.message) << "\n";
        return report.ok ? 0 : 1;
    }

    if (doSample)
    {
        GSSampler sampler;
        sampler.Start();
        for (int run = 0; run < repeat; ++run)
        {
            auto backend = makeBackend(backendName);
            GSGolden::Replay(recording, *backend, false);
        }
        sampler.StopAndReport(40);
        return 0;
    }

    int status = 0;
    double best = 1e30;
    for (int run = 0; run < repeat; ++run)
    {
        auto backend = makeBackend(backendName);
        if (!backend)
        {
            std::cerr << "unknown backend " << backendName << "\n";
            return 2;
        }
        const GSGolden::Report report = GSGolden::Replay(recording, *backend, compare);
        best = std::min(best, report.seconds);
        std::printf("run %d [%s]: %.3f s (%.1f ms/field) %s%s\n", run, backendName.c_str(), report.seconds,
                    recording.checkpoints ? 1000.0 * report.seconds / static_cast<double>(recording.checkpoints) : 0.0,
                    compare ? (report.ok ? "bit-exact" : "MISMATCH: ") : "", report.ok ? "" : report.message.c_str());
        if (!report.ok)
            status = 1;
    }
    std::printf("best: %.3f s\n", best);
    return status;
}
