#include "MiniTest.h"
#include "gs_golden.h"
#include "gs_reference_backend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_threaded_backend.h"
#include "runtime/gs/gs_swizzle.h"
#include "runtime/gs/ps2_gs_memory.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <xmmintrin.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace
{
    void runGolden(TestCase &t, const std::string &path, std::unique_ptr<GSRasterBackend> backend)
    {
        GSGolden::Recording recording;
        std::string error;
        if (!GSGolden::Load(path, recording, error))
        {
            t.Fail(path + ": " + error);
            return;
        }
        GSGolden::Report report = GSGolden::Replay(recording, *backend, true, false);
        if (!report.ok)
        {
            const GSGolden::Report detailed = GSGolden::Replay(recording, *backend, true, true);
            if (!detailed.ok)
                report.message = detailed.message;
        }
        t.IsTrue(report.ok, path + ": " + report.message);
        t.Equals(report.checkpointsCompared, recording.checkpoints, path + ": every checkpoint compared");
    }
}

namespace GSRaster
{
    long ExactLround(float value);
}

void register_gs_golden_tests()
{
    MiniTest::Case("GSGolden", [](TestCase &tc)
                   {
        tc.Run("fast swizzle matches GSMem reads and writes for every format", [](TestCase &t)
               {
            GSMem::InitLookupTables();
            using ReadFn = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
            using WriteFn = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
            struct Entry { uint32_t psm; ReadFn read; WriteFn write; };
            const Entry entries[] = {
                {0x00, GSMem::ReadCT32, GSMem::WriteCT32}, {0x01, GSMem::ReadCT24, GSMem::WriteCT24},
                {0x02, GSMem::ReadCT16, GSMem::WriteCT16}, {0x0A, GSMem::ReadCT16S, GSMem::WriteCT16S},
                {0x13, GSMem::ReadP8, GSMem::WriteP8}, {0x14, GSMem::ReadP4, GSMem::WriteP4},
                {0x1B, GSMem::ReadP8H, GSMem::WriteP8H}, {0x24, GSMem::ReadP4HL, GSMem::WriteP4HL},
                {0x2C, GSMem::ReadP4HH, GSMem::WriteP4HH}, {0x30, GSMem::ReadZ32, GSMem::WriteZ32},
                {0x31, GSMem::ReadZ24, GSMem::WriteZ24}, {0x32, GSMem::ReadZ16, GSMem::WriteZ16},
                {0x3A, GSMem::ReadZ16S, GSMem::WriteZ16S}};
            std::vector<uint8_t> a(4u << 20u);
            std::vector<uint8_t> b(4u << 20u);
            std::mt19937 rng(1234u);
            for (auto &byte : a)
                byte = static_cast<uint8_t>(rng());
            b = a;
            bool ok = true;
            for (const Entry &e : entries)
            {
                const GSSwizzle::Format &f = GSSwizzle::GetFormat(e.psm);
                for (int i = 0; i < 200000 && ok; ++i)
                {
                    const uint32_t base = rng() & 0x3FFFu;
                    const uint32_t bw = rng() & 0x3Fu;
                    const uint32_t x = rng() & 0x7FFu;
                    const uint32_t y = rng() & 0x7FFu;
                    const uint32_t value = rng();
                    const GSSwizzle::Location loc = GSSwizzle::Locate(f, base, bw, x, y);
                    if (GSSwizzle::LoadAt(f, a.data() + loc.byte, loc.shift) != e.read(a.data(), base, bw, x, y))
                        ok = false;
                    GSSwizzle::StoreAt(f, a.data() + loc.byte, loc.shift, value);
                    e.write(b.data(), base, bw, x, y, value);
                }
                if (!ok || a != b)
                {
                    t.Fail("swizzle mismatch for psm " + std::to_string(e.psm));
                    return;
                }
            } });
        tc.Run("rasterizer lround matches std::lround under the EE rounding mode", [](TestCase &t)
               {
            const unsigned saved = _mm_getcsr();
            std::mt19937 rng(99u);
            bool ok = true;
            for (unsigned mode : {0x1F80u, 0xFFC0u})
            {
                _mm_setcsr(mode);
                for (int i = 0; i < 4000000 && ok; ++i)
                {
                    uint32_t bits = rng();
                    float value;
                    std::memcpy(&value, &bits, 4u);
                    if (!(std::fabs(value) <= 1024.0f))
                        value = std::ldexp(static_cast<float>(rng() & 0xFFFFFFu), -14) - 512.0f;
                    ok = GSRaster::ExactLround(value) == std::lround(value);
                }
                for (int k = -2048; k <= 2048 && ok; ++k)
                {
                    const float half = static_cast<float>(k) * 0.5f;
                    ok = GSRaster::ExactLround(half) == std::lround(half) &&
                         GSRaster::ExactLround(std::nextafter(half, 0.0f)) == std::lround(std::nextafter(half, 0.0f));
                }
            }
            _mm_setcsr(saved);
            t.IsTrue(ok, "ExactLround differs from std::lround"); });
        const std::string directory = GSGolden::DefaultDirectory();
        const std::vector<std::string> recordings = GSGolden::ListRecordings(directory);
        tc.Run("golden GS recordings are present", [directory, recordings](TestCase &t)
               { t.IsFalse(recordings.empty(), "no .gsr recordings in " + directory + " (set PS2X_GS_GOLDEN_DIR)"); });
        for (const std::string &path : recordings)
        {
            const std::string name = std::filesystem::path(path).stem().string();
            tc.Run("golden " + name + " replays bit-exact through the reference rasterizer", [path](TestCase &t)
                   { runGolden(t, path, std::make_unique<GSReferenceBackend>()); });
            tc.Run("golden " + name + " replays bit-exact through GSCpuBackend", [path](TestCase &t)
                   { runGolden(t, path, std::make_unique<GSCpuBackend>()); });
            tc.Run("golden " + name + " replays bit-exact through the threaded GS", [path](TestCase &t)
                   { runGolden(t, path, std::make_unique<GSThreadedBackend>(std::make_unique<GSCpuBackend>())); });
        }
    });
}
