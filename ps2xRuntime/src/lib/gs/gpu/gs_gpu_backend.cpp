#include "runtime/gs/gs_gpu_backend.h"
#include "runtime/gs/gs_swizzle.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "../gs_cpu_backend_internal.h"
#include "../gs_sprite_rules.h"
#include "../gs_triangle_rules.h"
#include "gs_gl.h"
#include "gs_gpu_shaders.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace GSGL;
using namespace GSCpuInternal;

namespace
{
    constexpr uint32_t kPrimWords = 40u;
    constexpr uint32_t kStateWords = 32u;
    constexpr uint32_t kTileSize = 16u;
    constexpr uint32_t kTileGrid = 128u;
    constexpr uint32_t kClutSlots = 1024u;
    constexpr uint32_t kPages = 512u;
    constexpr uint32_t kMaxPresentationWidth = 2048u;
    constexpr uint32_t kMaxPresentationHeight = 2048u;

    enum PrimType : uint32_t
    {
        PrimPoint = 0u,
        PrimSprite = 1u,
        PrimTriangle = 2u,
    };

    enum StateWord : uint32_t
    {
        SFbp = 0u,
        SFbw,
        SFpsm,
        SFbmsk,
        SZbp,
        SZpsm,
        SZmask,
        SScissorX,
        SScissorY,
        STbp0,
        STbw,
        STpsm,
        STexMode,
        SCsa,
        STexSize,
        SClampLo,
        SClampHi,
        SAlpha,
        SAlphaFix,
        STest,
        SFba,
        SFlags,
        STexa,
        SFogCol,
        SClut,
    };

    constexpr uint32_t FIip = 1u;
    constexpr uint32_t FTme = 2u;
    constexpr uint32_t FFge = 4u;
    constexpr uint32_t FAbe = 8u;
    constexpr uint32_t FFst = 16u;
    constexpr uint32_t FPabe = 32u;
    constexpr uint32_t FLinear = 64u;
    constexpr uint32_t FWrap = 128u;

    uint32_t fbits(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    uint32_t packColor(const GSVertex &v)
    {
        return static_cast<uint32_t>(v.r) | (static_cast<uint32_t>(v.g) << 8u) | (static_cast<uint32_t>(v.b) << 16u) |
               (static_cast<uint32_t>(v.a) << 24u);
    }

    struct ClutOp
    {
        uint32_t target = 0;
        uint32_t source = 0;
        uint32_t psm = 0;
        uint32_t cbp = 0;
        uint32_t packed = 0;
        uint32_t cbw = 0;
        uint32_t cou = 0;
        uint32_t cov = 0;
        std::array<uint16_t, 32> writers{};
    };

    std::atomic<bool> s_asyncPresent{false};

    constexpr size_t kRingChunk = 16u * 1024u * 1024u;
    constexpr uint32_t kRingChunks = 4u;
    constexpr size_t kRingAlign = 256u;

    struct Readback
    {
        GLuint buffer = 0;
        const uint8_t *mapped = nullptr;
        size_t capacity = 0;
        GLsync fence = nullptr;
        GLuint query = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t displayFbp = 0;
    };

    struct Program
    {
        GLuint id = 0;
        GLint loc[4] = {-1, -1, -1, -1};
    };

    struct CrtcCircuit
    {
        bool enabled = false;
        uint32_t fbp = 0u;
        uint32_t fbw = 1u;
        uint8_t psm = 0u;
        uint32_t dbx = 0u;
        uint32_t dby = 0u;
        int32_t dx = 0;
        int32_t dy = 0;
        int32_t dw = 0;
        int32_t dh = 0;
        int32_t hdiv = 1;
        int32_t vdiv = 1;
    };

    CrtcCircuit decodeCircuit(bool enabled, uint64_t dispfb, uint64_t display, bool halfHeightBuffer)
    {
        CrtcCircuit c{};
        c.fbp = static_cast<uint32_t>(dispfb & 0x1FFu);
        c.fbw = std::max<uint32_t>(1u, static_cast<uint32_t>((dispfb >> 9) & 0x3Fu));
        c.psm = static_cast<uint8_t>((dispfb >> 15) & 0x1Fu);
        c.dbx = static_cast<uint32_t>((dispfb >> 32) & 0x7FFu);
        c.dby = static_cast<uint32_t>((dispfb >> 43) & 0x7FFu);
        c.dx = static_cast<int32_t>(display & 0xFFFu);
        c.dy = static_cast<int32_t>((display >> 12) & 0x7FFu);
        c.hdiv = static_cast<int32_t>((display >> 23) & 0xFu) + 1;
        c.vdiv = (static_cast<int32_t>((display >> 27) & 0x3u) + 1) * (halfHeightBuffer ? 2 : 1);
        c.dw = static_cast<int32_t>((display >> 32) & 0xFFFu) + 1;
        c.dh = static_cast<int32_t>((display >> 44) & 0x7FFu) + 1;
        c.enabled = enabled && display != 0u && c.dw >= c.hdiv && c.dh >= c.vdiv;
        return c;
    }

    void markPages(std::bitset<kPages> &pages, uint32_t psm, uint32_t base, uint32_t bw, int x0, int y0, int x1, int y1)
    {
        const GSSwizzle::Format &format = GSSwizzle::GetFormat(psm);
        if (format.kind == GSSwizzle::Kind::Invalid || x1 < x0 || y1 < y0)
            return;
        const uint32_t pagesPerRow = (bw * 64u) >> format.pageShiftX;
        const uint32_t basePage = base >> 5u;
        const bool spill = (base & 31u) != 0u;
        const uint32_t px0 = static_cast<uint32_t>(std::max(x0, 0)) >> format.pageShiftX;
        const uint32_t px1 = static_cast<uint32_t>(std::max(x1, 0)) >> format.pageShiftX;
        const uint32_t py0 = static_cast<uint32_t>(std::max(y0, 0)) >> format.pageShiftY;
        const uint32_t py1 = static_cast<uint32_t>(std::max(y1, 0)) >> format.pageShiftY;
        if ((py1 - py0 + 1u) * (px1 - px0 + 1u) >= kPages)
        {
            pages.set();
            return;
        }
        for (uint32_t py = py0; py <= py1; ++py)
            for (uint32_t px = px0; px <= px1; ++px)
            {
                const uint32_t page = basePage + py * pagesPerRow + px;
                pages.set(page % kPages);
                if (spill)
                    pages.set((page + 1u) % kPages);
            }
    }
}

struct GSGpuBackend::Impl
{
    GSGpuBackend::Stats &stats;
    Context context;
    bool initialized = false;
    bool failed = false;
    uint8_t *hostVram = nullptr;
    uint32_t hostVramSize = 0u;

    GLuint vramBuffer = 0;
    GLuint swizzleBuffer = 0;
    GLuint clutBuffer = 0;
    GLuint primBuffer = 0;
    GLuint stateBuffer = 0;
    GLuint tileBuffer = 0;
    GLuint tileListBuffer = 0;
    GLuint dataBuffer = 0;
    GLuint clutOpBuffer = 0;
    std::vector<uint32_t> clutOpWords;
    std::array<uint16_t, 32> clutWriters{};
    size_t dataCapacity = 0;
    GLuint ringBuffer = 0;
    uint8_t *ring = nullptr;
    uint32_t ringChunk = 0u;
    size_t ringOffset = 0u;
    std::array<GLsync, kRingChunks> ringFences{};

    Program raster;
    Program clutLoad;
    Program upload;
    Program copyRead;
    Program copyWrite;
    Program clear;
    Program poke;
    Program present;

    std::vector<uint32_t> prims;
    uint32_t primCount = 0u;
    std::vector<uint32_t> states;
    std::array<uint32_t, kStateWords> lastState{};
    uint32_t stateCount = 0u;
    std::vector<std::vector<uint32_t>> bins;
    std::vector<uint32_t> touched;
    std::vector<uint32_t> tileHeaders;
    std::vector<uint32_t> tileList;
    std::bitset<kPages> writePages;
    std::bitset<kPages> readPages;
    uint64_t targetKey = 0u;
    bool hasTarget = false;
    std::vector<ClutOp> clutOps;
    uint32_t clutSlot = 0u;
    bool clutKeyValid = false;
    std::array<uint32_t, 6> clutKey{};
    std::bitset<kPages> clutSourcePages;
    std::array<uint32_t, 2> clutCbp{};

    std::array<Readback, 3> readbacks{};
    uint32_t readbackNext = 0u;
    PresentationFrame lastFrame;

    struct PresentStats
    {
        bool enabled = false;
        bool queryOpen = false;
        std::array<GLuint, 4> queries{};
        uint32_t queryNext = 0u;
        uint32_t frames = 0u;
        uint64_t gpuNs = 0u;
        uint64_t gpuFrames = 0u;
        std::chrono::steady_clock::duration wait{};
        std::chrono::steady_clock::duration submit{};
        std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
        uint64_t batches = 0u;
        uint64_t clutLoads = 0u;
        uint64_t transfers = 0u;
    } presentStats;

    GSTransferCommand transfer{};
    GSTransferSnapshot transferState{};
    std::vector<uint8_t> localToHost;
    size_t localToHostReadPos = 0u;
    std::vector<uint32_t> scratch;

    explicit Impl(GSGpuBackend::Stats &s) : stats(s)
    {
        const char *statsValue = std::getenv("PS2X_GS_GPU_STATS");
        presentStats.enabled = statsValue && *statsValue && *statsValue != '0';
        bins.resize(kTileGrid * kTileGrid);
        transfer.direction = 3u;
        transferState.direction = 3u;
    }

    ~Impl()
    {
        if (initialized)
        {
            const Api &gl = context.gl();
            for (Readback &slot : readbacks)
            {
                if (slot.fence)
                    gl.DeleteSync(slot.fence);
                if (slot.buffer)
                    gl.DeleteBuffers(1, &slot.buffer);
            }
            for (GLuint query : presentStats.queries)
                if (query)
                    gl.DeleteQueries(1, &query);
            const GLuint buffers[] = {vramBuffer, swizzleBuffer, clutBuffer, primBuffer, stateBuffer, tileBuffer, tileListBuffer, dataBuffer, clutOpBuffer};
            gl.DeleteBuffers(static_cast<GLsizei>(std::size(buffers)), buffers);
            for (Program *program : {&raster, &clutLoad, &upload, &copyRead, &copyWrite, &clear, &poke, &present})
                if (program->id)
                    gl.DeleteProgram(program->id);
        }
    }

    const Api &gl() const
    {
        return context.gl();
    }

    GLuint makeBuffer(size_t size, const void *data, GLuint binding)
    {
        GLuint buffer = 0;
        gl().GenBuffers(1, &buffer);
        gl().BindBuffer(kShaderStorageBuffer, buffer);
        gl().BufferData(kShaderStorageBuffer, static_cast<GLsizeiptr>(size), data, kDynamicDraw);
        gl().BindBufferBase(kShaderStorageBuffer, binding, buffer);
        return buffer;
    }

    size_t ringPut(const void *data, size_t size)
    {
        size = std::max<size_t>(size, 16u);
        if (!ring || size > kRingChunk)
            return SIZE_MAX;
        ringOffset = (ringOffset + kRingAlign - 1u) & ~(kRingAlign - 1u);
        if (ringOffset + size > kRingChunk)
        {
            ringFences[ringChunk] = gl().FenceSync(kSyncGpuCommandsComplete, 0u);
            ringChunk = (ringChunk + 1u) % kRingChunks;
            if (GLsync fence = ringFences[ringChunk])
            {
                gl().ClientWaitSync(fence, kSyncFlushCommandsBit, 5000000000ull);
                gl().DeleteSync(fence);
                ringFences[ringChunk] = nullptr;
            }
            ringOffset = 0u;
        }
        const size_t offset = static_cast<size_t>(ringChunk) * kRingChunk + ringOffset;
        if (data)
            std::memcpy(ring + offset, data, size);
        ringOffset += size;
        return offset;
    }

    void bindUpload(GLuint binding, GLuint fallbackBuffer, const void *data, size_t size)
    {
        const size_t offset = ringPut(data, size);
        if (offset == SIZE_MAX)
        {
            setBuffer(fallbackBuffer, size, data);
            gl().BindBufferBase(kShaderStorageBuffer, binding, fallbackBuffer);
            return;
        }
        gl().BindBufferRange(kShaderStorageBuffer, binding, ringBuffer, static_cast<GLintptr>(offset),
                             static_cast<GLsizeiptr>(std::max<size_t>(size, 16u)));
    }

    void setBuffer(GLuint buffer, size_t size, const void *data)
    {
        gl().BindBuffer(kShaderStorageBuffer, buffer);
        gl().BufferData(kShaderStorageBuffer, static_cast<GLsizeiptr>(std::max<size_t>(size, 16u)), nullptr, kStreamDraw);
        if (size != 0u && data)
            gl().BufferSubData(kShaderStorageBuffer, 0, static_cast<GLsizeiptr>(size), data);
    }

    bool compile(Program &program, const std::string &prefix, const char *body, std::initializer_list<const char *> uniforms)
    {
        const std::string source = prefix + GSGpuShaders::kCommon + body;
        const char *text = source.c_str();
        const GLuint shader = gl().CreateShader(kComputeShader);
        gl().ShaderSource(shader, 1, &text, nullptr);
        gl().CompileShader(shader);
        GLint ok = 0;
        gl().GetShaderiv(shader, kCompileStatus, &ok);
        if (!ok)
        {
            GLint length = 0;
            gl().GetShaderiv(shader, kInfoLogLength, &length);
            std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
            gl().GetShaderInfoLog(shader, length, nullptr, log.data());
            std::fprintf(stderr, "[gs-gpu] shader compile failed:\n%s\n", log.c_str());
            std::fflush(stderr);
            gl().DeleteShader(shader);
            return false;
        }
        program.id = gl().CreateProgram();
        gl().AttachShader(program.id, shader);
        gl().LinkProgram(program.id);
        gl().DeleteShader(shader);
        gl().GetProgramiv(program.id, kLinkStatus, &ok);
        if (!ok)
        {
            GLint length = 0;
            gl().GetProgramiv(program.id, kInfoLogLength, &length);
            std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
            gl().GetProgramInfoLog(program.id, length, nullptr, log.data());
            std::fprintf(stderr, "[gs-gpu] program link failed:\n%s\n", log.c_str());
            std::fflush(stderr);
            return false;
        }
        size_t index = 0;
        for (const char *name : uniforms)
            program.loc[index++] = gl().GetUniformLocation(program.id, name);
        return true;
    }

    bool ensure()
    {
        if (initialized)
            return true;
        if (failed)
            return false;
        std::string error;
        if (!context.Create(error))
        {
            std::fprintf(stderr, "[gs-gpu] %s\n", error.c_str());
            std::fflush(stderr);
            failed = true;
            return false;
        }
        std::fprintf(stderr, "[gs-gpu] %s / %s\n", reinterpret_cast<const char *>(gl().GetString(kRenderer)),
                     reinterpret_cast<const char *>(gl().GetString(kVersion)));
        std::fflush(stderr);

        GSSwizzle::GetFormat(0u);
        using GSMem::PixelStorageMode;
        const std::pair<const char *, PixelStorageMode> tables[] = {
            {"TABLE_C32", PixelStorageMode::C32}, {"TABLE_Z32", PixelStorageMode::Z32}, {"TABLE_C16", PixelStorageMode::C16},
            {"TABLE_C16S", PixelStorageMode::C16S}, {"TABLE_Z16", PixelStorageMode::Z16}, {"TABLE_Z16S", PixelStorageMode::Z16S},
            {"TABLE_P8", PixelStorageMode::P8}, {"TABLE_P4", PixelStorageMode::P4}};
        const uint32_t sizes[] = {65536u, 65536u, 131072u, 131072u, 131072u, 131072u, 262144u, 524288u};
        std::vector<uint16_t> swizzle;
        std::string prefix = "#version 460 core\n";
        for (size_t i = 0; i < std::size(tables); ++i)
        {
            prefix += "#define " + std::string(tables[i].first) + " " + std::to_string(swizzle.size()) + "u\n";
            const uint16_t *data = GSMem::PageTableData(tables[i].second);
            swizzle.insert(swizzle.end(), data, data + sizes[i]);
        }
        std::string common = GSGpuShaders::kCommon;
        (void)common;

        vramBuffer = makeBuffer(GSSwizzle::kMemorySize, nullptr, 0u);
        swizzleBuffer = makeBuffer(swizzle.size() * sizeof(uint16_t), swizzle.data(), 1u);
        std::vector<uint32_t> zeros(kClutSlots * 256u, 0u);
        clutBuffer = makeBuffer(zeros.size() * sizeof(uint32_t), zeros.data(), 2u);
        primBuffer = makeBuffer(16u, nullptr, 3u);
        stateBuffer = makeBuffer(16u, nullptr, 4u);
        tileBuffer = makeBuffer(16u, nullptr, 5u);
        tileListBuffer = makeBuffer(16u, nullptr, 6u);
        dataBuffer = makeBuffer(16u, nullptr, 7u);
        clutOpBuffer = makeBuffer(16u, nullptr, 8u);
        dataCapacity = 16u;
        gl().GenBuffers(1, &ringBuffer);
        gl().BindBuffer(kShaderStorageBuffer, ringBuffer);
        gl().BufferStorage(kShaderStorageBuffer, static_cast<GLsizeiptr>(kRingChunk * kRingChunks), nullptr,
                           kMapWriteBit | kMapPersistentBit | kMapCoherentBit);
        ring = static_cast<uint8_t *>(gl().MapBufferRange(kShaderStorageBuffer, 0, static_cast<GLsizeiptr>(kRingChunk * kRingChunks),
                                                          kMapWriteBit | kMapPersistentBit | kMapCoherentBit));

        const bool ok = compile(raster, prefix, GSGpuShaders::kRaster, {}) &&
                        compile(clutLoad, prefix, GSGpuShaders::kClutLoad, {"uClutA", "uClutB"}) &&
                        compile(upload, prefix, GSGpuShaders::kUpload, {"uDst", "uRange"}) &&
                        compile(copyRead, prefix, GSGpuShaders::kCopyRead, {"uSrc", "uSize"}) &&
                        compile(copyWrite, prefix, GSGpuShaders::kCopyWrite, {"uDst", "uSize"}) &&
                        compile(clear, prefix, GSGpuShaders::kClear, {"uFrame", "uRect"}) &&
                        compile(poke, prefix, GSGpuShaders::kPoke, {"uArgs", "uArgs2"}) &&
                        compile(present, prefix, GSGpuShaders::kPresent, {"uOut", "uC1", "uC2", "uMode"});
        if (!ok)
        {
            failed = true;
            return false;
        }
        initialized = true;
        if (hostVram)
            uploadHostVram();
        return true;
    }

    void barrier()
    {
        gl().MemBarrier(kAllBarrierBits);
    }

    void uploadHostVram()
    {
        clutKeyValid = false;
        if (!hostVram)
            return;
        gl().BindBuffer(kShaderStorageBuffer, vramBuffer);
        gl().BufferSubData(kShaderStorageBuffer, 0, static_cast<GLsizeiptr>(std::min<uint32_t>(hostVramSize, GSSwizzle::kMemorySize)), hostVram);
    }

    void downloadVram(uint8_t *out, size_t size)
    {
        barrier();
        gl().BindBuffer(kShaderStorageBuffer, vramBuffer);
        gl().GetBufferSubData(kShaderStorageBuffer, 0, static_cast<GLsizeiptr>(std::min<size_t>(size, GSSwizzle::kMemorySize)), out);
    }

    void readData(void *out, size_t size)
    {
        barrier();
        gl().BindBuffer(kShaderStorageBuffer, dataBuffer);
        gl().GetBufferSubData(kShaderStorageBuffer, 0, static_cast<GLsizeiptr>(size), out);
    }

    void resetBatch()
    {
        prims.clear();
        primCount = 0u;
        states.clear();
        stateCount = 0u;
        for (uint32_t tile : touched)
            bins[tile].clear();
        touched.clear();
        writePages.reset();
        readPages.reset();
        hasTarget = false;
        clutOps.clear();
    }

    void flushBatch()
    {
        if (!initialized || (primCount == 0u && clutOps.empty()))
        {
            resetBatch();
            return;
        }
        const Api &g = gl();
        if (!clutOps.empty())
        {
            clutOpWords.assign(clutOps.size() * 24u, 0u);
            for (size_t i = 0; i < clutOps.size(); ++i)
            {
                const ClutOp &op = clutOps[i];
                uint32_t *w = clutOpWords.data() + i * 24u;
                w[0] = op.target;
                w[1] = isFourBitIndexedPsm(static_cast<uint8_t>(op.psm)) ? 1u : 0u;
                w[2] = op.cbp;
                w[3] = op.packed;
                w[4] = op.cbw;
                w[5] = op.cou;
                w[6] = op.cov;
                for (uint32_t b = 0; b < 16u; ++b)
                    w[8u + b] = static_cast<uint32_t>(op.writers[b * 2u]) | (static_cast<uint32_t>(op.writers[b * 2u + 1u]) << 16u);
            }
            bindUpload(8u, clutOpBuffer, clutOpWords.data(), clutOpWords.size() * sizeof(uint32_t));
            g.UseProgram(clutLoad.id);
            g.Uniform4ui(clutLoad.loc[0], clutOps.front().source, 0u, 0u, 0u);
            g.DispatchCompute(static_cast<GLuint>(clutOps.size()), 1u, 1u);
            barrier();
            stats.clutLoads += clutOps.size();
        }
        if (primCount != 0u)
        {
            tileHeaders.clear();
            tileList.clear();
            for (uint32_t tile : touched)
            {
                const std::vector<uint32_t> &bin = bins[tile];
                tileHeaders.push_back(tile % kTileGrid);
                tileHeaders.push_back(tile / kTileGrid);
                tileHeaders.push_back(static_cast<uint32_t>(tileList.size()));
                tileHeaders.push_back(static_cast<uint32_t>(bin.size()));
                tileList.insert(tileList.end(), bin.begin(), bin.end());
            }
            bindUpload(3u, primBuffer, prims.data(), prims.size() * sizeof(uint32_t));
            bindUpload(4u, stateBuffer, states.data(), states.size() * sizeof(uint32_t));
            bindUpload(5u, tileBuffer, tileHeaders.data(), tileHeaders.size() * sizeof(uint32_t));
            bindUpload(6u, tileListBuffer, tileList.data(), tileList.size() * sizeof(uint32_t));
            g.UseProgram(raster.id);
            g.DispatchCompute(static_cast<GLuint>(touched.size()), 1u, 1u);
            barrier();
            stats.batches += 1u;
            stats.tiles += touched.size();
            stats.prims += primCount;
        }
        resetBatch();
    }

    uint32_t addState(const GSDrawState &state)
    {
        const GSContext &ctx = state.context;
        std::array<uint32_t, kStateWords> s{};
        s[SFbp] = ctx.frame.fbp << 5u;
        s[SFbw] = std::max<uint32_t>(ctx.frame.fbw, 1u);
        s[SFpsm] = ctx.frame.psm;
        s[SFbmsk] = ctx.frame.fbmsk;
        s[SZbp] = ctx.zbuf.zbp << 5u;
        s[SZpsm] = ctx.zbuf.psm;
        s[SZmask] = ctx.zbuf.zmask ? 1u : 0u;
        s[SScissorX] = static_cast<uint32_t>(ctx.scissor.x0) | (static_cast<uint32_t>(ctx.scissor.x1) << 16u);
        s[SScissorY] = static_cast<uint32_t>(ctx.scissor.y0) | (static_cast<uint32_t>(ctx.scissor.y1) << 16u);
        const uint32_t flags = (state.prim.iip ? FIip : 0u) | (state.prim.tme ? FTme : 0u) | (state.prim.fge ? FFge : 0u) |
                               (state.prim.abe ? FAbe : 0u) | (state.prim.fst ? FFst : 0u) | (state.pabe ? FPabe : 0u) |
                               (state.linearFilter ? FLinear : 0u) | ((state.colclamp & 1u) == 0u ? FWrap : 0u);
        s[SFlags] = flags;
        if (state.prim.tme)
        {
            s[STbp0] = ctx.tex0.tbp0;
            s[STbw] = ctx.tex0.tbw;
            s[STpsm] = ctx.tex0.psm;
            s[STexMode] = static_cast<uint32_t>(ctx.tex0.tcc) | (static_cast<uint32_t>(ctx.tex0.tfx) << 8u) |
                          (static_cast<uint32_t>(ctx.tex0.cpsm) << 16u) | (static_cast<uint32_t>(ctx.tex0.csm) << 24u);
            s[SCsa] = ctx.tex0.csa;
            s[STexSize] = static_cast<uint32_t>(state.textureWidth) | (static_cast<uint32_t>(state.textureHeight) << 16u);
            s[SClampLo] = static_cast<uint32_t>(ctx.clamp);
            s[SClampHi] = static_cast<uint32_t>(ctx.clamp >> 32u);
            s[STexa] = static_cast<uint32_t>(state.texa.ta0) | (state.texa.aem ? 0x100u : 0u) | (static_cast<uint32_t>(state.texa.ta1) << 16u);
            s[SClut] = clutSlot;
        }
        s[SAlpha] = static_cast<uint32_t>(ctx.alpha);
        s[SAlphaFix] = static_cast<uint32_t>(ctx.alpha >> 32u);
        s[STest] = static_cast<uint32_t>(ctx.test);
        s[SFba] = static_cast<uint32_t>(ctx.fba);
        s[SFogCol] = static_cast<uint32_t>(state.fogR) | (static_cast<uint32_t>(state.fogG) << 8u) | (static_cast<uint32_t>(state.fogB) << 16u);
        if (stateCount != 0u && s == lastState)
            return stateCount - 1u;
        lastState = s;
        states.insert(states.end(), s.begin(), s.end());
        return stateCount++;
    }

    bool prepareTarget(const GSDrawState &state, int x0, int y0, int x1, int y1)
    {
        const GSContext &ctx = state.context;
        const uint64_t key = static_cast<uint64_t>(ctx.frame.fbp) | (static_cast<uint64_t>(ctx.frame.fbw) << 9u) |
                             (static_cast<uint64_t>(ctx.frame.psm) << 16u) | (static_cast<uint64_t>(ctx.zbuf.zbp) << 24u) |
                             (static_cast<uint64_t>(ctx.zbuf.psm) << 33u);
        if (hasTarget && key != targetKey)
        {
            ++stats.flushTarget;
            flushBatch();
        }

        std::bitset<kPages> write;
        const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
        markPages(write, ctx.frame.psm, ctx.frame.fbp << 5u, fbw, x0, y0, x1, y1);
        const uint32_t ztest = static_cast<uint32_t>((ctx.test >> 17) & 3u);
        if (!ctx.zbuf.zmask || ztest >= 2u)
            markPages(write, ctx.zbuf.psm, ctx.zbuf.zbp << 5u, fbw, x0, y0, x1, y1);

        std::bitset<kPages> read;
        if (state.prim.tme)
        {
            const uint64_t clamp = ctx.clamp;
            const bool regionU = (clamp & 0x2u) != 0u;
            const bool regionV = ((clamp >> 2) & 0x2u) != 0u;
            const int maxU = regionU ? 1023 : state.textureWidth - 1;
            const int maxV = regionV ? 1023 : state.textureHeight - 1;
            markPages(read, ctx.tex0.psm, ctx.tex0.tbp0, ctx.tex0.tbw, 0, 0, maxU, maxV);
        }
        if ((read & writePages).any() || (write & readPages).any())
        {
            ++stats.flushTexture;
            flushBatch();
        }
        if (clutKeyValid && (write & clutSourcePages).any())
            clutKeyValid = false;
        writePages |= write;
        readPages |= read;
        targetKey = key;
        hasTarget = true;
        return true;
    }

    void bin(uint32_t prim, int x0, int y0, int x1, int y1)
    {
        const uint32_t tx0 = static_cast<uint32_t>(std::clamp(x0, 0, 2047)) / kTileSize;
        const uint32_t tx1 = static_cast<uint32_t>(std::clamp(x1, 0, 2047)) / kTileSize;
        const uint32_t ty0 = static_cast<uint32_t>(std::clamp(y0, 0, 2047)) / kTileSize;
        const uint32_t ty1 = static_cast<uint32_t>(std::clamp(y1, 0, 2047)) / kTileSize;
        for (uint32_t ty = ty0; ty <= ty1; ++ty)
            for (uint32_t tx = tx0; tx <= tx1; ++tx)
            {
                std::vector<uint32_t> &b = bins[ty * kTileGrid + tx];
                if (b.empty())
                    touched.push_back(ty * kTileGrid + tx);
                b.push_back(prim);
            }
    }

    uint32_t *newPrim(uint32_t type, uint32_t stateIndex)
    {
        const size_t offset = prims.size();
        prims.resize(offset + kPrimWords, 0u);
        prims[offset] = type | (stateIndex << 8u);
        ++primCount;
        return prims.data() + offset;
    }

    void addPoint(const GSDrawState &state, int x, int y, uint32_t z, const GSVertex &color, uint8_t fog)
    {
        const GSContext &ctx = state.context;
        if (x < ctx.scissor.x0 || x > ctx.scissor.x1 || y < ctx.scissor.y0 || y > ctx.scissor.y1)
            return;
        prepareTarget(state, x, y, x, y);
        const uint32_t s = addState(state);
        uint32_t *p = newPrim(PrimPoint, s);
        p[1] = static_cast<uint32_t>(x);
        p[2] = static_cast<uint32_t>(y);
        p[3] = z;
        p[4] = packColor(color);
        p[5] = fog;
        bin(primCount - 1u, x, y, x, y);
    }

    void addSprite(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &state = batch.state;
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        const auto &ctx = state.context;
        const GSSpriteAxis axisX = gsSpriteAxis(v0.x, v1.x, static_cast<float>(ctx.xyoffset.ofx) / 16.0f);
        const GSSpriteAxis axisY = gsSpriteAxis(v0.y, v1.y, static_cast<float>(ctx.xyoffset.ofy) / 16.0f);
        const int unclippedX0 = axisX.first;
        const int unclippedY0 = axisY.first;
        const int unclippedX1 = axisX.last;
        const int unclippedY1 = axisY.last;
        if (unclippedX1 < unclippedX0 || unclippedY1 < unclippedY0)
            return;
        if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 || unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
            return;
        const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
        const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
        const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
        const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);

        prepareTarget(state, drawX0, drawY0, drawX1, drawY1);
        const uint32_t s = addState(state);
        uint32_t *p = newPrim(PrimSprite, s);
        p[1] = static_cast<uint32_t>(drawX0) | (static_cast<uint32_t>(drawX1) << 16u);
        p[2] = static_cast<uint32_t>(drawY0) | (static_cast<uint32_t>(drawY1) << 16u);
        p[3] = fbits(axisX.origin);
        p[4] = fbits(axisY.origin);
        p[5] = fbits(axisX.scale);
        p[6] = fbits(axisY.scale);
        if (state.prim.tme)
        {
            float u0f, v0f, u1f, v1f;
            if (state.prim.fst)
            {
                u0f = static_cast<float>(v0.u) / 16.0f;
                v0f = static_cast<float>(v0.v) / 16.0f;
                u1f = static_cast<float>(v1.u) / 16.0f;
                v1f = static_cast<float>(v1.v) / 16.0f;
            }
            else
            {
                const float q0 = fabsQ(v0.q);
                const float q1 = fabsQ(v1.q);
                u0f = (v0.s / q0) * static_cast<float>(state.textureWidth);
                v0f = (v0.t / q0) * static_cast<float>(state.textureHeight);
                u1f = (v1.s / q1) * static_cast<float>(state.textureWidth);
                v1f = (v1.t / q1) * static_cast<float>(state.textureHeight);
            }
            p[7] = fbits(u0f);
            p[8] = fbits(v0f);
            p[9] = fbits(u1f);
            p[10] = fbits(v1f);
        }
        p[13] = static_cast<uint32_t>(v1.z);
        p[14] = packColor(v1);
        p[15] = v1.fog;
        bin(primCount - 1u, drawX0, drawY0, drawX1, drawY1);
    }

    void addTriangle(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &state = batch.state;
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        const GSVertex &v2 = batch.vertices[2];
        const auto &ctx = state.context;
        GSTriangleRaster tri;
        if (!gsTriangleSetup(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, ctx.xyoffset.ofx, ctx.xyoffset.ofy, tri))
            return;
        const int minX = std::max(tri.minX, static_cast<int>(ctx.scissor.x0));
        const int maxX = std::min(tri.maxX, static_cast<int>(ctx.scissor.x1));
        const int minY = std::max(tri.minY, static_cast<int>(ctx.scissor.y0));
        const int maxY = std::min(tri.maxY, static_cast<int>(ctx.scissor.y1));
        if (minX > maxX || minY > maxY)
            return;

        prepareTarget(state, minX, minY, maxX, maxY);
        const uint32_t s = addState(state);
        uint32_t *p = newPrim(PrimTriangle, s);
        p[1] = static_cast<uint32_t>(minX) | (static_cast<uint32_t>(maxX) << 16u);
        p[2] = static_cast<uint32_t>(minY) | (static_cast<uint32_t>(maxY) << 16u);
        for (uint32_t i = 0; i < 3u; ++i)
        {
            p[3u + i * 2u] = static_cast<uint32_t>(static_cast<int32_t>(tri.x[i]));
            p[4u + i * 2u] = static_cast<uint32_t>(static_cast<int32_t>(tri.y[i]));
        }
        p[9] = static_cast<uint32_t>(tri.winding);
        p[10] = fbits(static_cast<float>(tri.invArea));
        std::memcpy(p + 11, &v0.z, 8u);
        std::memcpy(p + 13, &v1.z, 8u);
        std::memcpy(p + 15, &v2.z, 8u);
        p[17] = packColor(v0);
        p[18] = packColor(v1);
        p[19] = packColor(v2);
        const GSVertex *vs[3] = {&v0, &v1, &v2};
        for (uint32_t i = 0; i < 3u; ++i)
        {
            p[20u + i * 3u] = fbits(vs[i]->s);
            p[21u + i * 3u] = fbits(vs[i]->t);
            p[22u + i * 3u] = fbits(vs[i]->q);
            p[29u + i] = static_cast<uint32_t>(vs[i]->u) | (static_cast<uint32_t>(vs[i]->v) << 16u);
        }
        p[32] = static_cast<uint32_t>(v0.fog) | (static_cast<uint32_t>(v1.fog) << 8u) | (static_cast<uint32_t>(v2.fog) << 16u);
        bin(primCount - 1u, minX, minY, maxX, maxY);
    }

    void addLine(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &state = batch.state;
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        const auto &ctx = state.context;
        const int ofx = ctx.xyoffset.ofx >> 4;
        const int ofy = ctx.xyoffset.ofy >> 4;
        int x0 = static_cast<int>(v0.x) - ofx;
        int y0 = static_cast<int>(v0.y) - ofy;
        const int x1 = static_cast<int>(v1.x) - ofx;
        const int y1 = static_cast<int>(v1.y) - ofy;
        const int dx = std::abs(x1 - x0);
        const int dy = -std::abs(y1 - y0);
        const int sx = (x0 < x1) ? 1 : -1;
        const int sy = (y0 < y1) ? 1 : -1;
        int err = dx + dy;
        int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
        if (totalSteps == 0)
            totalSteps = 1;
        int step = 0;
        for (;;)
        {
            const float t = static_cast<float>(step) / static_cast<float>(totalSteps);
            GSVertex color = v1;
            if (state.prim.iip)
            {
                color.r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
                color.g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
                color.b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
                color.a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
            }
            const double z = (v0.z + (v1.z - v0.z) * t);
            const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
            addPoint(state, x0, y0, static_cast<uint32_t>(z), color, fog);
            if (x0 == x1 && y0 == y1)
                break;
            const int e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
            ++step;
        }
    }

    void loadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
    {
        if (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm))
            return;
        switch (tex0.cld)
        {
        case 1u:
            break;
        case 2u:
            clutCbp[0] = tex0.cbp;
            break;
        case 3u:
            clutCbp[1] = tex0.cbp;
            break;
        case 4u:
            if (clutCbp[0] == tex0.cbp)
                return;
            clutCbp[0] = tex0.cbp;
            break;
        case 5u:
            if (clutCbp[1] == tex0.cbp)
                return;
            clutCbp[1] = tex0.cbp;
            break;
        default:
            return;
        }
        const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
        const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
        if (!sixteenBit && !thirtyTwoBit)
            return;
        std::bitset<kPages> source;
        if (tex0.csm == 0u)
            markPages(source, tex0.cpsm, tex0.cbp, 1u, 0, 0, 15, 15);
        else
        {
            const uint32_t width = texclut.cbw != 0u ? texclut.cbw : 1u;
            const int x0 = static_cast<int>(texclut.cou) * 16;
            markPages(source, tex0.cpsm, tex0.cbp, width, x0, texclut.cov, x0 + 255, texclut.cov);
        }
        if ((source & writePages).any() || clutOps.size() + 2u >= kClutSlots)
        {
            ++stats.flushTexture;
            flushBatch();
        }
        const std::array<uint32_t, 6> key = {isFourBitIndexedPsm(tex0.psm) ? 1u : 0u, tex0.cbp,
                                             static_cast<uint32_t>(tex0.cpsm) | (static_cast<uint32_t>(tex0.csm) << 8u) | (static_cast<uint32_t>(tex0.csa) << 16u),
                                             texclut.cbw, texclut.cou, texclut.cov};
        if (clutKeyValid && key == clutKey)
            return;
        clutKey = key;
        clutKeyValid = true;
        clutSourcePages = source;
        if (clutOps.empty())
            clutWriters.fill(0xFFFFu);
        {
            const uint16_t index = static_cast<uint16_t>(clutOps.size());
            const bool fourBit = isFourBitIndexedPsm(tex0.psm);
            const uint32_t count = fourBit ? 16u : 256u;
            const uint32_t destinationBase = (tex0.csa & (sixteenBit ? 0x1Fu : 0x0Fu)) << 4u;
            const bool suffix = tex0.csm == 0u && !sixteenBit && !fourBit;
            for (uint32_t entry = suffix ? destinationBase : 0u; entry < count; entry += 16u)
            {
                if (sixteenBit)
                    clutWriters[((destinationBase + entry) & 0x1FFu) >> 4u] = index;
                else
                {
                    const uint32_t block = ((suffix ? entry : destinationBase + entry) & 0xFFu) >> 4u;
                    clutWriters[block] = index;
                    clutWriters[block + 16u] = index;
                }
            }
        }
        ClutOp op;
        op.writers = clutWriters;
        op.source = clutSlot;
        clutSlot = (clutSlot + 1u) % kClutSlots;
        op.target = clutSlot;
        op.psm = tex0.psm;
        op.cbp = tex0.cbp;
        op.packed = static_cast<uint32_t>(tex0.cpsm) | (static_cast<uint32_t>(tex0.csm) << 8u) | (static_cast<uint32_t>(tex0.csa) << 16u);
        op.cbw = texclut.cbw;
        op.cou = texclut.cou;
        op.cov = texclut.cov;
        clutOps.push_back(op);
    }

    void uploadData(const void *data, size_t size)
    {
        bindUpload(7u, dataBuffer, data, size);
    }

    void reserveData(size_t size)
    {
        if (size > dataCapacity)
        {
            dataCapacity = std::max(size, dataCapacity * 2u);
            gl().BindBuffer(kShaderStorageBuffer, dataBuffer);
            gl().BufferData(kShaderStorageBuffer, static_cast<GLsizeiptr>(dataCapacity), nullptr, kDynamicDraw);
        }
        gl().BindBufferBase(kShaderStorageBuffer, 7u, dataBuffer);
    }

    void dispatchLinear(uint32_t count)
    {
        gl().DispatchCompute((count + 255u) / 256u, 1u, 1u);
        barrier();
    }

    void copyRegion(uint32_t spsm, uint32_t sbp, uint32_t sbw, uint32_t ssax, uint32_t ssay, uint32_t rrw, uint32_t rrh, uint32_t dir)
    {
        reserveData(static_cast<size_t>(rrw) * rrh * 4u);
        gl().UseProgram(copyRead.id);
        gl().Uniform4ui(copyRead.loc[0], sbp, sbw, spsm, ssax | (ssay << 16u));
        gl().Uniform4ui(copyRead.loc[1], rrw, rrh, dir, 0u);
        dispatchLinear(rrw * rrh);
    }

    void localToLocal()
    {
        const uint32_t rrw = transfer.trxreg.rrw;
        const uint32_t rrh = transfer.trxreg.rrh;
        const uint32_t total = rrw * rrh;
        if (total == 0u)
        {
            transferState.direction = 3u;
            return;
        }
        const GSBitBltBuf &b = transfer.bitbltbuf;
        copyRegion(b.spsm, b.sbp, std::max<uint32_t>(b.sbw, 1u), transfer.trxpos.ssax, transfer.trxpos.ssay, rrw, rrh, transfer.trxpos.dir);
        clutKeyValid = false;
        gl().UseProgram(copyWrite.id);
        gl().Uniform4ui(copyWrite.loc[0], b.dbp, std::max<uint32_t>(b.dbw, 1u), b.dpsm,
                        static_cast<uint32_t>(transfer.trxpos.dsax) | (static_cast<uint32_t>(transfer.trxpos.dsay) << 16u));
        gl().Uniform4ui(copyWrite.loc[1], rrw, rrh, transfer.trxpos.dir, 0u);
        dispatchLinear(total);
        transferState.copiedPixels = total;
        transferState.direction = 3u;
    }

    void localToHostTransfer()
    {
        localToHost.clear();
        localToHostReadPos = 0u;
        const uint32_t rrw = transfer.trxreg.rrw;
        const uint32_t rrh = transfer.trxreg.rrh;
        const uint32_t total = rrw * rrh;
        const GSBitBltBuf &b = transfer.bitbltbuf;
        const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(b.spsm)));
        if (total != 0u)
        {
            copyRegion(b.spsm, b.sbp, std::max<uint32_t>(b.sbw, 1u), transfer.trxpos.ssax, transfer.trxpos.ssay, rrw, rrh, 0u);
            scratch.resize(total);
            readData(scratch.data(), static_cast<size_t>(total) * 4u);
        }
        localToHost.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);
        for (uint32_t pixel = 0u; pixel < total; ++pixel)
        {
            const uint32_t value = scratch[pixel];
            switch (bpp)
            {
            case 32:
                localToHost.push_back(static_cast<uint8_t>(value));
                localToHost.push_back(static_cast<uint8_t>(value >> 8u));
                localToHost.push_back(static_cast<uint8_t>(value >> 16u));
                localToHost.push_back(static_cast<uint8_t>(value >> 24u));
                break;
            case 24:
                localToHost.push_back(static_cast<uint8_t>(value));
                localToHost.push_back(static_cast<uint8_t>(value >> 8u));
                localToHost.push_back(static_cast<uint8_t>(value >> 16u));
                break;
            case 16:
                localToHost.push_back(static_cast<uint8_t>(value));
                localToHost.push_back(static_cast<uint8_t>(value >> 8u));
                break;
            case 8:
                localToHost.push_back(static_cast<uint8_t>(value));
                break;
            case 4:
                if ((pixel & 1u) == 0u)
                {
                    const uint32_t next = pixel + 1u < total ? scratch[pixel + 1u] : 0u;
                    localToHost.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
                }
                break;
            default:
                break;
            }
        }
        transferState.copiedPixels = total;
        transferState.localToHostPendingBytes = localToHost.size();
    }

    void uploadImage(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes == 0u || transferState.direction != 0u)
            return;
        if (transfer.trxreg.rrw == 0u || transfer.trxreg.rrh == 0u || transferState.totalPixels == 0u)
            return;
        const uint8_t dpsm = transfer.bitbltbuf.dpsm;
        uint32_t bitsPer = 0u;
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
            bitsPer = 32u;
            break;
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            bitsPer = 24u;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            bitsPer = 16u;
            break;
        case GS_PSM_T8:
        case GS_PSM_T8H:
            bitsPer = 8u;
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            bitsPer = 4u;
            break;
        default:
            return;
        }
        const uint32_t remaining = transferState.totalPixels - transferState.copiedPixels;
        const uint32_t available = static_cast<uint32_t>((static_cast<uint64_t>(sizeBytes) * 8u) / bitsPer);
        const uint32_t count = std::min(remaining, available);
        if (count == 0u)
            return;
        const size_t bytes = (static_cast<size_t>(count) * bitsPer + 7u) / 8u;
        scratch.assign((bytes + 3u) / 4u, 0u);
        std::memcpy(scratch.data(), data, std::min<size_t>(bytes, sizeBytes));
        uploadData(scratch.data(), scratch.size() * 4u);
        const uint32_t dbw = std::max<uint32_t>(transfer.bitbltbuf.dbw, 1u);
        clutKeyValid = false;
        gl().UseProgram(upload.id);
        gl().Uniform4ui(upload.loc[0], transfer.bitbltbuf.dbp, dbw, dpsm, transfer.trxreg.rrw);
        gl().Uniform4ui(upload.loc[1], transferState.copiedPixels, count, transfer.trxpos.dsax, transfer.trxpos.dsay);
        dispatchLinear(count);

        transferState.copiedPixels = std::min<uint32_t>(transferState.totalPixels, transferState.copiedPixels + count);
        if (transferState.copiedPixels >= transferState.totalPixels)
        {
            transferState.direction = 3u;
            transferState.totalPixels = 0u;
            return;
        }
        transferState.x = transfer.trxpos.dsax + (transferState.copiedPixels % transfer.trxreg.rrw);
        transferState.y = transfer.trxpos.dsay + (transferState.copiedPixels / transfer.trxreg.rrw);
    }

    void harvest(Readback &slot, bool wait)
    {
        if (!slot.fence)
            return;
        const auto waitStart = std::chrono::steady_clock::now();
        const GLenum status = gl().ClientWaitSync(slot.fence, wait ? kSyncFlushCommandsBit : 0u, wait ? 1000000000ull : 0ull);
        if (wait)
            presentStats.wait += std::chrono::steady_clock::now() - waitStart;
        if (status != kAlreadySignaled && status != kConditionSatisfied)
            return;
        gl().DeleteSync(slot.fence);
        slot.fence = nullptr;
        if (slot.query)
        {
            GLuint64 ns = 0u;
            gl().GetQueryObjectui64v(slot.query, kQueryResult, &ns);
            presentStats.gpuNs += ns;
            ++presentStats.gpuFrames;
            slot.query = 0u;
        }
        lastFrame.width = slot.width;
        lastFrame.height = slot.height;
        lastFrame.displayFbp = slot.displayFbp;
        lastFrame.sourceFbp = slot.displayFbp;
        lastFrame.usedPreferred = false;
        lastFrame.pixels.resize(static_cast<size_t>(slot.width) * slot.height * 4u);
        std::memcpy(lastFrame.pixels.data(), slot.mapped, lastFrame.pixels.size());
    }

    PresentationFrame presentAsync(uint32_t width, uint32_t height, uint32_t displayFbp)
    {
        Readback &slot = readbacks[readbackNext];
        harvest(slot, true);
        const size_t bytes = static_cast<size_t>(width) * height * 4u;
        if (slot.capacity < bytes)
        {
            if (slot.buffer)
            {
                gl().BindBuffer(kCopyWriteBuffer, slot.buffer);
                gl().UnmapBuffer(kCopyWriteBuffer);
                gl().DeleteBuffers(1, &slot.buffer);
            }
            gl().GenBuffers(1, &slot.buffer);
            gl().BindBuffer(kCopyWriteBuffer, slot.buffer);
            const GLbitfield flags = kMapReadBit | kMapPersistentBit | kMapCoherentBit;
            gl().BufferStorage(kCopyWriteBuffer, static_cast<GLsizeiptr>(bytes), nullptr, flags | kClientStorageBit);
            slot.mapped = static_cast<const uint8_t *>(gl().MapBufferRange(kCopyWriteBuffer, 0, static_cast<GLsizeiptr>(bytes), flags));
            slot.capacity = bytes;
        }
        barrier();
        gl().BindBuffer(kCopyReadBuffer, dataBuffer);
        gl().BindBuffer(kCopyWriteBuffer, slot.buffer);
        gl().CopyBufferSubData(kCopyReadBuffer, kCopyWriteBuffer, 0, 0, static_cast<GLsizeiptr>(bytes));
        if (presentStats.enabled && presentStats.queryOpen)
        {
            gl().EndQuery(kTimeElapsed);
            slot.query = presentStats.queries[presentStats.queryNext];
            presentStats.queryNext = (presentStats.queryNext + 1u) % static_cast<uint32_t>(presentStats.queries.size());
        }
        slot.fence = gl().FenceSync(kSyncGpuCommandsComplete, 0u);
        const auto flushStart = std::chrono::steady_clock::now();
        gl().Flush();
        presentStats.submit += std::chrono::steady_clock::now() - flushStart;
        slot.width = width;
        slot.height = height;
        slot.displayFbp = displayFbp;
        readbackNext = (readbackNext + 1u) % static_cast<uint32_t>(readbacks.size());
        for (uint32_t i = 0; i < readbacks.size(); ++i)
            harvest(readbacks[(readbackNext + i) % readbacks.size()], false);
        if (presentStats.enabled)
        {
            GLuint &query = presentStats.queries[presentStats.queryNext];
            if (!query)
                gl().GenQueries(1, &query);
            gl().BeginQuery(kTimeElapsed, query);
            presentStats.queryOpen = true;
            reportPresentStats();
        }
        return lastFrame;
    }

    void reportPresentStats()
    {
        PresentStats &s = presentStats;
        ++s.frames;
        const auto now = std::chrono::steady_clock::now();
        const double window = std::chrono::duration<double>(now - s.windowStart).count();
        if (window < 2.0)
            return;
        const double frames = s.frames;
        std::fprintf(stderr,
                     "[gs-gpu] %.1f presents/s, GPU %.2f ms/frame, wait %.2f ms/frame, flush %.2f ms/frame, "
                     "%.1f batches, %.1f clut loads, %.1f transfers per frame\n",
                     frames / window, s.gpuFrames ? s.gpuNs / 1e6 / s.gpuFrames : 0.0,
                     std::chrono::duration<double, std::milli>(s.wait).count() / frames,
                     std::chrono::duration<double, std::milli>(s.submit).count() / frames,
                     (stats.batches - s.batches) / frames, (stats.clutLoads - s.clutLoads) / frames,
                     (stats.flushOther - s.transfers) / frames);
        std::fflush(stderr);
        s.frames = 0u;
        s.gpuNs = 0u;
        s.gpuFrames = 0u;
        s.wait = {};
        s.submit = {};
        s.windowStart = now;
        s.batches = stats.batches;
        s.clutLoads = stats.clutLoads;
        s.transfers = stats.flushOther;
    }

    void writeClut(const std::array<uint16_t, 512> &values)
    {
        clutKeyValid = false;
        std::array<uint32_t, 256> words{};
        for (uint32_t i = 0; i < 256u; ++i)
            words[i] = static_cast<uint32_t>(values[i * 2u]) | (static_cast<uint32_t>(values[i * 2u + 1u]) << 16u);
        gl().BindBuffer(kShaderStorageBuffer, clutBuffer);
        gl().BufferSubData(kShaderStorageBuffer, static_cast<GLintptr>(clutSlot) * 1024, 1024, words.data());
    }

    void readClut(std::array<uint16_t, 512> &values)
    {
        std::array<uint32_t, 256> words{};
        barrier();
        gl().BindBuffer(kShaderStorageBuffer, clutBuffer);
        gl().GetBufferSubData(kShaderStorageBuffer, static_cast<GLintptr>(clutSlot) * 1024, 1024, words.data());
        for (uint32_t i = 0; i < 256u; ++i)
        {
            values[i * 2u] = static_cast<uint16_t>(words[i]);
            values[i * 2u + 1u] = static_cast<uint16_t>(words[i] >> 16u);
        }
    }
};

void GSGpuBackend::SetAsyncPresentDefault(bool enabled)
{
    s_asyncPresent.store(enabled, std::memory_order_relaxed);
}

GSGpuBackend::GSGpuBackend()
    : m_impl(std::make_unique<Impl>(m_stats))
{
}

GSGpuBackend::~GSGpuBackend() = default;

void GSGpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    Impl &d = *m_impl;
    d.hostVram = vram;
    d.hostVramSize = vramSize;
    d.resetBatch();
    Reset();
    if (d.ensure())
        d.uploadHostVram();
}

void GSGpuBackend::Reset()
{
    Impl &d = *m_impl;
    d.flushBatch();
    d.clutCbp.fill(0u);
    d.transfer = {};
    d.transfer.direction = 3u;
    d.transferState = {};
    d.transferState.direction = 3u;
    d.localToHost.clear();
    d.localToHostReadPos = 0u;
    if (d.ensure())
    {
        std::array<uint16_t, 512> zeros{};
        d.writeClut(zeros);
    }
}

void GSGpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    Impl &d = *m_impl;
    if (batch.vertexCount == 0u || !d.ensure())
        return;
    switch (batch.state.prim.type)
    {
    case GS_PRIM_SPRITE:
        d.addSprite(batch);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        d.addTriangle(batch);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        d.addLine(batch);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = batch.state.context;
        d.addPoint(batch.state, static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4), static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4),
                   static_cast<uint32_t>(v.z), v, v.fog);
        break;
    }
    default:
        break;
    }
}

void GSGpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    Impl &d = *m_impl;
    if (d.ensure())
        d.loadClut(tex0, texclut);
}

void GSGpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    Impl &d = *m_impl;
    if (!d.ensure())
        return;
    ++m_stats.flushOther;
    d.flushBatch();
    d.transfer = command;
    d.transferState.x = command.trxpos.dsax;
    d.transferState.y = command.trxpos.dsay;
    d.transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    d.transferState.copiedPixels = 0u;
    d.transferState.direction = command.direction;
    d.transferState.localToHostPendingBytes = 0u;
    if (command.direction == 2u)
        d.localToLocal();
    else if (command.direction == 1u)
        d.localToHostTransfer();
}

void GSGpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    Impl &d = *m_impl;
    if (!d.ensure())
        return;
    d.flushBatch();
    d.uploadImage(data, sizeBytes);
}

void GSGpuBackend::Flush()
{
    m_impl->flushBatch();
}

void GSGpuBackend::TextureFlush()
{
}

void GSGpuBackend::Sync(GSSyncReason reason)
{
    Impl &d = *m_impl;
    if (!d.ensure())
        return;
    d.flushBatch();
    if (reason == GSSyncReason::DebugReadback && d.hostVram)
        d.downloadVram(d.hostVram, d.hostVramSize);
}

PresentationFrame GSGpuBackend::Present(const GSPresentationRequest &request)
{
    Impl &d = *m_impl;
    PresentationFrame result{};
    if (!d.ensure())
        return result;
    d.flushBatch();
    const bool enable1 = (request.pmode & 0x1ull) != 0ull;
    const bool enable2 = (request.pmode & 0x2ull) != 0ull;
    const bool mmod = ((request.pmode >> 5) & 0x1ull) != 0ull;
    const bool slbg = ((request.pmode >> 7) & 0x1ull) != 0ull;
    const uint32_t alp = static_cast<uint32_t>((request.pmode >> 8) & 0xFFull);
    const bool halfHeightBuffer = (request.smode2 & 0x3ull) == 0x3ull;
    const CrtcCircuit circuits[2] = {decodeCircuit(enable1, request.dispfb1, request.display1, halfHeightBuffer),
                                     decodeCircuit(enable2, request.dispfb2, request.display2, halfHeightBuffer)};
    if (!circuits[0].enabled && !circuits[1].enabled)
        return result;
    int32_t x0 = INT32_MAX, y0 = INT32_MAX, x1 = INT32_MIN, y1 = INT32_MIN;
    int32_t hstep = INT32_MAX, vstep = INT32_MAX;
    for (const CrtcCircuit &c : circuits)
    {
        if (!c.enabled)
            continue;
        x0 = std::min(x0, c.dx);
        y0 = std::min(y0, c.dy);
        x1 = std::max(x1, c.dx + c.dw);
        y1 = std::max(y1, c.dy + c.dh);
        hstep = std::min(hstep, c.hdiv);
        vstep = std::min(vstep, c.vdiv);
    }
    const uint32_t width = std::min<uint32_t>(kMaxPresentationWidth, static_cast<uint32_t>((x1 - x0 + hstep - 1) / hstep));
    const uint32_t height = std::min<uint32_t>(kMaxPresentationHeight, static_cast<uint32_t>((y1 - y0 + vstep - 1) / vstep));
    result.width = width;
    result.height = height;
    auto pack = [](const CrtcCircuit &c, std::array<uint32_t, 12> &out)
    {
        out = {c.enabled ? 1u : 0u, c.fbp, c.fbw, c.psm, c.dbx, c.dby, static_cast<uint32_t>(c.dx), static_cast<uint32_t>(c.dy),
               static_cast<uint32_t>(c.dw), static_cast<uint32_t>(c.dh), static_cast<uint32_t>(c.hdiv), static_cast<uint32_t>(c.vdiv)};
    };
    std::array<uint32_t, 12> c1{}, c2{};
    pack(circuits[0], c1);
    pack(circuits[1], c2);
    d.reserveData(static_cast<size_t>(width) * height * 4u);
    const Api &gl = d.gl();
    gl.UseProgram(d.present.id);
    gl.Uniform4ui(d.present.loc[0], width, height, static_cast<uint32_t>(x0 & 0xFFFF) | (static_cast<uint32_t>(y0) << 16u),
                  static_cast<uint32_t>(hstep) | (static_cast<uint32_t>(vstep) << 16u));
    gl.Uniform1uiv(d.present.loc[1], 12, c1.data());
    gl.Uniform1uiv(d.present.loc[2], 12, c2.data());
    gl.Uniform4ui(d.present.loc[3], static_cast<uint32_t>(request.bgcolor & 0xFFFFFFu), (slbg ? 1u : 0u) | (mmod ? 2u : 0u), alp, 0u);
    gl.DispatchCompute((width + 15u) / 16u, (height + 15u) / 16u, 1u);
    if (s_asyncPresent.load(std::memory_order_relaxed))
        return d.presentAsync(width, height, circuits[0].enabled ? circuits[0].fbp : circuits[1].fbp);
    result.pixels.resize(static_cast<size_t>(width) * height * 4u);
    d.readData(result.pixels.data(), result.pixels.size());
    result.displayFbp = circuits[0].enabled ? circuits[0].fbp : circuits[1].fbp;
    result.sourceFbp = result.displayFbp;
    result.usedPreferred = false;
    return result;
}

bool GSGpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    Impl &d = *m_impl;
    if (!d.ensure() || context.frame.fbw == 0u)
        return false;
    const uint8_t psm = context.frame.psm;
    const bool full = psm == GS_PSM_CT32 || psm == GS_PSM_CT24;
    const bool half = psm == GS_PSM_CT16 || psm == GS_PSM_CT16S;
    if (!full && !half)
        return false;
    d.flushBatch();
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && psm != GS_PSM_CT24)
        a |= 0x80u;
    uint32_t source;
    uint32_t mask = context.frame.fbmsk;
    if (full)
        source = static_cast<uint32_t>(r) | (static_cast<uint32_t>(g) << 8u) | (static_cast<uint32_t>(b) << 16u) | (static_cast<uint32_t>(a) << 24u);
    else
    {
        source = ((r >> 3) & 0x1Fu) | (((g >> 3) & 0x1Fu) << 5) | (((b >> 3) & 0x1Fu) << 10) | ((a >= 0x40u) ? 0x8000u : 0u);
        mask &= 0xFFFFu;
    }
    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    const Api &gl = d.gl();
    d.clutKeyValid = false;
    gl.UseProgram(d.clear.id);
    gl.Uniform4ui(d.clear.loc[0], context.frame.fbp << 5u, std::max<uint32_t>(context.frame.fbw, 1u) | (static_cast<uint32_t>(psm) << 16u), mask, source);
    gl.Uniform4ui(d.clear.loc[1], x0, y0, x1, y1);
    gl.DispatchCompute((x1 - x0 + 16u) / 16u, (y1 - y0 + 16u) / 16u, 1u);
    d.barrier();
    return true;
}

uint32_t GSGpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    Impl &d = *m_impl;
    if (!dst || maxBytes == 0u || d.localToHostReadPos >= d.localToHost.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, d.localToHost.size() - d.localToHostReadPos);
    std::memcpy(dst, d.localToHost.data() + d.localToHostReadPos, count);
    d.localToHostReadPos += count;
    d.transferState.localToHostPendingBytes = d.localToHost.size() - d.localToHostReadPos;
    return static_cast<uint32_t>(count);
}

uint32_t GSGpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    Impl &d = *m_impl;
    if (!d.ensure())
        return 0u;
    d.flushBatch();
    d.reserveData(16u);
    const Api &gl = d.gl();
    d.clutKeyValid = false;
    gl.UseProgram(d.poke.id);
    gl.Uniform4ui(d.poke.loc[0], psm, base, bw, (x & 0xFFFFu) | (y << 16u));
    gl.Uniform4ui(d.poke.loc[1], 0u, 0u, 0u, 0u);
    gl.DispatchCompute(1u, 1u, 1u);
    uint32_t value = 0u;
    d.readData(&value, sizeof(value));
    return value;
}

void GSGpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    Impl &d = *m_impl;
    if (!d.ensure())
        return;
    d.flushBatch();
    const Api &gl = d.gl();
    d.clutKeyValid = false;
    gl.UseProgram(d.poke.id);
    gl.Uniform4ui(d.poke.loc[0], psm, base, bw, (x & 0xFFFFu) | (y << 16u));
    gl.Uniform4ui(d.poke.loc[1], value, 1u, 0u, 0u);
    gl.DispatchCompute(1u, 1u, 1u);
    d.barrier();
}

void GSGpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    Impl &d = *m_impl;
    if (!d.ensure())
    {
        out.clear();
        return;
    }
    d.flushBatch();
    out.resize(GSSwizzle::kMemorySize);
    d.downloadVram(out.data(), out.size());
}

GSTransferSnapshot GSGpuBackend::GetTransferSnapshot() const
{
    const Impl &d = *m_impl;
    GSTransferSnapshot result = d.transferState;
    result.localToHostPendingBytes = d.localToHostReadPos < d.localToHost.size() ? d.localToHost.size() - d.localToHostReadPos : 0u;
    return result;
}

bool GSGpuBackend::ExportState(GSBackendState &out)
{
    Impl &d = *m_impl;
    if (!d.ensure())
        return false;
    d.flushBatch();
    d.readClut(out.clut);
    out.clutCbp = d.clutCbp;
    out.cachePageBase = UINT32_MAX;
    out.cacheBytes.fill(0u);
    out.transfer = d.transfer;
    out.transferState = d.transferState;
    out.localToHost = d.localToHost;
    out.localToHostReadPos = d.localToHostReadPos;
    return true;
}

bool GSGpuBackend::ImportState(const GSBackendState &state)
{
    Impl &d = *m_impl;
    if (!d.ensure())
        return false;
    d.flushBatch();
    d.writeClut(state.clut);
    d.clutCbp = state.clutCbp;
    d.transfer = state.transfer;
    d.transferState = state.transferState;
    d.localToHost = state.localToHost;
    d.localToHostReadPos = static_cast<size_t>(state.localToHostReadPos);
    return true;
}
