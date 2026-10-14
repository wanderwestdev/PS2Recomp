#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_backend_state.h"

#include <array>
#include <bitset>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GSGpuBackend final : public GSRasterBackend, public GSBackendStateAccess
{
public:
    GSGpuBackend();
    ~GSGpuBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    bool ExportState(GSBackendState &out) override;
    bool ImportState(const GSBackendState &state) override;

    struct Stats
    {
        uint64_t prims = 0;
        uint64_t batches = 0;
        uint64_t tiles = 0;
        uint64_t clutLoads = 0;
        uint64_t flushTarget = 0;
        uint64_t flushTexture = 0;
        uint64_t flushOther = 0;
    };
    const Stats &GetStats() const { return m_stats; }

    static void SetAsyncPresentDefault(bool enabled);
    // Internal resolution, 1 (native) to 4 times in each direction. A request from any thread takes
    // effect on the GS thread at the next presented frame; InternalScale reports the latest request.
    static void RequestInternalScale(uint32_t scale);
    static uint32_t InternalScale();

    // A textured quad (two triangles with FST UVs) whose UVs cover exactly [u0, u1] x [v0, v1]
    // of the texture at tbp0 is drawn as these pieces instead: each piece is a UV rectangle of
    // the same texture placed dx (1/16 texel) right of the quad's left edge, at the quad's scale,
    // height and colour. Lets a game layer relabel text built from a font atlas. UVs are in
    // 1/16 texel units, as the GS UV register.
    // Remaps can also match by screen rectangle (GS primitive coordinates in pixels, the
    // triangle's centre inside it), and instead of pieces either move every matching triangle by
    // (dx, dy) pixels or hide it. The first matching remap applies.
    struct QuadRemap
    {
        struct Piece
        {
            uint16_t u0 = 0, v0 = 0, u1 = 0, v1 = 0; // v0 > v1 flips the piece vertically
            int32_t dx = 0;
            float dy = 0.0f;      // pixels below the quad's top edge
            uint32_t color = 0u;  // RGBA (r in the low byte) instead of the quad's, when not 0
            float scale = 1.0f;   // size of the piece relative to the quad's texel scale
        };
        enum class Action
        {
            Pieces,
            Move,
            Hide
        };
        uint32_t tbp0 = 0;
        bool matchUv = true;
        uint16_t u0 = 0, v0 = 0, u1 = 0, v1 = 0;
        bool matchScreen = false;
        uint16_t maxUvHeight = 0;  // when not 0, only triangles at most this tall in V (1/16 texel)
        uint8_t tbw = 0;           // when not 0, only this texture buffer width
        float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
        Action action = Action::Pieces;
        float dx = 0.0f, dy = 0.0f;
        std::vector<Piece> pieces;
    };
    static void SetQuadRemaps(std::vector<QuadRemap> remaps);

    // Texels a game layer adds to a texture the game uploads itself (new glyphs in a font atlas):
    // written as a host-to-local transfer once per presented frame, before the first textured
    // triangle that reads tbp0. data is packed in psm's host layout (two texels a byte for PSMT4).
    struct TexturePatch
    {
        uint32_t tbp0 = 0;
        uint32_t dbp = 0;
        uint8_t dbw = 0;
        uint8_t dpsm = 0;
        uint16_t x = 0, y = 0, width = 0, height = 0;
        std::vector<uint8_t> data;
    };
    static void SetTexturePatch(std::shared_ptr<const TexturePatch> patch);
    void SetTexturePageCacheEnabled(bool enabled);
    void SetFastTriangleSetupEnabled(bool enabled);
    void SetHardwareRasterAllowed(bool allowed);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    Stats m_stats;
};
