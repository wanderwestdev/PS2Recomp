#include "gs_gpu_shaders.h"

namespace GSGpuShaders
{
    const char *const kPresentHash = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 7) readonly buffer DataBuffer { uint data[]; };
layout(std430, binding = 10) buffer HashBuffer { uint imageHash[2]; };
uniform uint uPixels;
shared uint sums[256];
shared uint xors[256];
uint mixHash(uint v)
{
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    return v ^ (v >> 16u);
}
void main()
{
    uint lane = gl_LocalInvocationID.x;
    uint i = gl_GlobalInvocationID.x;
    uint value = i < uPixels ? mixHash(data[i] ^ mixHash(i + 1u)) : 0u;
    sums[lane] = value;
    xors[lane] = i < uPixels ? mixHash(value ^ 0x9e3779b9u) : 0u;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            sums[lane] += sums[lane + stride];
            xors[lane] ^= xors[lane + stride];
        }
        barrier();
    }
    if (lane == 0u)
    {
        atomicAdd(imageHash[0], sums[0]);
        atomicXor(imageHash[1], xors[0]);
    }
}
)GLSL";
    const char *const kCommon = R"GLSL(

layout(std430, binding = 0) buffer VramBuffer { uint vram[]; };
layout(std430, binding = 1) readonly buffer SwizzleBuffer { uint swz[]; };
layout(std430, binding = 2) buffer ClutBuffer { uint clut[]; };
layout(std430, binding = 9) readonly buffer PageMapBuffer { uint pageMap[]; };

const uint KIND_INVALID = 0u;
const uint KIND_32 = 1u;
const uint KIND_24 = 2u;
const uint KIND_16 = 3u;
const uint KIND_8 = 4u;
const uint KIND_4 = 5u;

uint fmtOf(uint psm)
{
    switch (psm & 0x3Fu)
    {
    case 0x00u: return 0u;
    case 0x01u: return 1u;
    case 0x02u: return 2u;
    case 0x0Au: return 3u;
    case 0x13u: return 4u;
    case 0x14u: return 5u;
    case 0x1Bu: return 6u;
    case 0x24u: return 7u;
    case 0x2Cu: return 8u;
    case 0x30u: return 9u;
    case 0x31u: return 10u;
    case 0x32u: return 11u;
    case 0x3Au: return 12u;
    }
    return 13u;
}

const uint FMT_SX[14] = uint[14](6u, 6u, 6u, 6u, 7u, 7u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 0u);
const uint FMT_SY[14] = uint[14](5u, 5u, 6u, 6u, 6u, 7u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 0u);
const uint FMT_US[14] = uint[14](5u, 5u, 4u, 4u, 3u, 2u, 5u, 5u, 5u, 5u, 5u, 4u, 4u, 0u);
const uint FMT_BO[14] = uint[14](0u, 0u, 0u, 0u, 0u, 0u, 24u, 24u, 28u, 0u, 0u, 0u, 0u, 0u);
const uint FMT_MASK[14] = uint[14](0x3FFFFCu, 0x3FFFFCu, 0x3FFFFEu, 0x3FFFFEu, 0x3FFFFFu, 0x3FFFFFu, 0x3FFFFFu, 0x3FFFFFu, 0x3FFFFFu, 0x3FFFFCu, 0x3FFFFCu, 0x3FFFFEu, 0x3FFFFEu, 0u);
const uint FMT_KIND[14] = uint[14](KIND_32, KIND_24, KIND_16, KIND_16, KIND_8, KIND_4, KIND_8, KIND_4, KIND_4, KIND_32, KIND_24, KIND_16, KIND_16, KIND_INVALID);
const uint FMT_TABLE[14] = uint[14](TABLE_C32, TABLE_C32, TABLE_C16, TABLE_C16S, TABLE_P8, TABLE_P4, TABLE_C32, TABLE_C32, TABLE_C32, TABLE_Z32, TABLE_Z32, TABLE_Z16, TABLE_Z16S, 0u);

uint swzAt(uint index)
{
    uint word = swz[index >> 1u];
    return (index & 1u) != 0u ? (word >> 16u) : (word & 0xFFFFu);
}

uvec2 locate(uint f, uint base, uint bw, uint x, uint y)
{
    uint sx = FMT_SX[f];
    uint sy = FMT_SY[f];
    uint pagesPerRow = (bw * 64u) >> sx;
    uint page = (base >> 5u) + (y >> sy) * pagesPerRow + (x >> sx);
    uint entry = ((base & 31u) << (sx + sy)) + ((y & ((1u << sy) - 1u)) << sx) + (x & ((1u << sx) - 1u));
    uint pixel = (page << (sx + sy)) + swzAt(FMT_TABLE[f] + entry);
    uint bits = (pixel << FMT_US[f]) + FMT_BO[f];
    return uvec2((bits >> 3u) & FMT_MASK[f], bits & 7u);
}

uint loadAt(uint f, uvec2 loc)
{
    uint word = vram[loc.x >> 2u];
    switch (FMT_KIND[f])
    {
    case KIND_32: return word;
    case KIND_24: return word & 0x00FFFFFFu;
    case KIND_16: return (word >> ((loc.x & 2u) * 8u)) & 0xFFFFu;
    case KIND_8: return (word >> ((loc.x & 3u) * 8u)) & 0xFFu;
    case KIND_4: return (word >> ((loc.x & 3u) * 8u + loc.y)) & 0x0Fu;
    }
    return 0u;
}

void storeAt(uint f, uvec2 loc, uint value)
{
    uint index = loc.x >> 2u;
    switch (FMT_KIND[f])
    {
    case KIND_32:
        vram[index] = value;
        break;
    case KIND_24:
        vram[index] = (vram[index] & 0xFF000000u) | (value & 0x00FFFFFFu);
        break;
    case KIND_16:
    {
        uint shift = (loc.x & 2u) * 8u;
        atomicAnd(vram[index], ~(0xFFFFu << shift));
        atomicOr(vram[index], (value & 0xFFFFu) << shift);
        break;
    }
    case KIND_8:
    {
        uint shift = (loc.x & 3u) * 8u;
        atomicAnd(vram[index], ~(0xFFu << shift));
        atomicOr(vram[index], (value & 0xFFu) << shift);
        break;
    }
    case KIND_4:
    {
        uint shift = (loc.x & 3u) * 8u + loc.y;
        atomicAnd(vram[index], ~(0x0Fu << shift));
        atomicOr(vram[index], (value & 0x0Fu) << shift);
        break;
    }
    }
}

uint readPixel(uint psm, uint base, uint bw, uint x, uint y)
{
    uint f = fmtOf(psm);
    if (FMT_KIND[f] == KIND_INVALID)
        return 0u;
    return loadAt(f, locate(f, base, bw, x, y));
}

void writePixelRaw(uint psm, uint base, uint bw, uint x, uint y, uint value)
{
    uint f = fmtOf(psm);
    if (FMT_KIND[f] == KIND_INVALID)
        return;
    storeAt(f, locate(f, base, bw, x, y), value);
}

uint readPixelEpoch(uint psm, uint base, uint bw, uint x, uint y, uint epoch, uint finalEpoch)
{
    uint f = fmtOf(psm);
    if (FMT_KIND[f] == KIND_INVALID)
        return 0u;
    uvec2 loc = locate(f, base, bw, x, y);
    if (epoch < finalEpoch)
        loc.x = (pageMap[epoch * 512u + (loc.x >> 13u)] << 13u) | (loc.x & 0x1FFFu);
    return loadAt(f, loc);
}

uint bitsPerPixel(uint psm)
{
    switch (psm)
    {
    case 0x02u: case 0x0Au: case 0x32u: case 0x3Au: return 16u;
    case 0x13u: case 0x1Bu: return 8u;
    case 0x14u: case 0x24u: case 0x2Cu: return 4u;
    }
    return 32u;
}

uint clampU8(int v)
{
    return uint(clamp(v, 0, 255));
}

uint rgba5551To8888(uint c)
{
    uint r = (c & 0x1Fu) << 3u;
    uint g = ((c >> 5u) & 0x1Fu) << 3u;
    uint b = ((c >> 10u) & 0x1Fu) << 3u;
    uint a = ((c >> 15u) & 0x01u) << 7u;
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

uint rgba8888To5551(uint c)
{
    uint r = (c & 0xFFu) >> 3u;
    uint g = ((c >> 8u) & 0xFFu) >> 3u;
    uint b = ((c >> 16u) & 0xFFu) >> 3u;
    uint a = ((c >> 24u) & 0xFFu) >> 7u;
    return r | (g << 5u) | (b << 10u) | (a << 15u);
}

uint clutEntry(uint version, uint index)
{
    uint word = clut[version * 256u + (index >> 1u)];
    return (index & 1u) != 0u ? (word >> 16u) : (word & 0xFFFFu);
}

void clutStore(uint version, uint index, uint value)
{
    uint word = version * 256u + (index >> 1u);
    uint shift = (index & 1u) * 16u;
    atomicAnd(clut[word], ~(0xFFFFu << shift));
    atomicOr(clut[word], (value & 0xFFFFu) << shift);
}
)GLSL";

    const char *const kRaster = R"GLSL(
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 3) readonly buffer PrimBuffer { uint prims[]; };
layout(std430, binding = 4) readonly buffer StateBuffer { uint states[]; };
layout(std430, binding = 5) readonly buffer TileBuffer { uvec4 tiles[]; };
layout(std430, binding = 6) readonly buffer TileListBuffer { uint tileList[]; };
uniform uvec4 uRaster;

// A value the whole wave shares (the compiler then reads what it indexes with scalar loads).
#if defined(GL_ARB_shader_ballot) && !defined(NO_WAVE_UNIFORM)
#define WAVE_UNIFORM(v) readFirstInvocationARB(v)
#else
#define WAVE_UNIFORM(v) (v)
#endif

#define RAW_READ(psm, b, w, x, y) readPixel(psm, b, w, x, y)
#define RAW_WRITE(psm, b, w, x, y, v) writePixelRaw(psm, b, w, x, y, v)
#define PIX_X float(x)
#define PIX_Y float(y)
#define SUB_X (x * 16)
#define SUB_Y (y * 16)
#define VSCALE 1
#define AREA_SCALE 1.0

#if defined(HW_TYPE) || defined(NO_PIXEL_CACHE)
#define RPIX(psm, b, w, x, y) RAW_READ(psm, b, w, x, y)
#define WPIX(psm, b, w, x, y, v) RAW_WRITE(psm, b, w, x, y, v)
#else
// A compute thread draws one pixel for the whole batch, and a batch has one frame and one depth
// buffer that it never samples as a texture (the backend flushes on both), so the pixel's frame
// and depth values stay in two register slots and reach memory once, when the thread ends.
// Slots are keyed by psm, base and width.
const uint NO_KEY = 0xFFFFFFFFu;
uint cKey0 = NO_KEY;
uint cKey1 = NO_KEY;
uint cVal0 = 0u;
uint cVal1 = 0u;
bool cDirty0 = false;
bool cDirty1 = false;
uint cX = 0u;
uint cY = 0u;

uint cacheKey(uint psm, uint b, uint w) { return (psm & 0x3Fu) | ((b & 0x3FFFu) << 6u) | ((w & 0x3Fu) << 20u); }
uint keyBase(uint k) { return (k >> 6u) & 0x3FFFu; }

void flushSlot(uint k, uint v, bool dirty)
{
    if (dirty && k != NO_KEY)
        RAW_WRITE(k & 0x3Fu, keyBase(k), (k >> 20u) & 0x3Fu, cX, cY, v);
}

void flushPixelCache()
{
    flushSlot(cKey0, cVal0, cDirty0);
    flushSlot(cKey1, cVal1, cDirty1);
    cKey0 = NO_KEY;
    cKey1 = NO_KEY;
    cDirty0 = false;
    cDirty1 = false;
}

// The slot holding k, or a free or evicted slot for it. A key at the base of another cached key
// (formats sharing memory) writes both slots back first.
int cacheSlot(uint k, uint x, uint y)
{
    if (k == cKey0)
        return 0;
    if (k == cKey1)
        return 1;
    if ((cKey0 != NO_KEY && keyBase(cKey0) == keyBase(k)) || (cKey1 != NO_KEY && keyBase(cKey1) == keyBase(k)))
        flushPixelCache();
    cX = x;
    cY = y;
    if (cKey0 == NO_KEY)
        return 0;
    if (cKey1 == NO_KEY)
        return 1;
    flushSlot(cKey1, cVal1, cDirty1);
    cKey1 = NO_KEY;
    cDirty1 = false;
    return 1;
}

uint cachedRead(uint psm, uint b, uint w, uint x, uint y)
{
    uint k = cacheKey(psm, b, w);
    int slot = cacheSlot(k, x, y);
    if (slot == 0)
    {
        if (cKey0 != k)
        {
            cKey0 = k;
            cVal0 = RAW_READ(psm, b, w, x, y);
            cDirty0 = false;
        }
        return cVal0;
    }
    if (cKey1 != k)
    {
        cKey1 = k;
        cVal1 = RAW_READ(psm, b, w, x, y);
        cDirty1 = false;
    }
    return cVal1;
}

void cachedWrite(uint psm, uint b, uint w, uint x, uint y, uint v)
{
    uint k = cacheKey(psm, b, w);
    int slot = cacheSlot(k, x, y);
    if (slot == 0)
    {
        cKey0 = k;
        cVal0 = v;
        cDirty0 = true;
    }
    else
    {
        cKey1 = k;
        cVal1 = v;
        cDirty1 = true;
    }
}
#define RPIX(psm, b, w, x, y) cachedRead(psm, b, w, x, y)
#define WPIX(psm, b, w, x, y, v) cachedWrite(psm, b, w, x, y, v)
#endif

const uint PRIM_WORDS = 40u;
const uint STATE_WORDS = 32u;

const uint S_FBP = 0u;
const uint S_FBW = 1u;
const uint S_FPSM = 2u;
const uint S_FBMSK = 3u;
const uint S_ZBP = 4u;
const uint S_ZPSM = 5u;
const uint S_ZMASK = 6u;
const uint S_SCISSOR_X = 7u;
const uint S_SCISSOR_Y = 8u;
const uint S_TBP0 = 9u;
const uint S_TBW = 10u;
const uint S_TPSM = 11u;
const uint S_TEXMODE = 12u;
const uint S_CSA = 13u;
const uint S_TEXSIZE = 14u;
const uint S_CLAMP_LO = 15u;
const uint S_CLAMP_HI = 16u;
const uint S_ALPHA = 17u;
const uint S_ALPHA_FIX = 18u;
const uint S_TEST = 19u;
const uint S_FBA = 20u;
const uint S_FLAGS = 21u;
const uint S_TEXA = 22u;
const uint S_FOGCOL = 23u;
const uint S_CLUT = 24u;
const uint S_EPOCH = 25u;
const uint S_MIP = 26u;
const uint S_MIPTBP1_LO = 27u;
const uint S_MIPTBP1_HI = 28u;
const uint S_MIPTBP2_LO = 29u;
const uint S_MIPTBP2_HI = 30u;

const uint F_IIP = 1u;
const uint F_TME = 2u;
const uint F_FGE = 4u;
const uint F_ABE = 8u;
const uint F_FST = 16u;
const uint F_PABE = 32u;
const uint F_LINEAR = 64u;
const uint F_WRAP = 128u;
const uint F_HITEX = 256u;

uint st(uint s, uint i) { return states[s * STATE_WORDS + i]; }
uint pw(uint p, uint i) { return prims[p * PRIM_WORDS + i]; }
float pf(uint p, uint i) { return uintBitsToFloat(prims[p * PRIM_WORDS + i]); }
double pd(uint p, uint i) { return packDouble2x32(uvec2(prims[p * PRIM_WORDS + i], prims[p * PRIM_WORDS + i + 1u])); }

bool passesAlphaTest(uint test, uint alpha)
{
    if ((test & 1u) == 0u)
        return true;
    uint atst = (test >> 1u) & 7u;
    uint aref = (test >> 4u) & 0xFFu;
    switch (atst)
    {
    case 0u: return false;
    case 1u: return true;
    case 2u: return alpha < aref;
    case 3u: return alpha <= aref;
    case 4u: return alpha == aref;
    case 5u: return alpha >= aref;
    case 6u: return alpha > aref;
    case 7u: return alpha != aref;
    }
    return true;
}

uvec3 classifyAlphaTest(uint test, uint alpha, uint framePsm)
{
    if (passesAlphaTest(test, alpha))
        return uvec3(1u, 1u, 1u);
    switch ((test >> 12u) & 3u)
    {
    case 1u: return uvec3(1u, 1u, 0u);
    case 2u: return uvec3(0u, 0u, 1u);
    case 3u:
        if (framePsm == 0u)
            return uvec3(1u, 0u, 0u);
        return uvec3(1u, 1u, 0u);
    }
    return uvec3(0u, 0u, 0u);
}

bool passesDestinationAlphaTest(uint test, uint framePsm, uint raw)
{
    if (((test >> 14u) & 1u) == 0u)
        return true;
    bool datm = ((test >> 15u) & 1u) != 0u;
    if (framePsm == 0u)
        return (((raw >> 31u) & 1u) != 0u) == datm;
    if (framePsm == 2u || framePsm == 10u)
        return (((raw >> 15u) & 1u) != 0u) == datm;
    return true;
}

uint applyTexa(uint s, uint psm, uint texel)
{
    if (psm == 0u)
        return texel;
    uint texa = st(s, S_TEXA);
    uint ta0 = texa & 0xFFu;
    bool aem = ((texa >> 8u) & 1u) != 0u;
    uint ta1 = (texa >> 16u) & 0xFFu;
    bool rgbZero = (texel & 0x00FFFFFFu) == 0u;
    uint a = texel >> 24u;
    if (psm == 1u)
        a = (aem && rgbZero) ? 0u : ta0;
    else if (psm == 2u || psm == 10u)
    {
        if ((a & 0x80u) != 0u)
            a = ta1;
        else
            a = (aem && rgbZero) ? 0u : ta0;
    }
    return (texel & 0x00FFFFFFu) | (a << 24u);
}

int wrapCoordinate(int coordinate, int size, uint mode, uint regionMin, uint regionMax)
{
    switch (mode & 3u)
    {
    case 0u: return int(uint(coordinate) & uint(size - 1));
    case 1u: return clamp(coordinate, 0, size - 1);
    case 2u: return min(max(coordinate, int(regionMin)), int(regionMax));
    case 3u: return int((uint(coordinate) & regionMin) | regionMax);
    }
    return coordinate;
}

bool isFourBit(uint psm) { return psm == 0x14u || psm == 0x24u || psm == 0x2Cu; }
bool isEightBit(uint psm) { return psm == 0x13u || psm == 0x1Bu; }

uint lookupClut(uint s, uint index)
{
    uint mode = st(s, S_TEXMODE);
    uint cpsm = (mode >> 16u) & 0xFFu;
    uint csm = (mode >> 24u) & 0xFFu;
    uint csa = st(s, S_CSA);
    uint sourcePsm = st(s, S_TPSM);
    uint version = st(s, S_CLUT);
    bool sixteenBit = cpsm == 2u || cpsm == 10u;
    uint csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    uint clutBase = (csa & csaMask) << 4u;
    uint sourceIndex = isFourBit(sourcePsm) ? (index & 0x0Fu) : index;
    uint clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBit(sourcePsm))
    {
        uint block = min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }
    if (cpsm == 0u)
        return applyTexa(s, cpsm, clutEntry(version, clutIndex) | (clutEntry(version, clutIndex + 256u) << 16u));
    if (cpsm == 1u)
        return applyTexa(s, cpsm, (clutEntry(version, clutIndex) | (clutEntry(version, clutIndex + 256u) << 16u)) & 0x00FFFFFFu);
    if (sixteenBit)
        return applyTexa(s, cpsm, rgba5551To8888(clutEntry(version, clutIndex)));
    return 0xFFFF00FFu;
}

// A texel of mip level `level` (base tbp, width tbw). smp is unused (kept for the callers).
uint samplePointLevel(uint s, int u, int v, int smp, uint level, uint tbp, uint tbw)
{
    uint size = st(s, S_TEXSIZE);
    int texW = max(int(size & 0xFFFFu) >> level, 1);
    int texH = max(int(size >> 16u) >> level, 1);
    uint lo = st(s, S_CLAMP_LO);
    uint hi = st(s, S_CLAMP_HI);
    uint wrapU = lo & 3u;
    uint wrapV = (lo >> 2u) & 3u;
    uint minU = ((lo >> 4u) & 0x3FFu) >> level;
    uint maxU = ((lo >> 14u) & 0x3FFu) >> level;
    uint minV = ((lo >> 24u) | ((hi & 0x3u) << 8u)) >> level;
    uint maxV = ((hi >> 2u) & 0x3FFu) >> level;
    u = wrapCoordinate(u, texW, wrapU, minU, maxU);
    v = wrapCoordinate(v, texH, wrapV, minV, maxV);
    uint psm = st(s, S_TPSM);
    uint raw = 0u;
    uint f = fmtOf(psm);
    if (FMT_KIND[f] != KIND_INVALID)
    {
        uvec2 loc = locate(f, tbp, tbw, uint(u), uint(v));
        uint epoch = st(s, S_EPOCH);
        if (epoch < uRaster.x)
            loc.x = (pageMap[epoch * 512u + (loc.x >> 13u)] << 13u) | (loc.x & 0x1FFFu);
        raw = loadAt(f, loc);
    }
    switch (psm)
    {
    case 0x00u: case 0x30u: case 0x01u: case 0x31u:
        return applyTexa(s, psm, raw);
    case 0x02u: case 0x0Au: case 0x32u: case 0x3Au:
        return applyTexa(s, psm, rgba5551To8888(raw));
    case 0x13u: case 0x1Bu: case 0x14u: case 0x24u: case 0x2Cu:
        return lookupClut(s, raw & 0xFFu);
    }
    return 0xFFFF00FFu;
}

uint samplePointAt(uint s, int u, int v, int smp)
{
    return samplePointLevel(s, u, v, smp, 0u, st(s, S_TBP0), st(s, S_TBW));
}

uint samplePoint(uint s, int u, int v)
{
    return samplePointAt(s, u, v, -1);
}


uint lerpChannel(uint c00, uint c10, uint c01, uint c11, float fx, float fy)
{
    precise float top = float(c00) + (float(c10) - float(c00)) * fx;
    precise float bottom = float(c01) + (float(c11) - float(c01)) * fx;
    precise float value = top + (bottom - top) * fy;
    return clampU8(int(value >= 0.0 ? floor(value + 0.5) : ceil(value - 0.5)));
}

float fabsQ(float q)
{
    return abs(q) > 1.0e-8 ? q : 1.0;
}

// Level 0 of the texture at texel coordinates texUf, texVf.
uint sampleLevel0(uint s, float texUf, float texVf, bool linear)
{
    if (!linear)
        return samplePoint(s, int(texUf), int(texVf));

    precise float sampleU = texUf - 0.5;
    precise float sampleV = texVf - 0.5;
    int u0 = int(floor(sampleU));
    int v0 = int(floor(sampleV));
    precise float fx = sampleU - float(u0);
    precise float fy = sampleV - float(v0);
    uint c00 = samplePoint(s, u0, v0);
    uint c10 = samplePoint(s, u0 + 1, v0);
    uint c01 = samplePoint(s, u0, v0 + 1);
    uint c11 = samplePoint(s, u0 + 1, v0 + 1);
    uint r = lerpChannel(c00 & 0xFFu, c10 & 0xFFu, c01 & 0xFFu, c11 & 0xFFu, fx, fy);
    uint g = lerpChannel((c00 >> 8u) & 0xFFu, (c10 >> 8u) & 0xFFu, (c01 >> 8u) & 0xFFu, (c11 >> 8u) & 0xFFu, fx, fy);
    uint b = lerpChannel((c00 >> 16u) & 0xFFu, (c10 >> 16u) & 0xFFu, (c01 >> 16u) & 0xFFu, (c11 >> 16u) & 0xFFu, fx, fy);
    uint a = lerpChannel(c00 >> 24u, c10 >> 24u, c01 >> 24u, c11 >> 24u, fx, fy);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

uint lerpTexel(uint c0, uint c1, float f)
{
    uint r = lerpChannel(c0 & 0xFFu, c1 & 0xFFu, 0u, 0u, f, 0.0);
    uint g = lerpChannel((c0 >> 8u) & 0xFFu, (c1 >> 8u) & 0xFFu, 0u, 0u, f, 0.0);
    uint b = lerpChannel((c0 >> 16u) & 0xFFu, (c1 >> 16u) & 0xFFu, 0u, 0u, f, 0.0);
    uint a = lerpChannel(c0 >> 24u, c1 >> 24u, 0u, 0u, f, 0.0);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

// Mip level `level` (1-6) at level-0 texel coordinates texUf, texVf. Levels past 0 are never
// render targets, so they are always read at native resolution.
uint sampleMipLevel(uint s, float texUf, float texVf, uint level, bool linear)
{
    if (level == 0u)
        return sampleLevel0(s, texUf, texVf, linear);
    uint shift = (level <= 3u ? level - 1u : level - 4u) * 20u;
    uint lo = st(s, level <= 3u ? S_MIPTBP1_LO : S_MIPTBP2_LO);
    uint hi = st(s, level <= 3u ? S_MIPTBP1_HI : S_MIPTBP2_HI);
    uint packed = shift == 0u ? lo : shift == 20u ? ((lo >> 20u) | (hi << 12u)) : (hi >> 8u);
    uint tbp = packed & 0x3FFFu;
    uint tbw = max((packed >> 14u) & 0x3Fu, 1u);
    precise float scale = 1.0 / float(1u << level);
    precise float lu = texUf * scale;
    precise float lv = texVf * scale;
    if (!linear)
        return samplePointLevel(s, int(lu), int(lv), -1, level, tbp, tbw);
    precise float sampleU = lu - 0.5;
    precise float sampleV = lv - 0.5;
    int u0 = int(floor(sampleU));
    int v0 = int(floor(sampleV));
    precise float fx = sampleU - float(u0);
    precise float fy = sampleV - float(v0);
    uint c00 = samplePointLevel(s, u0, v0, -1, level, tbp, tbw);
    uint c10 = samplePointLevel(s, u0 + 1, v0, -1, level, tbp, tbw);
    uint c01 = samplePointLevel(s, u0, v0 + 1, -1, level, tbp, tbw);
    uint c11 = samplePointLevel(s, u0 + 1, v0 + 1, -1, level, tbp, tbw);
    uint r = lerpChannel(c00 & 0xFFu, c10 & 0xFFu, c01 & 0xFFu, c11 & 0xFFu, fx, fy);
    uint g = lerpChannel((c00 >> 8u) & 0xFFu, (c10 >> 8u) & 0xFFu, (c01 >> 8u) & 0xFFu, (c11 >> 8u) & 0xFFu, fx, fy);
    uint b = lerpChannel((c00 >> 16u) & 0xFFu, (c10 >> 16u) & 0xFFu, (c01 >> 16u) & 0xFFu, (c11 >> 16u) & 0xFFu, fx, fy);
    uint a = lerpChannel(c00 >> 24u, c10 >> 24u, c01 >> 24u, c11 >> 24u, fx, fy);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

uint sampleTexture(uint s, float ss, float tt, float q, uint u, uint v)
{
    uint size = st(s, S_TEXSIZE);
    float texW = float(size & 0xFFFFu);
    float texH = float(size >> 16u);
    uint flags = st(s, S_FLAGS);
    precise float texUf;
    precise float texVf;
    if ((flags & F_FST) != 0u)
    {
        texUf = float(u) / 16.0;
        texVf = float(v) / 16.0;
    }
    else
    {
        precise float invQ = 1.0 / fabsQ(q);
        texUf = ss * invQ * texW;
        texVf = tt * invQ * texH;
    }
    uint mip = st(s, S_MIP);
    if (mip == 0u)
        return sampleLevel0(s, texUf, texVf, (flags & F_LINEAR) != 0u);
    uint mxl = (mip >> 2u) & 7u;
    // GS LOD: K alone (LCM = 1), or (log2(1/|Q|) << L) + K; K is signed 7.4 fixed point.
    precise float lod = float(int(mip << 4u) >> 20) / 16.0;
    if ((mip & 1u) == 0u && (flags & F_FST) == 0u)
        lod += -log2(abs(fabsQ(q))) * float(1u << ((mip >> 9u) & 3u));
    if (lod <= 0.0)
        return sampleLevel0(s, texUf, texVf, ((mip >> 5u) & 1u) != 0u);
    uint mmin = (mip >> 6u) & 7u;
    bool linear = mmin == 1u || mmin == 4u || mmin == 5u;
    if (mmin < 2u || mmin > 5u || mxl == 0u)
        return sampleLevel0(s, texUf, texVf, linear);
    lod = min(lod, float(mxl));
    if (mmin == 3u || mmin == 5u)
    {
        uint level = uint(floor(lod));
        precise float frac = lod - float(level);
        uint c0 = sampleMipLevel(s, texUf, texVf, level, linear);
        if (level >= mxl || frac <= 0.0)
            return c0;
        return lerpTexel(c0, sampleMipLevel(s, texUf, texVf, level + 1u, linear), frac);
    }
    return sampleMipLevel(s, texUf, texVf, min(uint(floor(lod + 0.5)), mxl), linear);
}

uvec4 combineTexture(uint s, uvec4 vc, uint texel)
{
    uint mode = st(s, S_TEXMODE);
    bool tcc = (mode & 0xFFu) != 0u;
    uint tfx = (mode >> 8u) & 0xFFu;
    int tr = int(texel & 0xFFu);
    int tg = int((texel >> 8u) & 0xFFu);
    int tb = int((texel >> 16u) & 0xFFu);
    int ta = int(texel >> 24u);
    int vr = int(vc.r);
    int vg = int(vc.g);
    int vb = int(vc.b);
    int va = int(vc.a);
    switch (tfx)
    {
    case 0u:
        return uvec4(clampU8((tr * vr) >> 7), clampU8((tg * vg) >> 7), clampU8((tb * vb) >> 7), tcc ? clampU8((ta * va) >> 7) : uint(va));
    case 2u:
        return uvec4(clampU8(((tr * vr) >> 7) + va), clampU8(((tg * vg) >> 7) + va), clampU8(((tb * vb) >> 7) + va), tcc ? clampU8(ta + va) : uint(va));
    case 3u:
        return uvec4(clampU8(((tr * vr) >> 7) + va), clampU8(((tg * vg) >> 7) + va), clampU8(((tb * vb) >> 7) + va), tcc ? uint(ta) : uint(va));
    }
    return uvec4(uint(tr), uint(tg), uint(tb), tcc ? uint(ta) : uint(va));
}

int pickRgb(uint sel, int cs, int cd)
{
    if (sel == 0u)
        return cs;
    if (sel == 1u)
        return cd;
    return 0;
}

)GLSL" R"GLSL(
void writePixel(uint s, int x, int y, uint z, uvec4 c, uint fog)
{
    uint sx = st(s, S_SCISSOR_X);
    uint sy = st(s, S_SCISSOR_Y);
    if (x < int(sx & 0xFFFFu) || x > int(sx >> 16u) || y < int(sy & 0xFFFFu) || y > int(sy >> 16u))
        return;
    uint flags = st(s, S_FLAGS);
    int r = int(c.r);
    int g = int(c.g);
    int b = int(c.b);
    int a = int(c.a);
    if ((flags & F_FGE) != 0u)
    {
        uint fogColor = st(s, S_FOGCOL);
        uint inverseFog = 255u - fog;
        r = int(((fog * uint(r)) >> 8u) + ((inverseFog * (fogColor & 0xFFu)) >> 8u)) & 0xFF;
        g = int(((fog * uint(g)) >> 8u) + ((inverseFog * ((fogColor >> 8u) & 0xFFu)) >> 8u)) & 0xFF;
        b = int(((fog * uint(b)) >> 8u) + ((inverseFog * ((fogColor >> 16u) & 0xFFu)) >> 8u)) & 0xFF;
    }

    uint fbp = st(s, S_FBP);
    uint fbw = st(s, S_FBW);
    uint fpsm = st(s, S_FPSM);
    uint fbmsk = st(s, S_FBMSK);
    uint zbp = st(s, S_ZBP);
    uint zpsm = st(s, S_ZPSM);
    uint test = st(s, S_TEST);

    uvec3 mask = classifyAlphaTest(test, uint(a), fpsm);
    bool writeRgb = mask.x != 0u;
    bool writeAlpha = mask.y != 0u;
    bool writeDepth = mask.z != 0u;
    bool writesFramebuffer = writeRgb || writeAlpha;
    if (!writesFramebuffer && !writeDepth)
        return;

    uint ztestMethod = (test >> 17u) & 3u;
    bool abe = (flags & F_ABE) != 0u;
    bool preserveDestinationAlpha = writeRgb && !writeAlpha && fpsm == 0u;
    bool destinationAlphaTestNeedsRead = ((test >> 14u) & 1u) != 0u && (fpsm == 0u || fpsm == 2u || fpsm == 10u);
    bool frmw = destinationAlphaTestNeedsRead || (writesFramebuffer && (fbmsk != 0u || abe || preserveDestinationAlpha));

    uint raw = 0u;
    uint fbrgba = 0u;
    if (frmw)
    {
        raw = RPIX(fpsm, fbp, fbw, uint(x), uint(y));
        fbrgba = raw;
        if (bitsPerPixel(fpsm) == 16u)
            fbrgba = rgba5551To8888(fbrgba);
        else if (fpsm == 1u)
            fbrgba |= 0x80000000u;
    }
    if (!passesDestinationAlphaTest(test, fpsm, raw))
        return;

    bool zpass = false;
    if (ztestMethod == 1u)
        zpass = true;
    else if (ztestMethod == 2u)
        zpass = z >= RPIX(zpsm, zbp, fbw, uint(x), uint(y));
    else if (ztestMethod == 3u)
        zpass = z > RPIX(zpsm, zbp, fbw, uint(x), uint(y));
    if (!zpass)
        return;

    if (writesFramebuffer)
    {
        if (abe && !((flags & F_PABE) != 0u && (a & 0x80) == 0))
        {
            int dr = int(fbrgba & 0xFFu);
            int dg = int((fbrgba >> 8u) & 0xFFu);
            int db = int((fbrgba >> 16u) & 0xFFu);
            int da = int(fbrgba >> 24u);
            uint alphaReg = st(s, S_ALPHA);
            uint asel = alphaReg & 3u;
            uint bsel = (alphaReg >> 2u) & 3u;
            uint csel = (alphaReg >> 4u) & 3u;
            uint dsel = (alphaReg >> 6u) & 3u;
            int fix = int(st(s, S_ALPHA_FIX) & 0xFFu);
            int cAlpha = csel == 0u ? a : (csel == 1u ? da : fix);
            int br = ((pickRgb(asel, r, dr) - pickRgb(bsel, r, dr)) * cAlpha >> 7) + pickRgb(dsel, r, dr);
            int bg = ((pickRgb(asel, g, dg) - pickRgb(bsel, g, dg)) * cAlpha >> 7) + pickRgb(dsel, g, dg);
            int bb = ((pickRgb(asel, b, db) - pickRgb(bsel, b, db)) * cAlpha >> 7) + pickRgb(dsel, b, db);
            if ((flags & F_WRAP) != 0u)
            {
                r = br & 0xFF;
                g = bg & 0xFF;
                b = bb & 0xFF;
            }
            else
            {
                r = int(clampU8(br));
                g = int(clampU8(bg));
                b = int(clampU8(bb));
            }
        }
        if (writeAlpha && (st(s, S_FBA) & 1u) != 0u && fpsm != 1u)
            a = a | 0x80;
        uint pixel = uint(r) | (uint(g) << 8u) | (uint(b) << 16u) | (uint(a & 0xFF) << 24u);
        if (fbmsk != 0u)
            pixel = (pixel & ~fbmsk) | (fbrgba & fbmsk);
        if (preserveDestinationAlpha)
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        if (bitsPerPixel(fpsm) == 16u)
            pixel = rgba8888To5551(pixel);
        WPIX(fpsm, fbp, fbw, uint(x), uint(y), pixel);
    }
    if (writeDepth && st(s, S_ZMASK) == 0u)
        WPIX(zpsm, zbp, fbw, uint(x), uint(y), z);
}

uvec4 unpackColor(uint c)
{
    return uvec4(c & 0xFFu, (c >> 8u) & 0xFFu, (c >> 16u) & 0xFFu, c >> 24u);
}

void drawPoint(uint p, uint s, int x, int y)
{
    if (x != int(pw(p, 1u)) || y != int(pw(p, 2u)))
        return;
    writePixel(s, x, y, pw(p, 3u), unpackColor(pw(p, 4u)), pw(p, 5u));
}

void drawSprite(uint p, uint s, int x, int y)
{
    uint xr = pw(p, 1u);
    uint yr = pw(p, 2u);
    int x0 = int(xr & 0xFFFFu);
    int x1 = int(xr >> 16u);
    int y0 = int(yr & 0xFFFFu);
    int y1 = int(yr >> 16u);
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return;
    uint flags = st(s, S_FLAGS);
    uint z = pw(p, 13u);
    uvec4 color = unpackColor(pw(p, 14u));
    uint fog = pw(p, 15u);
    if ((flags & F_TME) == 0u)
    {
        writePixel(s, x, y, z, color, fog);
        return;
    }
    float ox = pf(p, 3u);
    float oy = pf(p, 4u);
    float scaleX = pf(p, 5u);
    float scaleY = pf(p, 6u);
    float u0f = pf(p, 7u);
    float v0f = pf(p, 8u);
    float u1f = pf(p, 9u);
    float v1f = pf(p, 10u);
    precise float ty = (PIX_Y - oy) * scaleY;
    precise float texVf = v0f + (v1f - v0f) * ty;
    precise float tx = (PIX_X - ox) * scaleX;
    precise float texUf = u0f + (u1f - u0f) * tx;
    uint texel;
    if ((flags & F_FST) != 0u)
    {
        uint su = uint(clamp(int(texUf * 16.0 + 0.5), 0, 0xFFFF));
        uint sv = uint(clamp(int(texVf * 16.0 + 0.5), 0, 0xFFFF));
        texel = sampleTexture(s, 0.0, 0.0, 1.0, su, sv);
    }
    else
    {
        uint size = st(s, S_TEXSIZE);
        texel = sampleTexture(s, texUf / float(size & 0xFFFFu), texVf / float(size >> 16u), 1.0, 0u, 0u);
    }
    writePixel(s, x, y, z, combineTexture(s, color, texel), fog);
}

)GLSL" R"GLSL(
void triangleShade(uint p, uint s, vec3 e, out uint zi, out uvec4 color, out uint fog);

// The edge test: inside, or on the edge under the top-left rule.
bool edgeCovers(int ea, int eb, int hi, uint lo)
{
    bool bias = ea > 0 || (ea == 0 && eb > 0);
    bool positive = hi > 0 || (hi == 0 && lo != 0u);
    bool zero = hi == 0 && lo == 0u;
    return positive || (zero && bias);
}

uint triangleDepth(uint p, vec3 e);

void drawTriangle(uint p, uint s, int x, int y)
{
    uint xr = pw(p, 1u);
    uint yr = pw(p, 2u);
    if (x < int(xr & 0xFFFFu) || x > int(xr >> 16u) || y < int(yr & 0xFFFFu) || y > int(yr >> 16u))
        return;
    int vx[3];
    int vy[3];
    for (uint i = 0u; i < 3u; ++i)
    {
        vx[i] = int(pw(p, 3u + i * 2u)) * VSCALE;
        vy[i] = int(pw(p, 4u + i * 2u)) * VSCALE;
    }
    int sgn = int(pw(p, 9u));
    int sx = SUB_X;
    int sy = SUB_Y;
    float e[3];
    for (uint i = 0u; i < 3u; ++i)
    {
        uint a = (i + 1u) % 3u;
        uint b = (i + 2u) % 3u;
        int ea = -(vy[b] - vy[a]) * sgn;
        int eb = (vx[b] - vx[a]) * sgn;
        int hi1, lo1, hi2, lo2;
        imulExtended(ea, sx - vx[a], hi1, lo1);
        imulExtended(eb, sy - vy[a], hi2, lo2);
        uint carry;
        uint lo = uaddCarry(uint(lo1), uint(lo2), carry);
        int hi = hi1 + hi2 + int(carry);
        if (!edgeCovers(ea, eb, hi, lo))
            return;
        e[i] = float(hi) * 4294967296.0 + float(lo);
    }
    uint z;
    uvec4 color;
    uint fog;
    triangleShade(p, s, vec3(e[0], e[1], e[2]), z, color, fog);
    writePixel(s, x, y, z, color, fog);
}

// Depth of a triangle at a pixel, from its three edge values.
uint triangleDepth(uint p, vec3 e)
{
    float invArea = pf(p, 10u) * float(AREA_SCALE);
    float w1 = e[1] * invArea;
    float w2 = e[2] * invArea;
    uint zi;
#ifdef DOUBLE_Z
    double z0 = pd(p, 11u);
    double z1 = pd(p, 13u);
    double z2 = pd(p, 15u);
    precise double z = z0 + (z1 - z0) * double(w1) + (z2 - z0) * double(w2);
    zi = uint(clamp(z + 0.5, 0.0, 4294967295.0));
#else
    precise float dz = pf(p, 34u) + pf(p, 35u) * w1 + pf(p, 36u) * w2;
    float rz = floor(dz + 0.5);
    uint zBase = pw(p, 33u);
    if (rz >= 0.0)
    {
        uint add = uint(min(rz, 4294967040.0));
        zi = zBase + add < zBase ? 0xFFFFFFFFu : zBase + add;
    }
    else
    {
        uint sub = uint(min(-rz, 4294967040.0));
        zi = sub > zBase ? 0u : zBase - sub;
    }
#endif
    return zi;
}

// Depth, color and fog of a triangle at a covered pixel, from its three edge values.
void triangleShade(uint p, uint s, vec3 e, out uint zi, out uvec4 color, out uint fog)
{
    float invArea = pf(p, 10u) * float(AREA_SCALE);
    float w0 = e[0] * invArea;
    float w1 = e[1] * invArea;
    float w2 = e[2] * invArea;

#ifdef DOUBLE_Z
    double z0 = pd(p, 11u);
    double z1 = pd(p, 13u);
    double z2 = pd(p, 15u);
    precise double z = z0 + (z1 - z0) * double(w1) + (z2 - z0) * double(w2);
    zi = uint(clamp(z + 0.5, 0.0, 4294967295.0));
#else
    // Integer base plus float offset (fp64 runs at a small fraction of the float rate).
    precise float dz = pf(p, 34u) + pf(p, 35u) * w1 + pf(p, 36u) * w2;
    float rz = floor(dz + 0.5);
    uint zBase = pw(p, 33u);
    if (rz >= 0.0)
    {
        uint add = uint(min(rz, 4294967040.0));
        zi = zBase + add < zBase ? 0xFFFFFFFFu : zBase + add;
    }
    else
    {
        uint sub = uint(min(-rz, 4294967040.0));
        zi = sub > zBase ? 0u : zBase - sub;
    }
#endif

    uint flags = st(s, S_FLAGS);
    uvec4 c0 = unpackColor(pw(p, 17u));
    uvec4 c1 = unpackColor(pw(p, 18u));
    uvec4 c2 = unpackColor(pw(p, 19u));
    if ((flags & F_IIP) != 0u)
    {
        precise vec4 mixed = vec4(c0) * w0 + vec4(c1) * w1 + vec4(c2) * w2;
        color = uvec4(clampU8(int(mixed.r)), clampU8(int(mixed.g)), clampU8(int(mixed.b)), clampU8(int(mixed.a)));
    }
    else
        color = c2;

    if ((flags & F_TME) != 0u)
    {
        uint texel;
        if ((flags & F_FST) != 0u)
        {
            uint uv0 = pw(p, 29u);
            uint uv1 = pw(p, 30u);
            uint uv2 = pw(p, 31u);
            precise float fu = float(uv0 & 0xFFFFu) * w0 + float(uv1 & 0xFFFFu) * w1 + float(uv2 & 0xFFFFu) * w2;
            precise float fv = float(uv0 >> 16u) * w0 + float(uv1 >> 16u) * w1 + float(uv2 >> 16u) * w2;
            texel = sampleTexture(s, 0.0, 0.0, 1.0, uint(int(fu)) & 0xFFFFu, uint(int(fv)) & 0xFFFFu);
        }
        else
        {
            precise float is = pf(p, 20u) * w0 + pf(p, 23u) * w1 + pf(p, 26u) * w2;
            precise float it = pf(p, 21u) * w0 + pf(p, 24u) * w1 + pf(p, 27u) * w2;
            precise float iq = pf(p, 22u) * w0 + pf(p, 25u) * w1 + pf(p, 28u) * w2;
            texel = sampleTexture(s, is, it, iq, 0u, 0u);
        }
        color = combineTexture(s, color, texel);
    }
    uint fogs = pw(p, 32u);
    precise float ffog = float(fogs & 0xFFu) * w0 + float((fogs >> 8u) & 0xFFu) * w1 + float((fogs >> 16u) & 0xFFu) * w2;
    fog = clampU8(int(ffog));
}

bool coversPixel(uint p, int x, int y)
{
    uint header = pw(p, 0u);
    uint type = header & 0xFFu;
    uint xr = pw(p, 1u);
    uint yr = pw(p, 2u);
    if (type == 0u)
        return x == int(xr) && y == int(yr);
    if (x < int(xr & 0xFFFFu) || x > int(xr >> 16u) || y < int(yr & 0xFFFFu) || y > int(yr >> 16u))
        return false;
    if (type != 2u)
        return true;
    int sgn = int(pw(p, 9u));
    int sx = SUB_X;
    int sy = SUB_Y;
    for (uint i = 0u; i < 3u; ++i)
    {
        uint a = (i + 1u) % 3u;
        uint b = (i + 2u) % 3u;
        int vxa = int(pw(p, 3u + a * 2u)) * VSCALE;
        int vya = int(pw(p, 4u + a * 2u)) * VSCALE;
        int vxb = int(pw(p, 3u + b * 2u)) * VSCALE;
        int vyb = int(pw(p, 4u + b * 2u)) * VSCALE;
        int ea = -(vyb - vya) * sgn;
        int eb = (vxb - vxa) * sgn;
        int hi1, lo1, hi2, lo2;
        imulExtended(ea, sx - vxa, hi1, lo1);
        imulExtended(eb, sy - vya, hi2, lo2);
        uint carry;
        uint lo = uaddCarry(uint(lo1), uint(lo2), carry);
        int hi = hi1 + hi2 + int(carry);
        if (!edgeCovers(ea, eb, hi, lo))
            return false;
    }
    return true;
}

void drawPrim(uint p, int x, int y)
{
    uint header = pw(p, 0u);
    uint s = header >> 8u;
    switch (header & 0xFFu)
    {
    case 0u: drawPoint(p, s, x, y); break;
    case 1u: drawSprite(p, s, x, y); break;
    case 2u: drawTriangle(p, s, x, y); break;
    }
}

void main()
{
    uvec4 tile = tiles[gl_WorkGroupID.x];
    int x = int(tile.x * 16u + gl_LocalInvocationID.x);
    int y = int(tile.y * 16u + gl_LocalInvocationID.y);
    // The whole work group walks the tile's list in order; the primitive is the same for every
    // pixel, so its words and state are scalar loads.
    for (uint i = 0u; i < tile.w; ++i)
    {
        uint p = WAVE_UNIFORM(tileList[tile.z + i]);
        if (coversPixel(p, x, y))
            drawPrim(p, x, y);
    }
#ifndef NO_PIXEL_CACHE
    flushPixelCache();
#endif
}
)GLSL";

    const char *const kPageCopy = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 7) readonly buffer DataBuffer { uint data[]; };

void main()
{
    uint pair = data[gl_WorkGroupID.x];
    uint source = (pair & 0xFFFFu) * 2048u;
    uint target = (pair >> 16u) * 2048u;
    for (uint i = gl_LocalInvocationID.x; i < 2048u; i += 256u)
        vram[target + i] = vram[source + i];
}
)GLSL";

    const char *const kClutLoad = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 8) readonly buffer ClutOpBuffer { uint clutOps[]; };
uniform uvec4 uClutA;
uniform uvec4 uClutB;

uint clutHalf(uint op, uint hw)
{
    uint p = op * 24u;
    bool fourBit = clutOps[p + 1u] != 0u;
    uint cbp = clutOps[p + 2u];
    uint mode = clutOps[p + 3u];
    uint cbw = clutOps[p + 4u];
    uint cou = clutOps[p + 5u];
    uint cov = clutOps[p + 6u];
    uint cpsm = mode & 0xFFu;
    uint csm = (mode >> 8u) & 0xFFu;
    uint csa = (mode >> 16u) & 0xFFu;
    bool sixteenBit = cpsm == 2u || cpsm == 10u;
    uint csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    uint destinationBase = (csa & csaMask) << 4u;
    bool loadCsm1Suffix = csm == 0u && !sixteenBit && !fourBit;
    uint entry;
    uint shift = 0u;
    if (sixteenBit)
        entry = (hw - destinationBase) & 0x1FFu;
    else
    {
        uint destination = hw & 0xFFu;
        shift = (hw >> 8u) * 16u;
        entry = loadCsm1Suffix ? destination : ((destination - destinationBase) & 0xFFu);
    }
    uint sourceX;
    uint sourceY;
    uint sourceWidth = 1u;
    if (csm == 0u)
    {
        uint sourceIndex = (entry & ~0x18u) | ((entry & 0x08u) << 1u) | ((entry & 0x10u) >> 1u);
        sourceX = sourceIndex & 0x0Fu;
        sourceY = sourceIndex >> 4u;
    }
    else
    {
        sourceWidth = cbw != 0u ? cbw : 1u;
        sourceX = (cou << 4u) + entry;
        sourceY = cov;
    }
    uint raw = readPixelEpoch(cpsm, cbp, sourceWidth, sourceX, sourceY, clutOps[p + 7u], uClutA.y);
    return (raw >> shift) & 0xFFFFu;
}

void main()
{
    uint op = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;
    uint p = op * 24u;
    uint target = clutOps[p];
    uint word = 0u;
    for (uint part = 0u; part < 2u; ++part)
    {
        uint hw = lid * 2u + part;
        uint block = hw >> 4u;
        uint writer = (clutOps[p + 8u + (block >> 1u)] >> ((block & 1u) * 16u)) & 0xFFFFu;
        uint value = writer == 0xFFFFu ? ((clut[uClutA.x * 256u + lid] >> (part * 16u)) & 0xFFFFu) : clutHalf(writer, hw);
        word |= value << (part * 16u);
    }
    clut[target * 256u + lid] = word;
}
)GLSL";

    const char *const kUpload = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 7) readonly buffer DataBuffer { uint data[]; };
uniform uvec4 uUpload;

uint dataByte(uint base, uint index)
{
    return (data[base + (index >> 2u)] >> ((index & 3u) * 8u)) & 0xFFu;
}

void main()
{
    uint entry = data[uUpload.x + gl_WorkGroupID.x];
    uint d = (entry & 0xFFFFu) * 9u;
    uint i = (entry >> 16u) * 256u + gl_LocalInvocationID.x;
    if (i >= data[d + 5u])
        return;
    uint dbp = data[d];
    uint dbw = data[d + 1u];
    uint dpsm = data[d + 2u];
    uint rrw = data[d + 3u];
    uint pixel = data[d + 4u] + i;
    uint x = data[d + 6u] + pixel % rrw;
    uint y = data[d + 7u] + pixel / rrw;
    uint base = data[d + 8u];
    uint value;
    switch (dpsm)
    {
    case 0x00u: case 0x30u:
        value = data[base + i];
        break;
    case 0x01u: case 0x31u:
        value = dataByte(base, i * 3u) | (dataByte(base, i * 3u + 1u) << 8u) | (dataByte(base, i * 3u + 2u) << 16u);
        break;
    case 0x02u: case 0x0Au: case 0x32u: case 0x3Au:
        value = dataByte(base, i * 2u) | (dataByte(base, i * 2u + 1u) << 8u);
        break;
    case 0x13u: case 0x1Bu:
        value = dataByte(base, i);
        break;
    default:
        value = (dataByte(base, i >> 1u) >> ((i & 1u) * 4u)) & 0x0Fu;
        break;
    }
    writePixelRaw(dpsm, dbp, dbw, x, y, value);
}
)GLSL";

    const char *const kCopyRead = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 7) buffer DataBuffer { uint data[]; };
uniform uvec4 uSrc;
uniform uvec4 uSize;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    uint rrw = uSize.x;
    uint rrh = uSize.y;
    if (i >= rrw * rrh)
        return;
    uint x = i % rrw;
    uint y = i / rrw;
    if ((uSize.z & 2u) != 0u)
        x = rrw - x - 1u;
    if ((uSize.z & 1u) != 0u)
        y = rrh - y - 1u;
    data[i] = readPixel(uSrc.z, uSrc.x, uSrc.y, x + (uSrc.w & 0xFFFFu), y + (uSrc.w >> 16u));
}
)GLSL";

    const char *const kCopyWrite = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 7) readonly buffer DataBuffer { uint data[]; };
uniform uvec4 uDst;
uniform uvec4 uSize;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    uint rrw = uSize.x;
    uint rrh = uSize.y;
    if (i >= rrw * rrh)
        return;
    uint x = i % rrw;
    uint y = i / rrw;
    if ((uSize.z & 2u) != 0u)
        x = rrw - x - 1u;
    if ((uSize.z & 1u) != 0u)
        y = rrh - y - 1u;
    writePixelRaw(uDst.z, uDst.x, uDst.y, x + (uDst.w & 0xFFFFu), y + (uDst.w >> 16u), data[i]);
}
)GLSL";

    const char *const kClear = R"GLSL(
layout(local_size_x = 16, local_size_y = 16) in;
uniform uvec4 uFrame;
uniform uvec4 uRect;

void main()
{
    uint x = uRect.x + gl_GlobalInvocationID.x;
    uint y = uRect.y + gl_GlobalInvocationID.y;
    if (x > uRect.z || y > uRect.w)
        return;
    uint fbp = uFrame.x;
    uint fbw = uFrame.y & 0xFFFFu;
    uint psm = uFrame.y >> 16u;
    uint mask = uFrame.z;
    uint source = uFrame.w;
    uint pixel = source;
    if (mask != 0u)
        pixel = (pixel & ~mask) | (readPixel(psm, fbp, fbw, x, y) & mask);
    writePixelRaw(psm, fbp, fbw, x, y, pixel);
}
)GLSL";

    const char *const kPoke = R"GLSL(
layout(local_size_x = 1) in;
layout(std430, binding = 7) buffer DataBuffer { uint data[]; };
uniform uvec4 uArgs;
uniform uvec4 uArgs2;

void main()
{
    if (uArgs2.y != 0u)
        writePixelRaw(uArgs.x, uArgs.y, uArgs.z, uArgs.w & 0xFFFFu, uArgs.w >> 16u, uArgs2.x);
    else
        data[0] = readPixel(uArgs.x, uArgs.y, uArgs.z, uArgs.w & 0xFFFFu, uArgs.w >> 16u);
}
)GLSL";

    const char *const kPresent = R"GLSL(
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 7) buffer DataBuffer { uint data[]; };
uniform uvec4 uOut;
uniform uint uC1[12];
uniform uint uC2[12];
uniform uvec4 uMode;

uint displayToRgba(uint psm, uint raw)
{
    switch (psm & 0x0Fu)
    {
    case 0u: return raw;
    case 1u: return (raw & 0x00FFFFFFu) | 0x80000000u;
    case 2u:
    case 10u:
    {
        uint r = raw & 31u;
        uint g = (raw >> 5u) & 31u;
        uint b = (raw >> 10u) & 31u;
        return ((r << 3u) | (r >> 2u)) | (((g << 3u) | (g >> 2u)) << 8u) | (((b << 3u) | (b >> 2u)) << 16u) |
               ((raw & 0x8000u) != 0u ? 0x80000000u : 0u);
    }
    }
    return raw;
}

bool covers(uint c[12], int rx, int ry)
{
    return c[0] != 0u && rx >= int(c[6]) && rx < int(c[6]) + int(c[8]) && ry >= int(c[7]) && ry < int(c[7]) + int(c[9]);
}

uint readCircuit(uint c[12], int rx, int ry, float frx, float fry)
{
    uint bx = (uint((rx - int(c[6])) / int(c[10])) + c[4]) & 0x7FFu;
    uint by = (uint((ry - int(c[7])) / int(c[11])) + c[5]) & 0x7FFu;
    return displayToRgba(c[3], readPixel(c[3], c[1] * 32u, c[2], bx, by));
}

uint blendChannel(uint src, uint dst, uint factor)
{
    int delta = int(src) - int(dst);
    return clampU8(int(dst) + ((delta * int(factor)) / 255));
}

// The two read circuits merged at display position (rx, ry).
uint composeAt(int rx, int ry, float frx, float fry)
{
    uint c1[12] = uC1;
    uint c2[12] = uC2;
    bool in1 = covers(c1, rx, ry);
    bool in2 = covers(c2, rx, ry);
    uint outPixel = 0xFF000000u;
    if (in1 || in2)
    {
        uint bg = uMode.x;
        uint r = bg & 0xFFu;
        uint g = (bg >> 8u) & 0xFFu;
        uint b = (bg >> 16u) & 0xFFu;
        bool slbg = (uMode.y & 1u) != 0u;
        bool mmod = (uMode.y & 2u) != 0u;
        uint alp = uMode.z;
        if (in2 && !slbg)
        {
            uint color = readCircuit(c2, rx, ry, frx, fry);
            r = color & 0xFFu;
            g = (color >> 8u) & 0xFFu;
            b = (color >> 16u) & 0xFFu;
        }
        if (in1)
        {
            uint color = readCircuit(c1, rx, ry, frx, fry);
            uint factor = mmod ? alp : min(255u, (color >> 24u) * 2u);
            r = blendChannel(color & 0xFFu, r, factor);
            g = blendChannel((color >> 8u) & 0xFFu, g, factor);
            b = blendChannel((color >> 16u) & 0xFFu, b, factor);
        }
        outPixel = r | (g << 8u) | (b << 16u) | 0xFF000000u;
    }
    return outPixel;
}

void main()
{
    uint ox = gl_GlobalInvocationID.x;
    uint oy = gl_GlobalInvocationID.y;
    uint width = uOut.x;
    uint height = uOut.y;
    if (ox >= width || oy >= height)
        return;
    int x0 = int(uOut.z & 0xFFFFu);
    int y0 = int(uOut.z >> 16u);
    int hstep = int(uOut.w & 0xFFFFu);
    int vstep = int(uOut.w >> 16u);
    uint outPixel;
    int ry = y0 + int(oy) * vstep + vstep / 2;
    int rx = x0 + int(ox) * hstep + hstep / 2;
    outPixel = composeAt(rx, ry, float(rx), float(ry));
    data[oy * width + ox] = outPixel;
}
)GLSL";

    const char *const kHwVertex = R"GLSL(
layout(std430, binding = 3) readonly buffer PrimBuffer { uint prims[]; };
flat out uint vPrim;
uniform uvec4 uHw;

void main()
{
    uint p = uint(gl_VertexID) / 6u;
    uint c = uint(gl_VertexID) % 6u;
    uint header = prims[p * 40u];
    vPrim = p;
    if (uHw.x != 0u && (header & 0xFFu) == 2u)
    {
        vec2 v[3];
        for (uint i = 0u; i < 3u; ++i)
            v[i] = vec2(float(int(prims[p * 40u + 3u + i * 2u])), float(int(prims[p * 40u + 4u + i * 2u]))) / 16.0 + 0.5;
        vec2 e0 = v[1] - v[0];
        vec2 e1 = v[2] - v[0];
        float orient = e0.x * e1.y - e0.y * e1.x >= 0.0 ? 1.0 : -1.0;
        vec2 n[3];
        bool usable = true;
        for (uint i = 0u; i < 3u; ++i)
        {
            vec2 d = v[(i + 1u) % 3u] - v[i];
            float len = length(d);
            usable = usable && len > 1.0e-3;
            n[i] = len > 1.0e-3 ? vec2(d.y, -d.x) * (orient / len) : vec2(0.0);
        }
        vec2 corner[3];
        for (uint i = 0u; i < 3u; ++i)
        {
            vec2 a = n[(i + 2u) % 3u];
            vec2 b = n[i];
            float denom = 1.0 + dot(a, b);
            usable = usable && denom > 1.0e-4;
            corner[i] = v[i] + (a + b) * (0.75 / max(denom, 1.0e-4));
        }
        if (usable)
        {
            vec2 q = c < 3u ? corner[c] : corner[2];
            gl_Position = vec4(q.x / 1024.0 - 1.0, q.y / 1024.0 - 1.0, 0.0, 1.0);
            return;
        }
    }
    uint xr = prims[p * 40u + 1u];
    uint yr = prims[p * 40u + 2u];
    float x0;
    float x1;
    float y0;
    float y1;
    if ((header & 0xFFu) == 0u)
    {
        x0 = float(xr);
        x1 = x0 + 1.0;
        y0 = float(yr);
        y1 = y0 + 1.0;
    }
    else
    {
        x0 = float(xr & 0xFFFFu);
        x1 = float(xr >> 16u) + 1.0;
        y0 = float(yr & 0xFFFFu);
        y1 = float(yr >> 16u) + 1.0;
    }
    bool right = c == 1u || c == 3u || c == 4u;
    bool bottom = c == 2u || c == 4u || c == 5u;
    gl_Position = vec4((right ? x1 : x0) / 1024.0 - 1.0, (bottom ? y1 : y0) / 1024.0 - 1.0, 0.0, 1.0);
    vPrim = p;
}
)GLSL";

    const char *const kHwFragmentMain = R"GLSL(
layout(pixel_interlock_ordered) in;
flat in uint vPrim;

void main()
{
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);
    uint s;
    uint z;
    uvec4 c;
    uint fog;
    bool shaded = coversPixel(vPrim, x, y) && shadePrim(vPrim, x, y, s, z, c, fog);
    beginInvocationInterlockARB();
    if (shaded)
        writePixel(s, x, y, z, c, fog);
    endInvocationInterlockARB();
}
)GLSL";

    // Render targets (PS2X_GS_TARGETS): the fragment shader shades exactly as the compute path and
    // leaves the depth test, blending and write masks to the fixed-function stages, which keep
    // primitive order without an interlock. uTarget.x selects the pixels that pass the alpha test
    // (0) or the ones that fail it (1, drawn with the AFAIL write masks).
    const char *const kTargetFragmentMain = R"GLSL(
flat in uint vPrim;
layout(location = 0, index = 0) out vec4 oColor;
layout(location = 0, index = 1) out vec4 oFactor;
// uTarget.y = 1: exact GS blending in the shader against uDest, the frame target itself; the
// backend draws such runs so that no pixel is covered twice between texture barriers.
layout(binding = 1) uniform sampler2D uDest;
uniform uvec4 uTarget;

uint targetBlendExact(uint s, uint flags, int r, int g, int b, int a, uint fbrgba)
{
    if ((flags & F_ABE) != 0u && !((flags & F_PABE) != 0u && (a & 0x80) == 0))
    {
        int dr = int(fbrgba & 0xFFu);
        int dg = int((fbrgba >> 8u) & 0xFFu);
        int db = int((fbrgba >> 16u) & 0xFFu);
        int da = int(fbrgba >> 24u);
        uint alphaReg = st(s, S_ALPHA);
        uint asel = alphaReg & 3u;
        uint bsel = (alphaReg >> 2u) & 3u;
        uint csel = (alphaReg >> 4u) & 3u;
        uint dsel = (alphaReg >> 6u) & 3u;
        int fix = int(st(s, S_ALPHA_FIX) & 0xFFu);
        int cAlpha = csel == 0u ? a : (csel == 1u ? da : fix);
        int br = ((pickRgb(asel, r, dr) - pickRgb(bsel, r, dr)) * cAlpha >> 7) + pickRgb(dsel, r, dr);
        int bg = ((pickRgb(asel, g, dg) - pickRgb(bsel, g, dg)) * cAlpha >> 7) + pickRgb(dsel, g, dg);
        int bb = ((pickRgb(asel, b, db) - pickRgb(bsel, b, db)) * cAlpha >> 7) + pickRgb(dsel, b, db);
        if ((flags & F_WRAP) != 0u)
        {
            r = br & 0xFF;
            g = bg & 0xFF;
            b = bb & 0xFF;
        }
        else
        {
            r = int(clampU8(br));
            g = int(clampU8(bg));
            b = int(clampU8(bb));
        }
    }
    if ((st(s, S_FBA) & 1u) != 0u && st(s, S_FPSM) != 1u)
        a = a | 0x80;
    uint pixel = uint(r) | (uint(g) << 8u) | (uint(b) << 16u) | (uint(a & 0xFF) << 24u);
    uint fbmsk = st(s, S_FBMSK);
    if (fbmsk != 0u)
        pixel = (pixel & ~fbmsk) | (fbrgba & fbmsk);
    return pixel;
}

noperspective in vec4 vColor;
flat in uint vColorFlat;
noperspective in vec3 vStq;
noperspective in vec2 vUv;
noperspective in float vFog;
noperspective in float vDz;

void main()
{
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);
    // The rasterizer decides coverage; the attributes arrive interpolated (kTargetVertex).
    uint p = vPrim;
    uint header = pw(p, 0u);
    uint s = header >> 8u;
    uint type = header & 0xFFu;
    uint flags = st(s, S_FLAGS);
    uint z;
    uvec4 c;
    uint fog;
    if (type == 2u)
    {
        float dz = floor(vDz + 0.5);
        uint zBase = pw(p, 33u);
        if (dz >= 0.0)
        {
            uint add = uint(min(dz, 4294967040.0));
            z = zBase + add < zBase ? 0xFFFFFFFFu : zBase + add;
        }
        else
        {
            uint sub = uint(min(-dz, 4294967040.0));
            z = sub > zBase ? 0u : zBase - sub;
        }
        c = (flags & F_IIP) != 0u ? uvec4(clampU8(int(vColor.r)), clampU8(int(vColor.g)), clampU8(int(vColor.b)), clampU8(int(vColor.a)))
                                  : unpackColor(vColorFlat);
        fog = clampU8(int(vFog));
        if ((flags & F_TME) != 0u)
        {
            uint texel;
            if ((flags & F_FST) != 0u)
                texel = sampleTexture(s, 0.0, 0.0, 1.0, uint(int(vUv.x)) & 0xFFFFu, uint(int(vUv.y)) & 0xFFFFu);
            else
                texel = sampleTexture(s, vStq.x, vStq.y, vStq.z, 0u, 0u);
            c = combineTexture(s, c, texel);
        }
    }
    else if (type == 1u)
    {
        z = pw(p, 13u);
        c = unpackColor(pw(p, 14u));
        fog = pw(p, 15u);
        if ((flags & F_TME) != 0u)
        {
            uint texel;
            if ((flags & F_FST) != 0u)
                texel = sampleTexture(s, 0.0, 0.0, 1.0, uint(clamp(int(vUv.x * 16.0 + 0.5), 0, 0xFFFF)),
                                      uint(clamp(int(vUv.y * 16.0 + 0.5), 0, 0xFFFF)));
            else
            {
                uint size = st(s, S_TEXSIZE);
                texel = sampleTexture(s, vUv.x / float(size & 0xFFFFu), vUv.y / float(size >> 16u), 1.0, 0u, 0u);
            }
            c = combineTexture(s, c, texel);
        }
    }
    else
    {
        z = pw(p, 3u);
        c = unpackColor(pw(p, 4u));
        fog = pw(p, 5u);
    }
    uint sx = st(s, S_SCISSOR_X);
    uint sy = st(s, S_SCISSOR_Y);
    if (x < int(sx & 0xFFFFu) || x > int(sx >> 16u) || y < int(sy & 0xFFFFu) || y > int(sy >> 16u))
        discard;
    uint r = c.r;
    uint g = c.g;
    uint b = c.b;
    uint a = c.a;
    if ((flags & F_FGE) != 0u)
    {
        uint fogColor = st(s, S_FOGCOL);
        uint inverseFog = 255u - fog;
        r = (((fog * r) >> 8u) + ((inverseFog * (fogColor & 0xFFu)) >> 8u)) & 0xFFu;
        g = (((fog * g) >> 8u) + ((inverseFog * ((fogColor >> 8u) & 0xFFu)) >> 8u)) & 0xFFu;
        b = (((fog * b) >> 8u) + ((inverseFog * ((fogColor >> 16u) & 0xFFu)) >> 8u)) & 0xFFu;
    }
    if (passesAlphaTest(st(s, S_TEST), a) == (uTarget.x != 0u))
        discard;
    uint zpsmExact = st(s, S_ZPSM) & 0xFu;
    uint zmaxExact = zpsmExact == 1u ? 0xFFFFFFu : (zpsmExact == 0u ? 0xFFFFFFFFu : 0xFFFFu);
    if (uTarget.y != 0u)
    {
        uvec4 d = uvec4(texelFetch(uDest, ivec2(x, y), 0) * 255.0 + 0.5);
        uint fbrgba = d.r | (d.g << 8u) | (d.b << 16u) | (d.a << 24u);
        if (!passesDestinationAlphaTest(st(s, S_TEST), st(s, S_FPSM), fbrgba))
            discard;
        uint pixel = targetBlendExact(s, flags, int(r), int(g), int(b), int(a), fbrgba);
        gl_FragDepth = float(min(z, zmaxExact)) * (1.0 / 16777216.0);
        oColor = vec4(float(pixel & 0xFFu), float((pixel >> 8u) & 0xFFu), float((pixel >> 16u) & 0xFFu), float(pixel >> 24u)) / 255.0;
        oFactor = vec4(1.0);
        return;
    }
    float factor = float(a) / 128.0;
    if ((flags & F_PABE) != 0u && (a & 0x80u) == 0u)
        factor = 1.0;
    if ((st(s, S_FBA) & 1u) != 0u && st(s, S_FPSM) != 1u)
        a |= 0x80u;
    uint zpsm = st(s, S_ZPSM) & 0xFu;
    uint zmax = zpsm == 1u ? 0xFFFFFFu : (zpsm == 0u ? 0xFFFFFFFFu : 0xFFFFu);
    gl_FragDepth = float(min(z, zmax)) * (1.0 / 16777216.0);
    oColor = vec4(float(r), float(g), float(b), float(a & 0xFFu)) / 255.0;
    oFactor = vec4(factor);
}
)GLSL";

    // Six vertices per primitive, pulled from the primitive buffer: a triangle's three (the
    // other three collapse onto the last) or a sprite's or point's two triangles. GS position P
    // lands at P + 0.5 so the pixel centre samples the GS integer position. Attributes are
    // interpolated linearly in screen space, as the GS does.
    const char *const kTargetVertex = R"GLSL(
layout(std430, binding = 3) readonly buffer PrimBuffer { uint prims[]; };
flat out uint vPrim;
noperspective out vec4 vColor;
flat out uint vColorFlat;
noperspective out vec3 vStq;
noperspective out vec2 vUv;
noperspective out float vFog;
noperspective out float vDz;

uint pw(uint p, uint i) { return prims[p * 40u + i]; }
float pf(uint p, uint i) { return uintBitsToFloat(prims[p * 40u + i]); }

void emit(vec2 at)
{
    gl_Position = vec4(at.x / 1024.0 - 1.0, at.y / 1024.0 - 1.0, 0.0, 1.0);
}

void main()
{
    uint p = uint(gl_VertexID) / 6u;
    uint c = uint(gl_VertexID) % 6u;
    uint header = pw(p, 0u);
    uint type = header & 0xFFu;
    vPrim = p;
    vColor = vec4(0.0);
    vColorFlat = 0u;
    vStq = vec3(0.0, 0.0, 1.0);
    vUv = vec2(0.0);
    vFog = 0.0;
    vDz = 0.0;
    if (type == 2u)
    {
        uint i = min(c, 2u);
        uint col = pw(p, 17u + i);
        vColor = vec4(float(col & 0xFFu), float((col >> 8u) & 0xFFu), float((col >> 16u) & 0xFFu), float(col >> 24u));
        vColorFlat = pw(p, 19u);
        vStq = vec3(pf(p, 20u + i * 3u), pf(p, 21u + i * 3u), pf(p, 22u + i * 3u));
        uint uv = pw(p, 29u + i);
        vUv = vec2(float(uv & 0xFFFFu), float(uv >> 16u));
        vFog = float((pw(p, 32u) >> (i * 8u)) & 0xFFu);
        vDz = pf(p, 34u) + (i == 1u ? pf(p, 35u) : (i == 2u ? pf(p, 36u) : 0.0));
        emit(vec2(float(int(pw(p, 3u + i * 2u))), float(int(pw(p, 4u + i * 2u)))) / 16.0 + 0.5);
        return;
    }
    uint xr = pw(p, 1u);
    uint yr = pw(p, 2u);
    float x0;
    float x1;
    float y0;
    float y1;
    if (type == 0u)
    {
        x0 = float(xr);
        x1 = x0 + 1.0;
        y0 = float(yr);
        y1 = y0 + 1.0;
    }
    else
    {
        x0 = float(xr & 0xFFFFu);
        x1 = float(xr >> 16u) + 1.0;
        y0 = float(yr & 0xFFFFu);
        y1 = float(yr >> 16u) + 1.0;
    }
    bool right = c == 1u || c == 3u || c == 4u;
    bool bottom = c == 2u || c == 4u || c == 5u;
    vec2 at = vec2(right ? x1 : x0, bottom ? y1 : y0);
    if (type == 1u)
    {
        // The sprite's texture coordinate is linear in the GS position (drawSprite): evaluate it
        // at this corner, whose GS position is half a pixel left of and above the GL position.
        float tx = (at.x - 0.5 - pf(p, 3u)) * pf(p, 5u);
        float ty = (at.y - 0.5 - pf(p, 4u)) * pf(p, 6u);
        vUv = vec2(pf(p, 7u) + (pf(p, 9u) - pf(p, 7u)) * tx, pf(p, 8u) + (pf(p, 10u) - pf(p, 8u)) * ty);
    }
    emit(at);
}
)GLSL";

    // Draws the listed cells of a target from GS memory: one quad per cell (data: x0 | y0 << 16,
    // x1 | y1 << 16, exclusive ends), positions on the 2048 x 2048 target grid.
    const char *const kTargetLoadVertex = R"GLSL(
layout(std430, binding = 7) readonly buffer DataBuffer { uint data[]; };

void main()
{
    uint cell = uint(gl_VertexID) / 6u;
    uint c = uint(gl_VertexID) % 6u;
    uint lo = data[cell * 2u];
    uint hi = data[cell * 2u + 1u];
    bool right = c == 1u || c == 3u || c == 4u;
    bool bottom = c == 2u || c == 4u || c == 5u;
    float x = float(right ? (hi & 0xFFFFu) : (lo & 0xFFFFu));
    float y = float(bottom ? (hi >> 16u) : (lo >> 16u));
    gl_Position = vec4(x / 1024.0 - 1.0, y / 1024.0 - 1.0, 0.0, 1.0);
}
)GLSL";

    // uLoad: psm, base, buffer width, 1 for a depth target.
    const char *const kTargetLoadFragment = R"GLSL(
uniform uvec4 uLoad;
layout(location = 0) out vec4 oColor;

void main()
{
    uint x = uint(gl_FragCoord.x);
    uint y = uint(gl_FragCoord.y);
    uint v = readPixel(uLoad.x, uLoad.y, uLoad.z, x, y);
    if (uLoad.w != 0u)
    {
        gl_FragDepth = float(v) * (1.0 / 16777216.0);
        oColor = vec4(0.0);
        return;
    }
    gl_FragDepth = 0.0;
    if (bitsPerPixel(uLoad.x) == 16u)
        v = rgba5551To8888(v);
    else if ((uLoad.x & 0xFu) == 1u)
        v |= 0x80000000u;
    oColor = vec4(float(v & 0xFFu), float((v >> 8u) & 0xFFu), float((v >> 16u) & 0xFFu), float(v >> 24u)) / 255.0;
}
)GLSL";

    // Writes the listed cells of a target back to GS memory, one work group per cell.
    const char *const kTargetStore = R"GLSL(
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 7) readonly buffer DataBuffer { uint data[]; };
layout(binding = 0) uniform sampler2D uTex;
uniform uvec4 uStore;

void main()
{
    uint lo = data[gl_WorkGroupID.x * 2u];
    uint hi = data[gl_WorkGroupID.x * 2u + 1u];
    for (uint y = (lo >> 16u) + gl_LocalInvocationID.y; y < (hi >> 16u); y += 16u)
        for (uint x = (lo & 0xFFFFu) + gl_LocalInvocationID.x; x < (hi & 0xFFFFu); x += 16u)
        {
            vec4 t = texelFetch(uTex, ivec2(x, y), 0);
            uint v;
            if (uStore.w != 0u)
                v = uint(clamp(t.r * 16777216.0 + 0.5, 0.0, 4294967295.0));
            else
            {
                uvec4 c = uvec4(clamp(t * 255.0 + 0.5, 0.0, 255.0));
                v = c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
                if (bitsPerPixel(uStore.x) == 16u)
                    v = rgba8888To5551(v);
            }
            writePixelRaw(uStore.x, uStore.y, uStore.z, x, y, v);
        }
}
)GLSL";
}
