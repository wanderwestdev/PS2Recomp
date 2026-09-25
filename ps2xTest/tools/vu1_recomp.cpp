#include "ps2_vu1_impl.inl"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kCodeSize = 16u * 1024u;
    constexpr uint32_t kDefaultBlockPairs = 64u;

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

    std::string hexPc(uint32_t pc)
    {
        char text[16];
        std::snprintf(text, sizeof(text), "%04X", pc);
        return text;
    }

    std::string blockSignature(uint64_t hash, uint32_t block)
    {
        return "template <>\nVU1InterpreterAccess::StepResult VU1InterpreterAccess::runBlock<" + hex64(hash) + ", " +
               std::to_string(block) + "u>(VU1Interpreter &vu, const Vu1RunArgs &args)";
    }

    std::string imageSignature(uint64_t hash)
    {
        return "template <>\nVU1InterpreterAccess::StepResult VU1InterpreterAccess::runImage<" + hex64(hash) +
               ">(VU1Interpreter &vu, const Vu1RunArgs &args)";
    }

    uint32_t usedCodeEnd(const std::vector<uint8_t> &code)
    {
        uint32_t end = 0u;
        for (uint32_t pc = 0; pc + 8u <= code.size(); pc += 8u)
        {
            if (std::any_of(code.begin() + pc, code.begin() + pc + 8u, [](uint8_t b) { return b != 0u; }))
                end = pc + 8u;
        }
        return end;
    }

    bool writeImage(const std::filesystem::path &outDir, uint64_t hash, const std::vector<uint8_t> &code, uint32_t blockPairs)
    {
        VU1Interpreter vu(VU1Interpreter::Unit::VU1);
        const uint32_t pairs = usedCodeEnd(code) / 8u;
        const uint32_t blocks = (pairs + blockPairs - 1u) / blockPairs;

        std::ostringstream out;
        out << "#include \"ps2_vu1_impl.inl\"\n\n";
        for (uint32_t block = 0; block < blocks; ++block)
        {
            const uint32_t first = block * blockPairs;
            const uint32_t last = std::min(pairs, first + blockPairs);
            out << blockSignature(hash, block) << "\n{\n";
            out << "    static constexpr VU1Interpreter::DecodedInstructionPair pairs[] = {\n";
            std::vector<bool> reserved(last - first, false);
            for (uint32_t index = first; index < last; ++index)
            {
                bool isReserved = false;
                out << "        " << VU1InterpreterAccess::pairInitializer(vu, code.data(), index * 8u, isReserved) << ",\n";
                reserved[index - first] = isReserved;
            }
            out << "    };\n";
            out << "    switch (vu.m_state.pc)\n    {\n";
            for (uint32_t index = first; index < last; ++index)
                out << "    case 0x" << hexPc(index * 8u) << "u:\n        goto pc_" << hexPc(index * 8u) << ";\n";
            out << "    default:\n        return StepResult::Fallback;\n    }\n";
            for (uint32_t index = first; index < last; ++index)
            {
                const uint32_t pc = index * 8u;
                out << "pc_" << hexPc(pc) << ":\n";
                if (reserved[index - first])
                {
                    out << "    return StepResult::Fallback;\n";
                    continue;
                }
                out << "    if (const StepResult result = generatedStep(vu, pairs[" << (index - first)
                    << "], args); result != StepResult::Continue)\n        return result;\n";
                if (index + 1u < last)
                    out << "    if (vu.m_state.pc != 0x" << hexPc(pc + 8u) << "u)\n        return StepResult::Continue;\n";
            }
            out << "    return StepResult::Continue;\n}\n\n";
        }

        out << imageSignature(hash) << "\n{\n    static constexpr CompiledBlock table[] = {\n";
        for (uint32_t index = 0; index < pairs; ++index)
            out << "        &runBlock<" << hex64(hash) << ", " << (index / blockPairs) << "u>,\n";
        out << "    };\n    return runTable(vu, args, table, " << pairs << "u);\n}\n";

        char name[64];
        std::snprintf(name, sizeof(name), "vu1_%016" PRIx64 ".cpp", hash);
        std::ofstream file(outDir / name, std::ios::binary);
        file << out.str();
        std::cout << name << ": " << pairs << " pairs, " << blocks << " blocks\n";
        return static_cast<bool>(file);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: vu1_recomp <capture-dir> <out-dir> [pairs-per-block]\n";
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
        if (!writeImage(outDir, hash, code, blockPairs))
            return 1;
        hashes.push_back(hash);
    }

    std::ostringstream reg;
    reg << "#include \"ps2_vu1_impl.inl\"\n\n";
    for (const uint64_t hash : hashes)
        reg << imageSignature(hash).replace(imageSignature(hash).find("(VU1Interpreter &vu, const Vu1RunArgs &args)"),
                                            std::string("(VU1Interpreter &vu, const Vu1RunArgs &args)").size(),
                                            "(VU1Interpreter &, const Vu1RunArgs &)")
            << ";\n";
    reg << "\nvoid registerGeneratedVu1Programs()\n{\n";
    for (const uint64_t hash : hashes)
        reg << "    registerVu1CompiledProgram(" << hex64(hash) << ", &VU1InterpreterAccess::runImage<" << hex64(hash) << ">);\n";
    reg << "}\n";
    std::ofstream file(outDir / "vu1_programs.cpp", std::ios::binary);
    file << reg.str();
    return file ? 0 : 1;
}
