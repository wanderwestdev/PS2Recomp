#include "iop_sio2.h"

#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <sstream>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kSend3 = 0x1F808200u;
        constexpr uint32_t kPortControl = 0x1F808240u;
        constexpr uint32_t kDataIn = 0x1F808260u;
        constexpr uint32_t kDataOut = 0x1F808264u;
        constexpr uint32_t kControl = 0x1F808268u;
        constexpr uint32_t kRecv1 = 0x1F80826Cu;
        constexpr uint32_t kRecv2 = 0x1F808270u;
        constexpr uint32_t kRecv3 = 0x1F808274u;
        constexpr uint32_t kReg278 = 0x1F808278u;
        constexpr uint32_t kReg27C = 0x1F80827Cu;
        constexpr uint32_t kInterruptStatus = 0x1F808280u;

        constexpr uint32_t kControlStart = 0x1u;
        constexpr uint32_t kControlReset = 0xCu;
        constexpr uint32_t kRecv1Connected = 0x1100u;
        constexpr uint32_t kRecv1NoResponse = 0x1D100u;
        constexpr uint32_t kRecv2Default = 0xFu;
        constexpr uint32_t kInterruptTransferDone = 0x1u;

        constexpr uint32_t kSend3Port = 0x3u;
        constexpr uint32_t kSend3LengthMask = 0x1FFu;
        constexpr uint32_t kSend3TxShift = 8u;
        constexpr uint32_t kSend3RxShift = 18u;

        constexpr uint32_t kDmaToDevice = 1u;
        constexpr uint64_t kCompletionBaseCycles = 1'000u;
        constexpr uint64_t kCyclesPerByte = 1'200u;
        constexpr uint64_t kCardCyclesPerByte = 150u;
    }

    IopSio2::IopSio2(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    void IopSio2::reset()
    {
        m_send3.fill(0u);
        m_portControl.fill(0u);
        m_control = 0u;
        m_recv1 = 0u;
        m_recv2 = 0u;
        m_recv3 = 0u;
        m_reg278 = 0u;
        m_reg27C = 0u;
        m_interruptStatus = 0u;
        m_fifoIn.clear();
        m_fifoOut.clear();
        m_dmaIn = {};
        m_dmaOut = {};
        for (VirtualDualShock2 &pad : m_pads)
            pad.reset();
        m_smallMotor.fill(0u);
        m_largeMotor.fill(0u);
        m_reportedMode.fill(0u);
        for (VirtualMemoryCard &card : m_cards)
            card.reset();
        m_completion.reset();
        m_stats = {};
    }

    uint32_t IopSio2::read32(uint32_t address)
    {
        if (address >= kSend3 && address < kPortControl)
            return m_send3[(address - kSend3) / 4u];
        if (address >= kPortControl && address < kDataIn)
            return m_portControl[(address - kPortControl) / 4u];
        switch (address)
        {
        case kDataOut:
        {
            if (m_fifoOut.empty())
                return 0u;
            const uint8_t value = m_fifoOut.front();
            m_fifoOut.pop_front();
            return value;
        }
        case kControl:
            return m_control;
        case kRecv1:
            return m_recv1;
        case kRecv2:
            return m_recv2;
        case kRecv3:
            return m_recv3;
        case kReg278:
            return m_reg278;
        case kReg27C:
            return m_reg27C;
        case kInterruptStatus:
            return m_interruptStatus;
        default:
            return 0u;
        }
    }

    void IopSio2::write32(uint32_t address, uint32_t value)
    {
        if (address >= kSend3 && address < kPortControl)
        {
            m_send3[(address - kSend3) / 4u] = value;
            return;
        }
        if (address >= kPortControl && address < kDataIn)
        {
            m_portControl[(address - kPortControl) / 4u] = value;
            return;
        }
        switch (address)
        {
        case kDataIn:
            m_fifoIn.push_back(static_cast<uint8_t>(value));
            return;
        case kControl:
            if ((value & kControlReset) != 0u)
            {
                m_fifoIn.clear();
                m_fifoOut.clear();
            }
            m_control = value & ~(kControlReset | kControlStart);
            if ((value & kControlStart) != 0u)
                startTransfer();
            return;
        case kReg278:
            m_reg278 = value;
            return;
        case kReg27C:
            m_reg27C = value;
            return;
        case kInterruptStatus:
            m_interruptStatus &= ~value;
            return;
        default:
            return;
        }
    }

    bool IopSio2::dmaRequest(uint32_t channel, uint32_t address, uint32_t blockWords, uint32_t blockCount, uint32_t direction)
    {
        if (channel != kDmaIn && channel != kDmaOut)
            return false;
        DmaRequest &request = channel == kDmaIn ? m_dmaIn : m_dmaOut;
        if ((direction == kDmaToDevice) != (channel == kDmaIn))
            return false;
        request.address = IopMemory::physicalAddress(address);
        request.bytes = blockWords * blockCount * 4u;
        request.armed = false;
        return true;
    }

    bool IopSio2::dmaTransfer(uint32_t channel)
    {
        if (channel != kDmaIn && channel != kDmaOut)
            return false;
        DmaRequest &request = channel == kDmaIn ? m_dmaIn : m_dmaOut;
        request.armed = request.bytes != 0u;
        return true;
    }

    std::optional<IopSio2::Completion> IopSio2::takeCompletion() noexcept
    {
        std::optional<Completion> result = m_completion;
        m_completion.reset();
        return result;
    }

    void IopSio2::startTransfer()
    {
        std::deque<uint8_t> input;
        input.swap(m_fifoIn);
        if (m_dmaIn.armed)
        {
            std::vector<uint8_t> bytes(m_dmaIn.bytes);
            if (m_memory.readRam(m_dmaIn.address, bytes.data(), bytes.size()))
                input.insert(input.end(), bytes.begin(), bytes.end());
            m_dmaIn.armed = false;
        }

        std::vector<uint8_t> output;
        uint32_t recv1 = kRecv1Connected;
        uint64_t cycles = kCompletionBaseCycles;
        for (const uint32_t send3 : m_send3)
        {
            if (send3 == 0u)
                break;
            runCommand(send3, input, output, recv1, cycles);
        }

        size_t delivered = 0u;
        if (m_dmaOut.armed)
        {
            std::vector<uint8_t> bytes(m_dmaOut.bytes, uint8_t{0xFFu});
            delivered = std::min(output.size(), bytes.size());
            std::copy_n(output.begin(), delivered, bytes.begin());
            (void)m_memory.writeRam(m_dmaOut.address, bytes.data(), bytes.size());
            m_dmaOut.armed = false;
        }
        m_fifoOut.insert(m_fifoOut.end(), output.begin() + static_cast<std::ptrdiff_t>(delivered), output.end());

        m_recv1 = recv1;
        m_recv2 = kRecv2Default;
        m_recv3 = 0u;
        m_interruptStatus |= kInterruptTransferDone;
        ++m_stats.transfers;
        m_completion = Completion{cycles};
    }

    void IopSio2::attachCards()
    {
        m_cardsAttached = true;
        const std::filesystem::path root = std::filesystem::path(m_host.hostPath(HostPathKind::MemoryCardRoot));
        const char *enabled = std::getenv("PS2X_MEMCARD");
        if (root.empty() || !enabled || !*enabled || *enabled == '0')
            return;
        const std::filesystem::path folder = root.has_parent_path() ? root.parent_path() : root;
        m_cards[0].attach(folder / "Mcd001.ps2", true);
        m_cards[1].attach(folder / "Mcd002.ps2", false);
        std::ostringstream message;
        message << "[SIO2] memory card slot 1: " << (folder / "Mcd001.ps2").string()
                << (m_cards[0].present() ? "" : " (unavailable)");
        m_host.log(LogLevel::Info, message.str());
    }

    void IopSio2::runCommand(uint32_t send3, std::deque<uint8_t> &input, std::vector<uint8_t> &output, uint32_t &recv1, uint64_t &cycles)
    {
        const uint32_t port = send3 & kSend3Port;
        const size_t tx = (send3 >> kSend3TxShift) & kSend3LengthMask;
        const size_t rx = (send3 >> kSend3RxShift) & kSend3LengthMask;
        const size_t length = std::max(tx, rx);

        std::vector<uint8_t> command(length, uint8_t{0u});
        for (size_t i = 0; i < tx && !input.empty(); ++i)
        {
            command[i] = input.front();
            input.pop_front();
        }
        std::vector<uint8_t> response(length, uint8_t{0xFFu});

        size_t acknowledged = 0u;
        if (port < kPadPorts)
        {
            PadInputState state;
            const bool connected = m_host.readPad(port, state) && state.connected;
            if (connected)
            {
                acknowledged = m_pads[port].transfer(command, response, state);
                ++m_stats.padCommands;
                const uint8_t mode = m_pads[port].modeId();
                if (mode != m_reportedMode[port])
                {
                    std::ostringstream message;
                    message << "[SIO2] pad " << port << " mode 0x" << std::hex << static_cast<uint32_t>(mode);
                    m_host.log(LogLevel::Info, message.str());
                    m_reportedMode[port] = mode;
                }
                const uint8_t small = m_pads[port].smallMotor();
                const uint8_t large = m_pads[port].largeMotor();
                if (small != m_smallMotor[port] || large != m_largeMotor[port])
                {
                    m_smallMotor[port] = small;
                    m_largeMotor[port] = large;
                    m_host.padVibration(port, small, large);
                }
            }
            else
            {
                m_pads[port].reset();
                m_reportedMode[port] = 0u;
            }
        }
        else if (port - kPadPorts < kCardPorts)
        {
            if (!m_cardsAttached)
                attachCards();
            acknowledged = m_cards[port - kPadPorts].transfer(command, response);
            if (acknowledged != 0u)
                ++m_stats.cardCommands;
            static const bool trace = []
            {
                const char *value = std::getenv("PS2X_MC_TRACE");
                return value && *value && *value != '0';
            }();
            if (trace)
            {
                std::ostringstream message;
                message << "[SIO2] card " << (port - kPadPorts) << " tx=" << tx << " rx=" << rx << " ack=" << acknowledged << " in=" << std::hex;
                for (size_t i = 0; i < std::min<size_t>(command.size(), 12u); ++i)
                    message << ' ' << static_cast<uint32_t>(command[i]);
                message << " out=";
                for (size_t i = 0; i < std::min<size_t>(response.size(), 12u); ++i)
                    message << ' ' << static_cast<uint32_t>(response[i]);
                m_host.log(LogLevel::Info, message.str());
            }
        }
        cycles += rx * (port < kPadPorts ? kCyclesPerByte : kCardCyclesPerByte);
        if (acknowledged == 0u)
        {
            recv1 = kRecv1NoResponse;
            ++m_stats.unansweredCommands;
        }
        output.insert(output.end(), response.begin(), response.begin() + static_cast<std::ptrdiff_t>(rx));
    }
}
