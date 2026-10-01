#include "runtime/ps2_async_vif.h"
#include "runtime/ps2_memory.h"
#include "runtime/ee_scheduler.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "ThreadPriority.h"
#include "ThreadNaming.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <algorithm>
#include <xmmintrin.h>

namespace ps2x::asyncvif
{
    namespace
    {
        constexpr size_t kMaxQueuedJobs = 64u;
        constexpr size_t kMaxQueuedGifBytes = 8u * 1024u * 1024u;

        struct Job
        {
            std::vector<Item> items;
            uint32_t mxcsr = 0u;
            uint32_t fbrst = 0u;
        };

        struct GifPacketRef
        {
            uint32_t offset = 0u;
            uint32_t size = 0u;
            uint32_t mxcsr = 0u;
        };

        struct GifChunk
        {
            std::vector<uint8_t> data;
            std::vector<GifPacketRef> packets;
        };

        struct GifStage
        {
            std::mutex mutex;
            std::condition_variable workCv;
            std::condition_variable idleCv;
            std::deque<GifChunk> chunks;
            std::vector<GifChunk> spare;
            size_t queuedBytes = 0u;
            bool busy = false;
            bool stop = false;
            GifChunk current;
            std::thread thread;
        };

        constexpr size_t kGifChunkBytes = 256u * 1024u;

        void publishGif(GifStage &stage)
        {
            if (stage.current.packets.empty())
                return;
            GifChunk next;
            {
                std::unique_lock<std::mutex> lock(stage.mutex);
                stage.idleCv.wait(lock, [&]
                                  { return stage.queuedBytes < kMaxQueuedGifBytes; });
                stage.queuedBytes += stage.current.data.size();
                stage.chunks.push_back(std::move(stage.current));
                if (!stage.spare.empty())
                {
                    next = std::move(stage.spare.back());
                    stage.spare.pop_back();
                }
            }
            stage.workCv.notify_one();
            stage.current = std::move(next);
            stage.current.data.clear();
            stage.current.packets.clear();
        }

        struct Worker
        {
            PS2Memory *memory = nullptr;
            std::mutex mutex;
            std::condition_variable workCv;
            std::condition_variable idleCv;
            std::deque<Job> jobs;
            std::vector<std::vector<uint8_t>> spare;
            bool busy = false;
            bool stop = false;
            std::thread thread;
            GifStage gif;
        };

        std::function<void(const uint8_t *, uint32_t)> s_gifProcessor;
        thread_local bool t_onGifStage = false;

        std::atomic<bool> s_defaultEnabled{false};
        std::atomic<bool> s_hostPacingDefault{false};
        std::mutex s_bindMutex;
        std::atomic<Worker *> s_worker{nullptr};
        std::function<uint32_t()> s_fbrstProvider;
        thread_local bool t_onWorker = false;
        std::function<uint64_t()> s_vu1CycleReader;
        std::atomic<bool> s_vu1ReaderReady{false};
        std::atomic<uint64_t> s_vu1Completed{0u};
        thread_local uint32_t t_fbrst = 0u;

        bool enabled()
        {
            static const int fromEnvironment = []
            {
                const char *value = std::getenv("PS2X_MTVU");
                if (!value || !*value)
                    return -1;
                return *value != '0' ? 1 : 0;
            }();
            return fromEnvironment >= 0 ? fromEnvironment != 0 : s_defaultEnabled.load(std::memory_order_relaxed);
        }

        void runGif(GifStage &stage)
        {
            ThreadNaming::SetCurrentThreadName("GIFThread");
            ThreadPriority::RaiseCurrentThread(ThreadPriority::Level::AboveNormal);
            t_onGifStage = true;
            for (;;)
            {
                GifChunk chunk;
                {
                    std::unique_lock<std::mutex> lock(stage.mutex);
                    stage.workCv.wait(lock, [&]
                                      { return stage.stop || !stage.chunks.empty(); });
                    if (stage.stop && stage.chunks.empty())
                        return;
                    chunk = std::move(stage.chunks.front());
                    stage.chunks.pop_front();
                    stage.queuedBytes -= chunk.data.size();
                    stage.busy = true;
                }
                stage.idleCv.notify_all();
                for (const GifPacketRef &packet : chunk.packets)
                {
                    _mm_setcsr(packet.mxcsr);
                    if (s_gifProcessor)
                        s_gifProcessor(chunk.data.data() + packet.offset, packet.size);
                }
                {
                    std::lock_guard<std::mutex> lock(stage.mutex);
                    if (stage.spare.size() < 8u)
                        stage.spare.push_back(std::move(chunk));
                    stage.busy = false;
                }
                stage.idleCv.notify_all();
            }
        }

        void run(Worker &worker)
        {
            ThreadNaming::SetCurrentThreadName("VUThread");
            ThreadPriority::RaiseCurrentThread(ThreadPriority::Level::Highest);
            t_onWorker = true;
            const char *statsValue = std::getenv("PS2X_MTVU_STATS");
            const bool stats = statsValue && *statsValue && *statsValue != '0';
            using Clock = std::chrono::steady_clock;
            Clock::duration busyTime{};
            Clock::time_point windowStart = Clock::now();
            uint64_t windowField = worker.memory->gs().vsyncTick.load();
            uint64_t windowCycles = 0u;
            for (;;)
            {
                Job job;
                {
                    std::unique_lock<std::mutex> lock(worker.mutex);
                    worker.workCv.wait(lock, [&]
                                       { return worker.stop || !worker.jobs.empty(); });
                    if (worker.stop && worker.jobs.empty())
                        return;
                    job = std::move(worker.jobs.front());
                    worker.jobs.pop_front();
                    worker.busy = true;
                }
                worker.idleCv.notify_all();
                const Clock::time_point jobStart = Clock::now();
                _mm_setcsr(job.mxcsr);
                t_fbrst = job.fbrst;
                PS2Memory &memory = *worker.memory;
                for (Item &item : job.items)
                {
                    if (item.data.empty())
                        continue;
                    if (item.kind == ItemKind::GifPath3)
                        memory.submitGifPacket(GifPathId::Path3, item.data.data(), static_cast<uint32_t>(item.data.size()), false);
                    else
                        memory.processVIF1Data(item.data.data(), static_cast<uint32_t>(item.data.size()));
                }
                if (memory.m_gifArbiter)
                    memory.m_gifArbiter->drain();
                if (s_vu1ReaderReady.load(std::memory_order_acquire))
                    s_vu1Completed.store(s_vu1CycleReader(), std::memory_order_release);
                publishGif(worker.gif);
                if (stats)
                {
                    const Clock::time_point now = Clock::now();
                    busyTime += now - jobStart;
                    if (now - windowStart >= std::chrono::seconds(2))
                    {
                        const uint64_t field = worker.memory->gs().vsyncTick.load();
                        const double window = std::chrono::duration<double>(now - windowStart).count();
                        const double busy = std::chrono::duration<double>(busyTime).count();
                        const uint64_t fields = field > windowField ? field - windowField : 1u;
                        const uint64_t cycles = s_vu1Completed.load(std::memory_order_acquire);
                        std::fprintf(stderr, "[mtvu] field %llu: %.1f fields/s, VU busy %.0f%%, %.2f ms VU work per field, %.2f M VU cycles per field\n",
                                     static_cast<unsigned long long>(field), fields / window, 100.0 * busy / window, 1000.0 * busy / fields,
                                     static_cast<double>(cycles >= windowCycles ? cycles - windowCycles : cycles) / fields / 1000000.0);
                        std::fflush(stderr);
                        busyTime = {};
                        windowStart = now;
                        windowField = field;
                        windowCycles = cycles;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(worker.mutex);
                    for (Item &item : job.items)
                    {
                        if (worker.spare.size() < 32u && item.data.capacity() != 0u)
                        {
                            item.data.clear();
                            worker.spare.push_back(std::move(item.data));
                        }
                    }
                    worker.busy = false;
                }
                worker.idleCv.notify_all();
            }
        }
    }

    void setDefaultEnabled(bool value)
    {
        s_defaultEnabled.store(value, std::memory_order_relaxed);
    }

    void setHostPacingDefault(bool value)
    {
        s_hostPacingDefault.store(value, std::memory_order_relaxed);
    }

    void setFbrstProvider(std::function<uint32_t()> provider)
    {
        std::lock_guard<std::mutex> lock(s_bindMutex);
        s_fbrstProvider = std::move(provider);
    }

    bool onWorker()
    {
        return t_onWorker;
    }

    void setGifProcessor(std::function<void(const uint8_t *, uint32_t)> processor)
    {
        std::lock_guard<std::mutex> lock(s_bindMutex);
        s_gifProcessor = std::move(processor);
    }

    bool submitGif(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!t_onWorker)
            return false;
        Worker *worker = s_worker;
        if (!worker)
            return false;
        GifStage &stage = worker->gif;
        GifChunk &chunk = stage.current;
        GifPacketRef packet;
        packet.offset = static_cast<uint32_t>(chunk.data.size());
        packet.size = sizeBytes;
        packet.mxcsr = _mm_getcsr();
        if (chunk.data.capacity() < kGifChunkBytes)
            chunk.data.reserve(kGifChunkBytes);
        chunk.data.insert(chunk.data.end(), data, data + sizeBytes);
        chunk.packets.push_back(packet);
        if (chunk.data.size() >= kGifChunkBytes)
            publishGif(stage);
        return true;
    }

    uint32_t currentFbrst()
    {
        return t_fbrst;
    }

    bool active(PS2Memory &memory)
    {
        if (t_onWorker)
            return false;
        Worker *worker = s_worker;
        if (worker)
            return worker->memory == &memory;
        if (!enabled())
            return false;
        std::lock_guard<std::mutex> lock(s_bindMutex);
        if (!s_worker)
        {
            auto *created = new Worker();
            created->memory = &memory;
            created->gif.thread = std::thread([created]
                                              { runGif(created->gif); });
            created->thread = std::thread([created]
                                          { run(*created); });
            s_worker.store(created);
        }
        return s_worker.load()->memory == &memory;
    }

    std::vector<uint8_t> takeBuffer()
    {
        Worker *worker = s_worker;
        if (!worker)
            return {};
        std::lock_guard<std::mutex> lock(worker->mutex);
        if (worker->spare.empty())
            return {};
        std::vector<uint8_t> buffer = std::move(worker->spare.back());
        worker->spare.pop_back();
        return buffer;
    }

    void enqueue(PS2Memory &memory, std::vector<Item> &&items)
    {
        Worker *worker = s_worker;
        if (!worker || worker->memory != &memory || items.empty())
            return;
        Job job;
        job.items = std::move(items);
        job.mxcsr = _mm_getcsr();
        {
            std::lock_guard<std::mutex> lock(s_bindMutex);
            job.fbrst = s_fbrstProvider ? s_fbrstProvider() : 0u;
        }
        {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->idleCv.wait(lock, [&]
                                { return worker->jobs.size() < kMaxQueuedJobs; });
            worker->jobs.push_back(std::move(job));
        }
        worker->workCv.notify_one();
    }

    namespace
    {
        struct Vu1Timing
        {
            Vu1TimingHooks hooks;
            bool enabled = false;
            double scale = 1.0;
            bool primed = false;
            uint64_t accounted = 0;
            uint64_t busyUntil = 0;
            bool kickPending = false;
            uint64_t kickCycle = 0;
            bool blocking = false;
            bool hostPacing = false;
            uint64_t hostKickCycle = 0u;
            std::chrono::steady_clock::time_point hostKickTime{};
        };

        Vu1Timing &vu1Timing()
        {
            static Vu1Timing timing;
            return timing;
        }

        void syncVu(PS2Memory &memory)
        {
            if (t_onWorker)
                return;
            Worker *worker = s_worker;
            if (!worker || worker->memory != &memory)
                return;
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->idleCv.wait(lock, [&]
                                { return worker->jobs.empty() && !worker->busy; });
        }
    }

    void setVu1Timing(Vu1TimingHooks hooks)
    {
        Vu1Timing &timing = vu1Timing();
        timing = Vu1Timing{};
        timing.hooks = std::move(hooks);
        double scale = 1.0;
        if (const char *value = std::getenv("PS2X_VU1_TIMING"))
            scale = std::atof(value);
        timing.scale = scale;
        timing.enabled = scale > 0.0 && timing.hooks.eeCycle && timing.hooks.vu1Cycles && timing.hooks.consumeEeCycles;
        const char *blocking = std::getenv("PS2X_VU1_TIMING_SYNC");
        timing.blocking = blocking && *blocking && *blocking != '0';
        const char *hostPacing = std::getenv("PS2X_VU1_HOST_PACING");
        timing.hostPacing = hostPacing && *hostPacing ? *hostPacing != '0' : s_hostPacingDefault.load(std::memory_order_relaxed);
        if (timing.enabled && !s_vu1ReaderReady.load(std::memory_order_acquire))
        {
            s_vu1CycleReader = timing.hooks.vu1Cycles;
            s_vu1ReaderReady.store(true, std::memory_order_release);
        }
    }

    void vu1Kick()
    {
        Vu1Timing &timing = vu1Timing();
        if (!timing.enabled || t_onWorker)
            return;
        if (timing.hostPacing)
        {
            timing.hostKickCycle = timing.hooks.eeCycle();
            timing.hostKickTime = std::chrono::steady_clock::now();
        }
        if (timing.kickPending)
            return;
        timing.kickPending = true;
        timing.kickCycle = timing.hooks.eeCycle();
    }

    bool vu1Busy(PS2Memory &memory)
    {
        Vu1Timing &timing = vu1Timing();
        if (!timing.enabled || t_onWorker)
            return false;
        Worker *worker = s_worker;
        if (timing.hostPacing && timing.primed && timing.hostKickTime != std::chrono::steady_clock::time_point{} &&
            worker && worker->memory == &memory)
        {
            bool pending;
            {
                std::lock_guard<std::mutex> lock(worker->mutex);
                pending = worker->busy || !worker->jobs.empty();
            }
            if (pending)
            {
                const auto elapsed = std::chrono::steady_clock::now() - timing.hostKickTime;
                const uint64_t elapsedCycles = static_cast<uint64_t>(std::chrono::duration<double>(elapsed).count() * EeScheduler::kEeClockHz);
                const uint64_t target = timing.hostKickCycle + elapsedCycles;
                const uint64_t now = timing.hooks.eeCycle();
                if (target > now)
                    timing.hooks.consumeEeCycles(std::min<uint64_t>(target - now, 65536u));
                else
                    std::this_thread::yield();
                return true;
            }
        }
        const bool async = !timing.blocking && timing.primed && worker && worker->memory == &memory;
        if (!async)
            syncVu(memory);
        const uint64_t total = async ? std::max(timing.accounted, s_vu1Completed.load(std::memory_order_acquire)) : timing.hooks.vu1Cycles();
        uint64_t now = timing.hooks.eeCycle();
        if (!timing.primed)
        {
            timing.primed = true;
            timing.accounted = total;
            timing.busyUntil = now;
            timing.kickPending = false;
            return false;
        }
        if (total > timing.accounted)
        {
            const uint64_t work = static_cast<uint64_t>(static_cast<double>(total - timing.accounted) * timing.scale);
            const uint64_t start = timing.kickPending ? std::max(timing.busyUntil, timing.kickCycle) : timing.busyUntil;
            timing.busyUntil = start + work;
            timing.accounted = total;
            timing.kickPending = false;
        }
        if (now >= timing.busyUntil)
            return false;
        timing.hooks.consumeEeCycles(timing.busyUntil - now);
        now = timing.hooks.eeCycle();
        return now < timing.busyUntil;
    }

    void sync(PS2Memory &memory)
    {
        if (t_onWorker)
            return;
        Worker *worker = s_worker;
        if (!worker || worker->memory != &memory)
            return;
        {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->idleCv.wait(lock, [&]
                                { return worker->jobs.empty() && !worker->busy; });
        }
        GifStage &stage = worker->gif;
        std::unique_lock<std::mutex> lock(stage.mutex);
        stage.idleCv.wait(lock, [&]
                          { return stage.chunks.empty() && !stage.busy; });
    }

    void stop(PS2Memory &memory)
    {
        Worker *worker = nullptr;
        {
            std::lock_guard<std::mutex> lock(s_bindMutex);
            if (!s_worker.load() || s_worker.load()->memory != &memory)
                return;
            worker = s_worker;
            s_worker.store(nullptr);
        }
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->stop = true;
        }
        worker->workCv.notify_all();
        if (worker->thread.joinable())
            worker->thread.join();
        {
            std::lock_guard<std::mutex> lock(worker->gif.mutex);
            worker->gif.stop = true;
        }
        worker->gif.workCv.notify_all();
        if (worker->gif.thread.joinable())
            worker->gif.thread.join();
        delete worker;
    }

    void saveTimingState(ps2x::state::Writer &out)
    {
        Vu1Timing &timing = vu1Timing();
        ps2x::state::fields(out, timing.primed, timing.accounted, timing.busyUntil, timing.kickPending, timing.kickCycle);
    }

    void loadTimingState(ps2x::state::Reader &in)
    {
        Vu1Timing &timing = vu1Timing();
        ps2x::state::fields(in, timing.primed, timing.accounted, timing.busyUntil, timing.kickPending, timing.kickCycle);
        timing.hostKickCycle = timing.hooks.eeCycle ? timing.hooks.eeCycle() : 0u;
        timing.hostKickTime = std::chrono::steady_clock::now();
        if (s_vu1ReaderReady.load(std::memory_order_acquire))
            s_vu1Completed.store(s_vu1CycleReader(), std::memory_order_release);
    }
}
