// The worker threads (vk_jobs.h). As many as a third of the hardware threads (2 to 8): the game itself runs six
// virtual cores, and the replay thread is one of the busiest.
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "vk_jobs.h"
#include "vk_internal.h"

namespace mkwvk {
namespace {

std::mutex mu;
std::condition_variable work_cv, done_cv;
std::deque<std::function<void()>> queue;
std::atomic<int> pending{0};
std::atomic<int64_t> taken_over{0};
bool started = false;

void worker() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    SetThreadDescription(GetCurrentThread(), L"mkw_vk worker");
    for (;;) {
        std::function<void()> fn;
        {
            std::unique_lock lk(mu);
            work_cv.wait(lk, [] { return !queue.empty(); });
            fn = std::move(queue.front());
            queue.pop_front();
        }
        fn();
        pending--;
        jobs_notify();
    }
}

void start() {
    // (more on big CPUs: a 24-32 thread one still kept 4, and a new area's pipelines queued behind each other.
    // RX_VK_WORKERS=<n> sets it)
    unsigned n = std::clamp(std::thread::hardware_concurrency() / 3, 2u, 8u);
    if (const char* w = getenv("RX_VK_WORKERS"); w && atoi(w) > 0) n = std::min(atoi(w), 32);
    for (unsigned i = 0; i < n; i++) std::thread(worker).detach();
    started = true;
    logf("vk: %u background workers (shader translation, pipelines)", n);
}

} // namespace

void jobs_push(std::function<void()> fn, bool urgent) {
    std::lock_guard lk(mu);
    if (!started) start();
    pending++;
    if (urgent) queue.push_front(std::move(fn));
    else queue.push_back(std::move(fn));
    work_cv.notify_one();
}
void jobs_notify() {
    std::lock_guard lk(mu);
    done_cv.notify_all();
}
void jobs_wait(const std::atomic<bool>& flag) {
    std::unique_lock lk(mu);
    done_cv.wait(lk, [&] { return flag.load(std::memory_order_acquire); });
}
int jobs_pending() {
    return pending.load();
}
void jobs_push_claim(const std::shared_ptr<JobClaim>& claim) {
    jobs_push([claim] {
        if (!claim->taken.exchange(true, std::memory_order_acq_rel)) {
            claim->fn();
            claim->fn = nullptr;
        }
    });
}
void jobs_run_or_wait(JobClaim* claim, const std::atomic<bool>& flag) {
    if (flag.load(std::memory_order_acquire)) return;
    if (claim && !claim->taken.exchange(true, std::memory_order_acq_rel)) {
        claim->fn();
        claim->fn = nullptr;
        taken_over++;
        jobs_notify();
    }
    jobs_wait(flag);
}
int64_t jobs_taken_over() {
    return taken_over.load();
}

} // namespace mkwvk
