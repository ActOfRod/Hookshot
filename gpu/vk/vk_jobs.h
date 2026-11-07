// Background work of the back end (vk_jobs.cpp): shader translation and pipeline builds run on a few worker threads
// at below-normal priority while the replay goes on, after totk-pc-recomp's vk_pipeline.c background builds. A draw
// whose program or pipeline is not ready yet is left out of its frame, or waits for it (vk_draw.cpp decides).
#pragma once

#include <atomic>
#include <functional>
#include <memory>

namespace mkwvk {

/// Queues fn for a worker (urgent: ahead of the others, for work a draw waits on)
void jobs_push(std::function<void()> fn, bool urgent = false);
/// Waits until flag is set by a job (which calls jobs_notify after setting it)
void jobs_wait(const std::atomic<bool>& flag);
void jobs_notify();
int jobs_pending();

/// A job a waiting draw may take over: whichever of a worker and the waiter claims it first runs it
struct JobClaim {
    std::atomic<bool> taken{false};
    std::function<void()> fn;
};
/// Queues claim's job for a worker
void jobs_push_claim(const std::shared_ptr<JobClaim>& claim);
/// Waits until flag is set, running claim's job on this thread first when no worker has started it (instead of waiting
/// for the whole queue ahead of it)
void jobs_run_or_wait(JobClaim* claim, const std::atomic<bool>& flag);
/// How many jobs a waiter ran itself
int64_t jobs_taken_over();

} // namespace mkwvk
