# Your Inference Server Is Fast. But Is It On Time?

## SLO-Aware Scheduling for LLM Serving

Your vLLM server does 35 tokens per second. Your dashboard is green. Your P50 latency looks great. And 40% of your users are staring at a spinner.

This is the throughput trap. Every inference benchmark measures tokens per second, but production systems don't fail because they're slow on average — they fail because specific requests miss their deadlines. A chatbot that responds in 1 second 60% of the time and 30 seconds 40% of the time is worse than one that consistently responds in 3 seconds.

The metric that actually matters is **goodput**: the percentage of requests that meet their latency SLO. I built a scheduler that takes goodput from 70% to 100% on an L40S GPU under mixed workloads. Here's what I learned.

## Why FCFS Breaks Under Mixed Workloads

vLLM's default scheduler is FCFS — first come, first served. Under uniform workloads (same prompt length, same output length), every scheduling policy looks identical. FCFS is fine.

Real workloads aren't uniform. A production endpoint serves a mix of quick lookups ("What's the capital of France?" — 50 tokens), medium summaries (200 tokens), and long code generation (500+ tokens). Each has a different latency budget.

The problem: a long code generation request arrives first and occupies GPU batch slots for 14 seconds. The short request behind it — which only needs 2 seconds of compute — waits in line and misses its 2-second SLO. This is head-of-line blocking, and FCFS has no mechanism to prevent it.

I measured this on an L40S with Qwen3.5-9B. Mixed workload: 40% short (SLO: 2s), 40% medium (SLO: 8s), 20% long (SLO: 20s). At QPS=20:

| Policy | Goodput | Short class | Fairness |
|--------|---------|-------------|----------|
| FCFS | 70% | 10% | 0.10 |
| Priority | 70% | 10% | 0.10 |
| **SLO-Aware** | **100%** | **100%** | **1.00** |

FCFS and Priority both achieve 70% goodput — but look at the breakdown. Long and medium requests meet their SLOs (they have generous budgets), while short requests are at 10%. The fairness ratio (worst-class / best-class goodput) is 0.10 — catastrophic. Nine out of ten short requests are late.

## The Proxy Scheduler

Instead of modifying vLLM internals, I built a lightweight proxy that sits between the client and the inference engine:

```
Client requests → SLO Proxy (reorder + admit) → vLLM → Response
```

The proxy implements three mechanisms in ~70 lines of Python:

**Deadline queue.** Each request gets a deadline: `arrival_time + slo_seconds`. Requests are sorted by deadline urgency, not arrival order. When the GPU has capacity, the request closest to missing its deadline goes first.

```python
def enqueue(self, request: WorkloadRequest) -> bool:
    item = ScheduledRequest(request=request)
    if len(self._heap) < self.max_queue_depth:
        heapq.heappush(self._heap, item)
        return False  # Accepted

    # Queue full — evict least urgent if new request is more urgent
    least_urgent = max(self._heap, key=lambda x: x.sort_key)
    if item.sort_key < least_urgent.sort_key:
        self._heap.remove(least_urgent)
        heapq.heapify(self._heap)
        heapq.heappush(self._heap, item)
        return False  # Accepted, evicted a less urgent request

    return True  # Rejected — shed load
```

**Admission control.** When the queue is full, the proxy doesn't just reject the new request — it compares urgency. If the new request has a tighter deadline than something already queued, the least urgent queued request gets evicted. This is load shedding: it's better to reject one relaxed request than to let the entire queue degrade.

**Concurrency limiter.** Caps in-flight requests to prevent GPU memory pressure from degrading all requests simultaneously. Uses an `asyncio.Semaphore` so multiple requests can be dispatched concurrently up to the limit.

The key design decision: this is a proxy, not a vLLM plugin. It works with any OpenAI-compatible backend — I run the same scheduler code on an L40S (vLLM) and an M4 MacBook Pro (Ollama). The scheduling algorithm doesn't care about the hardware.

## The Experiment

I compared three policies on an L40S 48GB GPU with Qwen3.5-9B via vLLM 0.18.0:

- **FCFS**: vLLM default. Requests processed in arrival order.
- **Priority**: vLLM's built-in `--scheduling-policy priority`. Short requests get higher priority.
- **SLO-Aware**: My proxy scheduler with deadline queue + admission control.

The workload: 30 requests per QPS level, mixed 40/40/20 across short (64 max tokens, 2s SLO), medium (256 tokens, 8s SLO), and long (512 tokens, 20s SLO). QPS ramped from 1 to 20.

## Results

| Policy | QPS=1 | QPS=5 | QPS=10 | QPS=20 |
|--------|-------|-------|--------|--------|
| FCFS | 20% | 70% | 60% | 70% |
| Priority | 0% | 73% | 60% | 70% |
| **SLO-Aware** | **67%** | **100%** | **100%** | **100%** |

SLO-Aware achieves **100% goodput from QPS=5 onward** with perfect fairness (1.0). Every request class — short, medium, and long — meets its SLO.

FCFS and Priority both plateau at 60-70%. They serve long and medium requests fine (generous SLO budgets), but short requests are systematically starved. Priority scheduling doesn't help here because vLLM's priority mechanism operates at the engine level — it determines which request gets GPU attention next, but it doesn't reorder the queue by deadline urgency or shed load when the system is saturated.

The QPS=1 results are interesting: Priority actually scores 0% (worse than FCFS's 20%). At QPS=1, requests are sequential, and the model's thinking tokens push even short requests past their 2-second SLO. Priority can't help when there's only one request in the system. SLO-Aware scores 67% at QPS=1 because its deadline reordering kicks in as soon as multiple requests overlap.

SLAI (UT Austin, 2025) reported 53% median TTFT reduction with deadline-aware scheduling. SOLA (Tsinghua, MLSys 2025) improved SLO attainment from 45% to 99%. My results align with this literature — the mechanism is sound, and a proxy-level implementation captures most of the benefit without requiring engine modifications.

## What I Learned

**Goodput should be the default metric.** Throughput tells you how fast the engine is. Goodput tells you how many users are happy. They diverge exactly when it matters — under load.

**Scheduling is invisible at low load.** At QPS=1, every policy looks roughly the same. The differentiation only emerges at saturation. This is why benchmarks that test at a single load level miss the point entirely.

**The proxy pattern is underrated.** A 70-line Python proxy achieved what would otherwise require forking vLLM. It's hardware-agnostic (same code on L40S and M4), engine-agnostic (works with vLLM and Ollama), and composable with any engine-level scheduling that's already running.

**Admission control is the real lever.** Deadline reordering helps, but the biggest impact comes from load shedding. Rejecting one request that can't possibly meet its SLO is better than letting it degrade three other requests that could.

---

*The full implementation (scheduler, benchmarks, results) is open source: [07-slo-scheduling](../07-slo-scheduling/). 49 passing tests, runs on both GPU and Apple Silicon.*
