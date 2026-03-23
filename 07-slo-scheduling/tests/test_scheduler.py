import asyncio
import time

from src.scheduler import SLOScheduler, ScheduledRequest
from src.workload import WorkloadRequest


def _make_request(cls="short", slo=2.0, priority=0, deadline=None):
    now = time.monotonic()
    return WorkloadRequest(
        prompt="test", request_class=cls, slo_seconds=slo,
        priority=priority, deadline=deadline or (now + slo),
    )


class TestScheduledRequest:
    def test_ordering_by_deadline(self):
        now = time.monotonic()
        a = ScheduledRequest(_make_request(deadline=now + 1.0))
        b = ScheduledRequest(_make_request(deadline=now + 5.0))
        assert a < b

    def test_equal_deadline_uses_counter(self):
        now = time.monotonic()
        a = ScheduledRequest(_make_request(deadline=now + 1.0))
        b = ScheduledRequest(_make_request(deadline=now + 1.0))
        assert a < b


class TestSLOScheduler:
    def test_enqueue_dequeue_order(self):
        async def _test():
            scheduler = SLOScheduler(max_queue_depth=10, max_concurrent=5)
            now = time.monotonic()
            r1 = _make_request(deadline=now + 5.0)
            r2 = _make_request(deadline=now + 1.0)
            r3 = _make_request(deadline=now + 3.0)
            scheduler.enqueue(r1)
            scheduler.enqueue(r2)
            scheduler.enqueue(r3)
            out = await scheduler.dequeue()
            assert out.deadline == r2.deadline
        asyncio.run(_test())

    def test_admission_control_rejects(self):
        scheduler = SLOScheduler(max_queue_depth=2, max_concurrent=5)
        now = time.monotonic()
        scheduler.enqueue(_make_request(deadline=now + 1.0))
        scheduler.enqueue(_make_request(deadline=now + 2.0))
        rejected = scheduler.enqueue(_make_request(deadline=now + 10.0))
        assert rejected is True

    def test_admission_accepts_urgent_over_relaxed(self):
        scheduler = SLOScheduler(max_queue_depth=2, max_concurrent=5)
        now = time.monotonic()
        scheduler.enqueue(_make_request(deadline=now + 5.0))
        scheduler.enqueue(_make_request(deadline=now + 10.0))
        rejected = scheduler.enqueue(_make_request(deadline=now + 1.0))
        assert rejected is False
        assert scheduler.queue_size() == 2

    def test_concurrency_limiter(self):
        scheduler = SLOScheduler(max_queue_depth=10, max_concurrent=2)
        assert scheduler.can_dispatch() is True
        scheduler.mark_dispatched()
        scheduler.mark_dispatched()
        assert scheduler.can_dispatch() is False
        scheduler.mark_completed()
        assert scheduler.can_dispatch() is True

    def test_queue_size(self):
        scheduler = SLOScheduler(max_queue_depth=10, max_concurrent=5)
        assert scheduler.queue_size() == 0
        scheduler.enqueue(_make_request())
        assert scheduler.queue_size() == 1
