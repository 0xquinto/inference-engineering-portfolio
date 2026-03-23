import asyncio
import heapq
import itertools
from dataclasses import dataclass, field

from .workload import WorkloadRequest


_counter = itertools.count()


@dataclass(order=True)
class ScheduledRequest:
    sort_key: tuple = field(init=False, repr=False)
    request: WorkloadRequest = field(compare=False)
    counter: int = field(default_factory=lambda: next(_counter), compare=False)

    def __post_init__(self):
        self.sort_key = (self.request.deadline, self.counter)

    @property
    def deadline(self):
        return self.request.deadline


class SLOScheduler:
    def __init__(self, max_queue_depth: int = 50, max_concurrent: int = 10):
        self.max_queue_depth = max_queue_depth
        self.max_concurrent = max_concurrent
        self._heap: list[ScheduledRequest] = []
        self._in_flight = 0
        self._event = asyncio.Event()

    def enqueue(self, request: WorkloadRequest) -> bool:
        """Enqueue a request. Returns True if rejected (shed), False if accepted."""
        item = ScheduledRequest(request=request)
        if len(self._heap) < self.max_queue_depth:
            heapq.heappush(self._heap, item)
            self._event.set()
            return False

        # Queue full — reject if new request is less urgent than all queued
        least_urgent = max(self._heap, key=lambda x: x.sort_key)
        if item.sort_key < least_urgent.sort_key:
            self._heap.remove(least_urgent)
            heapq.heapify(self._heap)
            heapq.heappush(self._heap, item)
            self._event.set()
            return False

        return True  # Rejected

    async def dequeue(self) -> WorkloadRequest:
        """Wait for and return the most urgent request."""
        while not self._heap:
            self._event.clear()
            await self._event.wait()
        item = heapq.heappop(self._heap)
        return item.request

    def can_dispatch(self) -> bool:
        return self._in_flight < self.max_concurrent

    def mark_dispatched(self):
        self._in_flight += 1

    def mark_completed(self):
        self._in_flight = max(0, self._in_flight - 1)

    def queue_size(self) -> int:
        return len(self._heap)
