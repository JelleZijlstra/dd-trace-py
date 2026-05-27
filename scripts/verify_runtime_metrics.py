#!/usr/bin/env python3
r"""Workload to verify runtime metrics flow to the Datadog backend.

Run under ddtrace-run with runtime metrics (and optionally profiling):

    DD_RUNTIME_METRICS_ENABLED=1 \
    DD_PROFILING_ENABLED=1 \
    DD_SERVICE=verify-runtime-metrics \
    DD_ENV=dev \
    ddtrace-run python scripts/verify_runtime_metrics.py [duration_seconds]

Default duration is 60s (6 RuntimeWorker flush cycles at 10s interval).
Pass a shorter duration for quick smoke tests, e.g. 20.

Then search Metrics Explorer for:
    runtime.python.gc.count.*
    runtime.python.cpu.*
    runtime.python.mem.rss
    runtime.python.thread_count
    runtime.python.profiler.*       (requires DD_PROFILING_ENABLED=1)

Requires a running Datadog Agent with DogStatsD enabled.
"""

import sys
import threading
import time


stop = threading.Event()


def cpu_work() -> None:
    """Burn CPU until stopped."""
    while not stop.is_set():
        sum(range(50000))


def memory_churn() -> None:
    """Allocate and release memory in a loop until stopped."""
    while not stop.is_set():
        blob = [bytearray(1024 * 1024) for _ in range(10)]  # ~10MB
        stop.wait(0.5)
        del blob


def thread_work() -> None:
    """Spawn short-lived threads periodically until stopped."""
    while not stop.is_set():
        threads = []
        for _ in range(5):
            t = threading.Thread(target=lambda: sum(range(100000)))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        stop.wait(0.5)


def main() -> None:
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    print(f"Running mixed workload for {duration}s")
    print(f"RuntimeWorker flushes every 10s -- expect {duration // 10} flush cycles")
    print()

    workers = [
        threading.Thread(target=cpu_work, name="cpu-burn", daemon=True),
        threading.Thread(target=memory_churn, name="memory-churn", daemon=True),
        threading.Thread(target=thread_work, name="thread-work", daemon=True),
    ]
    for w in workers:
        w.start()

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        pass

    stop.set()
    for w in workers:
        w.join(timeout=3)

    print()
    print("Done. Check Metrics Explorer for runtime.python.* metrics")
    print("  service:verify-runtime-metrics  env:dev")


if __name__ == "__main__":
    main()
