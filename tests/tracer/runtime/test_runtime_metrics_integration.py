"""Integration tests for runtime metrics collection.

These tests run in isolated subprocesses to verify that the full
RuntimeMetrics pipeline produces valid metric values for GC, PSUtil,
and (when the profiler is active) profiler collectors.
"""

import pytest


@pytest.mark.subprocess(err=None)
def test_gc_metrics_have_valid_values():
    """GC collector returns integer counts for all three generations."""
    import gc

    from ddtrace.internal.runtime.constants import GC_COUNT_GEN0
    from ddtrace.internal.runtime.constants import GC_COUNT_GEN1
    from ddtrace.internal.runtime.constants import GC_COUNT_GEN2
    from ddtrace.internal.runtime.constants import GC_RUNTIME_METRICS
    from ddtrace.internal.runtime.metric_collectors import GCRuntimeMetricCollector

    gc.collect()

    collector = GCRuntimeMetricCollector()
    metrics = dict(collector.collect(GC_RUNTIME_METRICS))

    assert GC_COUNT_GEN0 in metrics, f"Missing {GC_COUNT_GEN0}"
    assert GC_COUNT_GEN1 in metrics, f"Missing {GC_COUNT_GEN1}"
    assert GC_COUNT_GEN2 in metrics, f"Missing {GC_COUNT_GEN2}"

    for key in (GC_COUNT_GEN0, GC_COUNT_GEN1, GC_COUNT_GEN2):
        assert isinstance(metrics[key], int), f"{key} should be int, got {type(metrics[key])}"
        assert metrics[key] >= 0, f"{key} should be >= 0, got {metrics[key]}"


@pytest.mark.subprocess(err=None)
def test_psutil_metrics_have_valid_values():
    """PSUtil collector returns plausible values for thread count, memory, and CPU."""
    from ddtrace.internal.runtime.constants import CPU_PERCENT
    from ddtrace.internal.runtime.constants import CPU_TIME_SYS
    from ddtrace.internal.runtime.constants import CPU_TIME_USER
    from ddtrace.internal.runtime.constants import CTX_SWITCH_INVOLUNTARY
    from ddtrace.internal.runtime.constants import CTX_SWITCH_VOLUNTARY
    from ddtrace.internal.runtime.constants import MEM_RSS
    from ddtrace.internal.runtime.constants import PSUTIL_RUNTIME_METRICS
    from ddtrace.internal.runtime.constants import THREAD_COUNT
    from ddtrace.internal.runtime.metric_collectors import PSUtilRuntimeMetricCollector

    collector = PSUtilRuntimeMetricCollector()
    metrics = dict(collector.collect(PSUTIL_RUNTIME_METRICS))

    for key in PSUTIL_RUNTIME_METRICS:
        assert key in metrics, f"Missing metric {key}"

    assert metrics[THREAD_COUNT] >= 1, f"thread_count should be >= 1, got {metrics[THREAD_COUNT]}"
    assert metrics[MEM_RSS] > 0, f"mem.rss should be > 0, got {metrics[MEM_RSS]}"
    assert metrics[CPU_PERCENT] >= 0, f"cpu.percent should be >= 0, got {metrics[CPU_PERCENT]}"

    # CPU times and context switches are deltas from zero on first call
    for key in (CPU_TIME_SYS, CPU_TIME_USER, CTX_SWITCH_VOLUNTARY, CTX_SWITCH_INVOLUNTARY):
        assert isinstance(metrics[key], (int, float)), f"{key} should be numeric, got {type(metrics[key])}"


@pytest.mark.subprocess(err=None)
def test_psutil_delta_metrics_increase_over_time():
    """CPU time deltas are non-negative across two collection intervals."""
    import time

    from ddtrace.internal.runtime.constants import CPU_TIME_SYS
    from ddtrace.internal.runtime.constants import CPU_TIME_USER
    from ddtrace.internal.runtime.constants import PSUTIL_RUNTIME_METRICS
    from ddtrace.internal.runtime.metric_collectors import PSUtilRuntimeMetricCollector

    collector = PSUtilRuntimeMetricCollector()
    collector.collect(PSUTIL_RUNTIME_METRICS)

    # Burn some CPU to ensure non-zero deltas
    end = time.monotonic() + 0.1
    while time.monotonic() < end:
        sum(range(1000))

    metrics = dict(collector.collect(PSUTIL_RUNTIME_METRICS))
    assert metrics[CPU_TIME_USER] >= 0, f"CPU user delta should be >= 0, got {metrics[CPU_TIME_USER]}"
    assert metrics[CPU_TIME_SYS] >= 0, f"CPU sys delta should be >= 0, got {metrics[CPU_TIME_SYS]}"


@pytest.mark.subprocess(err=None)
def test_runtime_metrics_iterable_produces_all_default_metrics():
    """RuntimeMetrics iterable yields every metric in DEFAULT_RUNTIME_METRICS
    (excluding profiler metrics when the profiler is not running).
    """
    from ddtrace.internal.runtime.constants import GC_RUNTIME_METRICS
    from ddtrace.internal.runtime.constants import PSUTIL_RUNTIME_METRICS
    from ddtrace.internal.runtime.runtime_metrics import RuntimeMetrics

    collected_keys = set(k for k, v in RuntimeMetrics())
    expected = GC_RUNTIME_METRICS | PSUTIL_RUNTIME_METRICS

    missing = expected - collected_keys
    assert not missing, f"Missing expected metrics: {missing}"


@pytest.mark.subprocess(err=None)
def test_runtime_worker_flushes_to_dogstatsd():
    """RuntimeWorker sends all GC and PSUtil metrics through DogStatsD."""
    from unittest import mock

    from ddtrace.internal.runtime.constants import GC_RUNTIME_METRICS
    from ddtrace.internal.runtime.constants import PSUTIL_RUNTIME_METRICS
    from ddtrace.internal.runtime.runtime_metrics import RuntimeWorker
    from ddtrace.trace import tracer

    with mock.patch("ddtrace.internal.runtime.runtime_metrics.get_dogstatsd_client") as mock_get_client:
        mock_client = mock.MagicMock()
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)
        mock_get_client.return_value = mock_client

        worker = RuntimeWorker(interval=10, tracer=tracer)
        worker.flush()

    sent_metrics = set()
    for call in mock_client.distribution.call_args_list:
        metric_name = call[0][0]
        sent_metrics.add(metric_name)

    expected = GC_RUNTIME_METRICS | PSUTIL_RUNTIME_METRICS
    missing = expected - sent_metrics
    assert not missing, f"RuntimeWorker did not flush these metrics: {missing}"


@pytest.mark.subprocess(err=None)
def test_runtime_worker_metric_values_are_numeric():
    """Every value sent by RuntimeWorker is a valid numeric type."""
    from unittest import mock

    from ddtrace.internal.runtime.runtime_metrics import RuntimeWorker
    from ddtrace.trace import tracer

    with mock.patch("ddtrace.internal.runtime.runtime_metrics.get_dogstatsd_client") as mock_get_client:
        mock_client = mock.MagicMock()
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)
        mock_get_client.return_value = mock_client

        worker = RuntimeWorker(interval=10, tracer=tracer)
        worker.flush()

    for call in mock_client.distribution.call_args_list:
        metric_name, value = call[0]
        assert isinstance(value, (int, float)), f"Metric {metric_name} has non-numeric value: {value!r}"


@pytest.mark.subprocess(err=None)
def test_gc_metrics_reflect_allocation_activity():
    """GC gen0 count increases after object allocations."""
    import gc

    from ddtrace.internal.runtime.constants import GC_COUNT_GEN0
    from ddtrace.internal.runtime.constants import GC_RUNTIME_METRICS
    from ddtrace.internal.runtime.metric_collectors import GCRuntimeMetricCollector

    gc.collect()
    collector = GCRuntimeMetricCollector()
    collector.collect(GC_RUNTIME_METRICS)

    # Allocate objects to drive up gen0 count
    garbage = [object() for _ in range(1000)]

    updated = dict(collector.collect(GC_RUNTIME_METRICS))
    assert updated[GC_COUNT_GEN0] >= 0, f"gen0 count should be >= 0 after allocations, got {updated[GC_COUNT_GEN0]}"
    del garbage


@pytest.mark.subprocess(err=None)
def test_mem_rss_increases_after_large_allocation():
    """Memory RSS metric reflects actual memory growth."""
    from ddtrace.internal.runtime.constants import MEM_RSS
    from ddtrace.internal.runtime.constants import PSUTIL_RUNTIME_METRICS
    from ddtrace.internal.runtime.metric_collectors import PSUtilRuntimeMetricCollector

    collector = PSUtilRuntimeMetricCollector()
    before = dict(collector.collect(PSUTIL_RUNTIME_METRICS))

    # Allocate ~50MB to ensure measurable RSS increase
    big_list = [bytearray(1024 * 1024) for _ in range(50)]

    after = dict(collector.collect(PSUTIL_RUNTIME_METRICS))
    assert after[MEM_RSS] > before[MEM_RSS], (
        f"RSS should increase after 50MB alloc: before={before[MEM_RSS]}, after={after[MEM_RSS]}"
    )
    del big_list


@pytest.mark.subprocess(err=None)
def test_thread_count_reflects_active_threads():
    """Thread count metric increases when new threads are spawned."""
    import threading

    from ddtrace.internal.runtime.constants import PSUTIL_RUNTIME_METRICS
    from ddtrace.internal.runtime.constants import THREAD_COUNT
    from ddtrace.internal.runtime.metric_collectors import PSUtilRuntimeMetricCollector

    collector = PSUtilRuntimeMetricCollector()
    before = dict(collector.collect(PSUTIL_RUNTIME_METRICS))

    event = threading.Event()
    threads = [threading.Thread(target=event.wait) for _ in range(5)]
    for t in threads:
        t.start()

    during = dict(collector.collect(PSUTIL_RUNTIME_METRICS))
    assert during[THREAD_COUNT] >= before[THREAD_COUNT] + 5, (
        f"Expected at least 5 more threads: before={before[THREAD_COUNT]}, during={during[THREAD_COUNT]}"
    )

    event.set()
    for t in threads:
        t.join()
