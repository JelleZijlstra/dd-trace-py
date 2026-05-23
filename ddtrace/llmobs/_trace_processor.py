"""LLMObs trace processor wired to ``SpanAggregator.llmobs_processor``."""

from typing import Optional

from ddtrace._trace.processor import TraceProcessor
from ddtrace._trace.span import Span
from ddtrace.ext import SpanTypes
from ddtrace.internal.logger import get_logger
from ddtrace.internal.settings import env
from ddtrace.internal.settings._config import config
from ddtrace.internal.utils.formats import asbool
from ddtrace.llmobs._constants import LLMOBS_STRUCT
from ddtrace.llmobs._constants import LLMOBS_SUBMITTED_TAG_KEY
from ddtrace.llmobs._constants import LLMObsExportMode
from ddtrace.llmobs._utils import assemble_llmobs_span_event
from ddtrace.llmobs._writer import LLMObsSpanWriter


log = get_logger(__name__)

__all__ = ["LLMObsTraceProcessor"]


class LLMObsTraceProcessor(TraceProcessor):
    """Submit LLMObs span events and optionally drop APM traces for LLMObs-only workloads.

    Runs after ``TraceSamplingProcessor`` (priority is final) and before
    ``TraceTagsProcessor``. ``LLMObs._on_span_finish`` must have finalized the payload on
    ``meta_struct["_llmobs"]``; this processor assembles and enqueues when appropriate.
    """

    def __init__(
        self,
        llmobs_span_writer: LLMObsSpanWriter,
        *,
        export_mode: LLMObsExportMode,
        suppress_apm_trace: bool = False,
    ) -> None:
        super().__init__()
        self._llmobs_span_writer = llmobs_span_writer
        self._export_mode = export_mode
        # True when wired from ``LLMObs.enable()``; unit tests leave the default (rescue only).
        self._suppress_apm_trace = suppress_apm_trace
        self._apm_tracing_enabled = asbool(env.get("DD_APM_TRACING_ENABLED", "true"))

    def process_trace(self, trace: list[Span]) -> Optional[list[Span]]:
        if not trace:
            return trace

        for span in trace:
            if span.span_type != SpanTypes.LLM:
                continue
            if span.get_tag(LLMOBS_SUBMITTED_TAG_KEY) == "1":
                continue
            try:
                root = span._local_root or span
                priority = root.context.sampling_priority
                should_submit = self._export_mode == LLMObsExportMode.LLMOBS_DIRECT or (
                    priority is not None and priority <= 0
                )
                if not should_submit:
                    continue
                try:
                    event = assemble_llmobs_span_event(span, self._export_mode)
                except (KeyError, TypeError, ValueError):
                    event = None
                if event is None:
                    span._remove_struct_tag(LLMOBS_STRUCT.KEY)
                    continue
                span.set_tag(LLMOBS_SUBMITTED_TAG_KEY, "1")
                span._remove_struct_tag(LLMOBS_STRUCT.KEY)
                self._llmobs_span_writer.enqueue(event)
            except Exception:
                log.debug(
                    "Failed to submit LLMObs event for span %s; APM trace continues.",
                    span,
                    exc_info=True,
                )

        if not self._suppress_apm_trace:
            return trace

        # LLMObs without APM tracing, or DD_TRACE_ENABLED=0: drop the APM payload.
        if not self._apm_tracing_enabled or not config._tracing_enabled:
            return None

        return trace
