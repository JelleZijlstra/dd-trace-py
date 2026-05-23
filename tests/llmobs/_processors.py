"""Test-only trace processors for LLMObs integration tests."""

from typing import Optional
from unittest import mock

from ddtrace._trace.processor import TraceProcessor
from ddtrace._trace.span import Span
from ddtrace._trace.tracer import Tracer
from ddtrace.ext import SpanTypes
from ddtrace.llmobs._constants import LLMOBS_SUBMITTED_TAG_KEY
from ddtrace.llmobs._constants import LLMObsExportMode
from ddtrace.llmobs._llmobs import LLMObs
from ddtrace.llmobs._utils import _get_llmobs_data_metastruct
from ddtrace.llmobs._utils import assemble_llmobs_span_event


class TestAlwaysEnqueueLLMObsProcessor(TraceProcessor):
    """Test-only processor: always enqueue from meta_struct and never scrub it.

      Lets tests assert against both the LLMObs writer events and the rendered payload
    that intake would extract from meta_struct.
    """

    def __init__(
        self,
        llmobs_span_writer,
        export_mode: LLMObsExportMode = LLMObsExportMode.APM_AGENT_PROXY,
    ) -> None:
        super().__init__()
        self._llmobs_span_writer = llmobs_span_writer
        self._export_mode = export_mode

    def process_trace(self, trace: list[Span]) -> Optional[list[Span]]:
        if not trace:
            return trace
        for span in trace:
            if span.span_type != SpanTypes.LLM:
                continue
            if span.get_tag(LLMOBS_SUBMITTED_TAG_KEY) == "1":
                continue
            if not _get_llmobs_data_metastruct(span):
                continue
            try:
                event = assemble_llmobs_span_event(span, self._export_mode)
            except (KeyError, TypeError, ValueError):
                continue
            if event is None:
                continue
            self._llmobs_span_writer.enqueue(event)
            span.set_tag(LLMOBS_SUBMITTED_TAG_KEY, "1")
        return trace


def install_mock_llmobs_writer(tracer: Tracer, mock_writer=None, export_mode=None):
    """Stop the real writer, attach a mock, and rebind the LLMObs processor.

    ``LLMObs.enable()`` wires ``LLMObsTraceProcessor`` to the real writer;
    contrib fixtures must call this after swapping in a mock so meta_struct is not
    scrubbed when the SDK predicts the trace will be dropped.
    """
    if mock_writer is None:
        mock_writer = mock.MagicMock()
    if export_mode is None:
        export_mode = LLMObs._instance._export_mode
    LLMObs._instance._llmobs_span_writer.stop()
    LLMObs._instance._llmobs_span_writer = mock_writer
    tracer._span_aggregator.llmobs_processor = TestAlwaysEnqueueLLMObsProcessor(
        mock_writer,
        export_mode=export_mode,
    )
    return mock_writer
