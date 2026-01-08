from os import getenv

from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.sdk.trace import TracerProvider

from .langsmith_processor import LangSmithSpanProcessor


def setup_tracing():
    """Setup OpenTelemetry tracing to export spans to LangSmith."""
    endpoint = getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = getenv("OTEL_EXPORTER_OTLP_HEADERS")

    if not endpoint or not headers:
        print("⚠️  Warning: OTEL environment variables not set. Tracing disabled.")
        return

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(LangSmithSpanProcessor())

    set_tracer_provider(trace_provider)

    print("✅ LangSmith tracing enabled")
