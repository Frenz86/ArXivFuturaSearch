"""OpenTelemetry distributed tracing integration.


# Copyright 2025 ArXivFuturaSearch Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Provides automatic tracing for FastAPI applications with export to
various backends (Jaeger, OTLP, etc.).
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from contextlib import asynccontextmanager, contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# OTEL CONFIGURATION
# =============================================================================

class OpenTelemetryConfig:
    """Configuration for OpenTelemetry tracing and metrics."""

    def __init__(
        self,
        service_name: str = None,
        service_version: str = None,
        enabled: bool = True,
        trace_exporter: str = "otlp",  # otlp, jaeger, console, none
        metrics_exporter: str = "console",  # otlp, console, none
        otlp_endpoint: str = None,
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        sample_rate: float = 1.0,
    ):
        """
        Initialize OpenTelemetry configuration.

        Args:
            service_name: Service name for tracing
            service_version: Service version
            enabled: Whether tracing is enabled
            trace_exporter: Trace exporter type
            metrics_exporter: Metrics exporter type
            otlp_endpoint: OTLP collector endpoint
            jaeger_host: Jaeger agent host
            jaeger_port: Jaeger agent port
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        self.service_name = service_name or settings.APP_NAME
        self.service_version = service_version or settings.VERSION
        self.enabled = enabled and settings.ENVIRONMENT != "test"
        self.trace_exporter = trace_exporter
        self.metrics_exporter = metrics_exporter
        self.otlp_endpoint = otlp_endpoint
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.sample_rate = sample_rate


# =============================================================================
# TRACING SETUP
# =============================================================================

_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None


def setup_tracing(config: OpenTelemetryConfig) -> Optional[TracerProvider]:
    """
    Setup OpenTelemetry tracing.

    Args:
        config: OpenTelemetry configuration

    Returns:
        TracerProvider or None if tracing disabled
    """
    if not config.enabled or config.trace_exporter == "none":
        logger.info("Tracing disabled")
        return None

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "service.environment": settings.ENVIRONMENT,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add span processor based on exporter type
    if config.trace_exporter == "otlp":
        endpoint = config.otlp_endpoint or settings.OTEL_EXPORTER_OTLP_ENDPOINT
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        logger.info("OTLP tracing configured", endpoint=endpoint)

    elif config.trace_exporter == "jaeger":
        exporter = JaegerExporter(
            agent_host_name=config.jaeger_host,
            agent_port=config.jaeger_port,
        )
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        logger.info(
            "Jaeger tracing configured",
            host=config.jaeger_host,
            port=config.jaeger_port,
        )

    elif config.trace_exporter == "console":
        exporter = ConsoleSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        logger.info("Console tracing configured")

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    global _tracer_provider
    _tracer_provider = provider

    logger.info("OpenTelemetry tracing initialized")
    return provider


def setup_metrics(config: OpenTelemetryConfig) -> Optional[MeterProvider]:
    """
    Setup OpenTelemetry metrics.

    Args:
        config: OpenTelemetry configuration

    Returns:
        MeterProvider or None if metrics disabled
    """
    if not config.enabled or config.metrics_exporter == "none":
        logger.info("Metrics disabled")
        return None

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
    })

    # Create metric reader
    if config.metrics_exporter == "console":
        exporter = ConsoleMetricExporter()
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15000)
        logger.info("Console metrics configured")
    else:
        # OTLP metrics exporter
        reader = PeriodicExportingMetricReader(
            OTLPSpanExporter(endpoint=config.otlp_endpoint),
            export_interval_millis=60000,
        )
        logger.info("OTLP metrics configured")

    # Create meter provider
    provider = MeterProvider(resource=resource, metric_readers=[reader])

    # Set global meter provider
    metrics.set_meter_provider(provider)

    global _meter_provider
    _meter_provider = provider

    logger.info("OpenTelemetry metrics initialized")
    return provider


def setup_fastapi_instrumentation(app, config: OpenTelemetryConfig) -> None:
    """
    Instrument FastAPI app with OpenTelemetry.

    Args:
        app: FastAPI application
        config: OpenTelemetry configuration
    """
    if not config.enabled:
        return

    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation added")
    except Exception as e:
        logger.warning("FastAPI instrumentation failed", error=str(e))


def setup_httpx_instrumentation(config: OpenTelemetryConfig) -> None:
    """
    Instrument HTTPX for outgoing request tracing.

    Args:
        config: OpenTelemetry configuration
    """
    if not config.enabled:
        return

    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumentation added")
    except Exception as e:
        logger.warning("HTTPX instrumentation failed", error=str(e))


def shutdown_telemetry() -> None:
    """Shutdown OpenTelemetry providers."""
    global _tracer_provider, _meter_provider

    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None
        logger.info("Tracer provider shutdown")

    if _meter_provider:
        _meter_provider.shutdown()
        _meter_provider = None
        logger.info("Meter provider shutdown")


# =============================================================================
# SPAN HELPERS
# =============================================================================

def get_tracer(name: str = None) -> trace.Tracer:
    """
    Get a tracer for creating spans.

    Args:
        name: Tracer name (default: service name)

    Returns:
        Tracer instance
    """
    name = name or settings.APP_NAME
    return trace.get_tracer(name)


@asynccontextmanager
async def async_trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for async span creation.

    Args:
        name: Span name
        attributes: Span attributes

    Example:
        async with async_trace_span("database.query", {"query": sql}):
            result = await db.execute(sql)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for sync span creation.

    Args:
        name: Span name
        attributes: Span attributes

    Example:
        with trace_span("file.read", {"path": filepath}):
            data = read_file(filepath)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to trace function execution.

    Args:
        name: Span name (default: function name)
        attributes: Static span attributes

    Example:
        @traced("llm.generate", {"model": "gpt-4"})
        async def generate(prompt: str):
            return await llm(prompt)
    """
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with async_trace_span(span_name, attributes):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with trace_span(span_name, attributes):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


# =============================================================================
# CUSTOM SPAN ATTRIBUTES
# =============================================================================

def set_span_attribute(key: str, value: Any) -> None:
    """
    Set attribute on current span.

    Args:
        key: Attribute key
        value: Attribute value
    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute(key, value)


def set_span_error(exception: Exception) -> None:
    """
    Record exception on current span.

    Args:
        exception: Exception to record
    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.record_exception(exception)
        current_span.set_status(
            trace.Status(trace.StatusCode.ERROR, str(exception))
        )


def add_span_event(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add event to current span.

    Args:
        name: Event name
        attributes: Event attributes
    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes=attributes or {})


# =============================================================================
# TRACE CONTEXT PROPAGATION
# =============================================================================

def get_trace_context() -> Dict[str, str]:
    """
    Get current trace context for propagation.

    Returns:
        Dict with traceparent headers
    """
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import Context

    ctx = Context()
    carrier = {}
    TraceContextTextMapPropagator().inject(carrier, context=ctx)
    return carrier


def inject_trace_context(headers: Dict[str, str]) -> None:
    """
    Inject trace context into headers dict.

    Args:
        headers: Headers dict to inject into
    """
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    TraceContextTextMapPropagator().inject(headers)


def extract_trace_context(headers: Dict[str, str]):
    """
    Extract trace context from headers.

    Args:
        headers: Headers dict with trace context

    Returns:
        Context object
    """
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    return TraceContextTextMapPropagator().extract(headers)


# =============================================================================
# INITIALIZATION HELPERS
# =============================================================================

async def init_telemetry(
    app=None,
    config: Optional[OpenTelemetryConfig] = None,
) -> OpenTelemetryConfig:
    """
    Initialize OpenTelemetry tracing and metrics.

    Args:
        app: Optional FastAPI app to instrument
        config: Optional OpenTelemetry config

    Returns:
        OpenTelemetryConfig instance
    """
    config = config or OpenTelemetryConfig()

    # Setup tracing
    setup_tracing(config)

    # Setup metrics
    setup_metrics(config)

    # Instrument FastAPI if provided
    if app:
        setup_fastapi_instrumentation(app, config)

    # Instrument HTTPX
    setup_httpx_instrumentation(config)

    return config


@asynccontextmanager
async def telemetry_context(
    config: Optional[OpenTelemetryConfig] = None,
):
    """
    Context manager for telemetry lifecycle.

    Args:
        config: Optional OpenTelemetry config

    Example:
        async with telemetry_context():
            # Your application code here
            pass
    """
    config = config or OpenTelemetryConfig()
    await init_telemetry(config=config)

    try:
        yield config
    finally:
        shutdown_telemetry()
