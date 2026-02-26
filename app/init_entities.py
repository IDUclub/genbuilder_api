from loguru import logger

from app.observability.config import PrometheusConfig
from app.observability.otel_agent import OpenTelemetryAgent
import app.dependencies as deps



async def start_prometheus():
    """
    Start the prometheus server
    """

    port = int(deps.config.get("PROMETHEUS_PORT"))
    if deps.otel_agent is not None:
        logger.info(f"Prometheus server already started on {port}")
        return

    logger.info(f"Starting Prometheus server on {port}")
    deps.otel_agent = OpenTelemetryAgent(
        prometheus_config=PrometheusConfig(host="0.0.0.0", port=port),
    )
    logger.info(f"Prometheus server started on {port}")


async def shutdown_prometheus():
    """
    Shutdowns prometheus service
    """

    logger.info("Shutting down Prometheus server")
    if deps.otel_agent is None:
        logger.info("Prometheus server was not started")
        return

    deps.otel_agent.shutdown()
    deps.otel_agent = None
    logger.info("Prometheus server was shut down")