"""ORION RPC server — launches agents as separate AgentScope RPC processes.

Provides a launcher script and client stubs for running agents in
distributed mode. Each agent runs as an independent RPC service.

Usage:
    python -m orion.orchestrator.rpc_server --agent planner --port 12310
    python -m orion.orchestrator.rpc_server --agent executor --port 12311
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default RPC ports per agent
_DEFAULT_PORTS = {
    "planner": 12310,
    "executor": 12311,
    "verifier": 12312,
    "supervisor": 12313,
}


class RPCConfig:
    """Configuration for an RPC agent server.

    Args:
        agent_name: Name of the agent to launch.
        host: Bind address.
        port: Bind port.
        model_config: Model wrapper configuration dict.
    """

    def __init__(
        self,
        agent_name: str,
        host: str = "0.0.0.0",
        port: int | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.host = host
        self.port = port or _DEFAULT_PORTS.get(agent_name, 12310)
        self.model_config = model_config or {}


def create_agent(config: RPCConfig) -> Any:
    """Create an agent instance from RPC config.

    Args:
        config: RPC configuration.

    Returns:
        Configured agent instance.
    """
    from orion.agents.executor import ExecutorAgent
    from orion.agents.planner import PlannerAgent
    from orion.agents.supervisor import SupervisorAgent
    from orion.agents.verifier import VerifierAgent

    agent_map = {
        "planner": PlannerAgent,
        "executor": ExecutorAgent,
        "verifier": VerifierAgent,
        "supervisor": SupervisorAgent,
    }

    agent_cls = agent_map.get(config.agent_name)
    if agent_cls is None:
        msg = f"Unknown agent: {config.agent_name}"
        raise ValueError(msg)

    return agent_cls()


async def serve(config: RPCConfig) -> None:
    """Start an RPC agent server.

    In the current implementation, this is a placeholder that logs
    the configuration and waits. Full AgentScope RPC integration
    requires the AgentScope server infrastructure.

    Args:
        config: RPC configuration.
    """
    create_agent(config)
    logger.info(
        "rpc_server_starting",
        agent=config.agent_name,
        host=config.host,
        port=config.port,
    )

    # Placeholder: In production, use agentscope.rpc server
    logger.info(
        "rpc_server_ready",
        agent=config.agent_name,
        endpoint=f"{config.host}:{config.port}",
    )

    # Keep alive
    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        logger.info("rpc_server_stopped", agent=config.agent_name)


def main() -> None:
    """CLI entry point for the RPC server."""
    parser = argparse.ArgumentParser(description="ORION Agent RPC Server")
    parser.add_argument(
        "--agent",
        choices=["planner", "executor", "verifier", "supervisor"],
        required=True,
        help="Agent to launch as RPC service.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: auto per agent).",
    )

    args = parser.parse_args()
    config = RPCConfig(
        agent_name=args.agent,
        host=args.host,
        port=args.port,
    )

    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
