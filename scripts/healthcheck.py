#!/usr/bin/env python3
"""ORION healthcheck — verify all system dependencies.

Checks:
  1. API server is responding
  2. Qdrant is reachable
  3. Tool registry is loaded
  4. Prometheus metrics endpoint is up

Usage:
    python scripts/healthcheck.py [--url http://localhost:8080]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request


def check_endpoint(
    url: str, name: str, timeout: float = 5.0
) -> bool:
    """Check if an HTTP endpoint returns 200.

    Args:
        url: Endpoint URL.
        name: Human-readable name.
        timeout: Request timeout.

    Returns:
        True if healthy.
    """
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode()
            status = resp.status
            print(f"  ✓ {name}: {status} — {data[:100]}")
            return status == 200
    except Exception as exc:
        print(f"  ✗ {name}: {exc}")
        return False


def main() -> int:
    """Run all health checks.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="ORION system health check"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the ORION API",
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print("═══ ORION Health Check ═══")
    print()

    results = []

    # 1. API liveness
    results.append(
        check_endpoint(f"{base}/health", "API Health")
    )

    # 2. API readiness
    results.append(
        check_endpoint(f"{base}/ready", "API Ready")
    )

    # 3. Metrics endpoint
    results.append(
        check_endpoint(
            "http://localhost:9091/metrics",
            "Prometheus Metrics",
        )
    )

    # 4. Qdrant direct
    results.append(
        check_endpoint(
            "http://localhost:6333/healthz",
            "Qdrant",
        )
    )

    print()
    passed = sum(results)
    total = len(results)
    print(f"  Result: {passed}/{total} checks passed")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
