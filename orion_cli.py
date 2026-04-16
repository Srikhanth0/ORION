"""ORION CLI — Terminal REPL for the ORION Agent API.

Interactive command-line interface powered by rich + prompt_toolkit.
Communicates with the ORION API server at http://localhost:8080.

Usage:
    uv run orion
    python orion_cli.py
    orion                      # Run as installed command
    orion --tools             # List tools (non-interactive)
    orion --submit "task"    # Submit a task
    orion --safe-mode         # Enable sandbox mode
"""

from __future__ import annotations

import argparse

# Force UTF-8 for Rich/Console on Windows
import io
import os
import sys

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if sys.platform == "win32" and getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()


def parse_args():  # type: ignore[no-untyped-def]
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ORION CLI — Autonomous OS Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="List available tools and exit",
    )
    parser.add_argument(
        "--submit",
        type=str,
        metavar="TASK",
        help="Submit a task instruction and exit",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Run task in Podman sandbox (Linux container)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8080",
        help="ORION API server URL",
    )
    return parser.parse_args()


API_BASE = "http://localhost:8080"

BANNER = r"""
   ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗
  ██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║
  ██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║
  ██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║
  ╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║
   ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
         Autonomous OS Agent  ·  v1.0
"""

HELP_TEXT = """
[bold cyan]ORION CLI Commands[/]

  [green]<instruction>[/]      Submit a task to the agent
  [green]/stream <task_id>[/]  Stream live events from a running task
  [green]/tools[/]             List all available tools
  [green]/safe-mode[/]         Toggle Podman sandbox safe mode (runs in container)
  [green]/help[/]              Show this help message
  [green]/quit[/]              Exit the CLI
"""


def print_banner() -> None:
    """Display the ORION ASCII art banner."""
    console.print(
        Panel(
            Text(BANNER, style="bold cyan"),
            border_style="bright_blue",
            subtitle="[dim]Type /help for commands[/]",
        )
    )


def submit_task(instruction: str, safe_mode: bool = False) -> None:
    """POST a task instruction to the ORION API."""
    if safe_mode:
        instruction = (
            f"[SAFE MODE: Podman Sandbox Enabled] Execute this task ENTIRELY inside "
            f"a podman sandbox using linux mcp tools and browser ui. "
            f"Task: {instruction}"
        )
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{API_BASE}/v1/tasks",
                json={"instruction": instruction, "safe_mode": safe_mode},
            )
            resp.raise_for_status()
            data = resp.json()
            task_id = data.get("task_id", "unknown")
            console.print(f"  [green]✓[/] Task submitted: [bold]{task_id}[/]")
            console.print(f"  [dim]Stream with:[/] /stream {task_id}")
    except httpx.ConnectError:
        console.print(
            "  [red]✗[/] Cannot connect to ORION API at " f"{API_BASE}. Is the server running?"
        )
    except httpx.HTTPStatusError as exc:
        console.print(
            f"  [red]✗[/] API error: {exc.response.status_code} — " f"{exc.response.text[:200]}"
        )
    except Exception as exc:
        console.print(f"  [red]✗[/] Error: {exc}")


def stream_task(task_id: str) -> None:
    """Stream SSE events from a running task."""
    url = f"{API_BASE}/v1/tasks/{task_id}/stream"
    console.print(f"  [dim]Streaming from {url}...[/]")

    try:
        with httpx.Client(timeout=None) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            with Live(console=console, refresh_per_second=4) as live:
                buffer = ""
                for chunk in response.iter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines[-1]  # keep incomplete line

                    for line in lines[:-1]:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            live.update(
                                Text(
                                    f"⚡ {event_type}",
                                    style="bold yellow",
                                )
                            )
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            console.print(f"  [cyan]{data}[/]")
                        if line == "event: done":
                            console.print("  [green]✓ Stream complete[/]")
                            return

    except httpx.ConnectError:
        console.print("  [red]✗[/] Cannot connect to ORION API.")
    except httpx.HTTPStatusError as exc:
        console.print(f"  [red]✗[/] Stream error: {exc.response.status_code}")
    except Exception as exc:
        console.print(f"  [red]✗[/] Error: {exc}")


def list_tools() -> None:
    """GET available tools and display as a table."""
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{API_BASE}/v1/tools")
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, list):
            tools = data
        elif isinstance(data, dict):
            tools = data.get("tools", [])
        else:
            tools = []

        if not tools:
            console.print("  [yellow]No tools available yet. Discovering...[/]")
            return

        table = Table(
            title=f"ORION Tools ({len(tools)} available)",
            border_style="bright_blue",
            show_lines=True,
        )
        table.add_column("Name", style="green", no_wrap=True)
        table.add_column("Category", style="cyan")
        table.add_column("Description", style="white")

        for tool in tools:
            if isinstance(tool, dict):
                table.add_row(
                    tool.get("name", "—"),
                    tool.get("category", "—"),
                    tool.get("description", "—")[:80],
                )

        console.print(table)

    except httpx.ConnectError:
        console.print("  [red]✗[/] Cannot connect to ORION API.")
    except Exception as exc:
        console.print(f"  [red]✗[/] Error: {exc}")


def ensure_server_running(api_base: str) -> None:
    """Ensure the API server is running, start it if not."""
    try:
        with httpx.Client(timeout=1.0) as client:
            client.get(f"{api_base}/docs")
        return
    except httpx.RequestError:
        pass

    console.print("  [yellow]API server not running. Starting it in the background...[/]")
    import subprocess
    import time

    server_script = os.path.join(os.path.dirname(__file__), "run_server.py")
    subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    with console.status("  [yellow]Waiting for API server to start...[/]"):
        for _ in range(30):
            try:
                with httpx.Client(timeout=1.0) as client:
                    client.get(f"{api_base}/docs")
                console.print("  [green]✓ API server started.[/]")
                return
            except httpx.RequestError:
                time.sleep(1)

    console.print("  [red]✗[/] API server failed to start within 30 seconds.")


def main() -> None:
    """ORION CLI entry point."""
    args = parse_args()

    # Support --server via env override
    global API_BASE
    if args.server:
        API_BASE = args.server

    ensure_server_running(API_BASE)

    # Non-interactive modes
    if args.tools:
        print_banner()
        list_tools()
        return

    if args.submit:
        print_banner()
        submit_task(args.submit, safe_mode=args.safe_mode)
        return

    # Interactive REPL mode
    print_banner()

    session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
    )
    safe_mode = args.safe_mode
    if args.safe_mode:
        console.print("  [bold green]✓[/] Safe Mode is enabled (Podman sandbox)")

    while True:
        try:
            prompt_html = "<ansibrightblue>[ORION]</ansibrightblue> "
            if safe_mode:
                prompt_html += "<ansienvironment>[SAFE]</ansienvironment> "
            prompt_html += "› "
            user_input = session.prompt(HTML(prompt_html)).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Goodbye.[/]")
            break

        if not user_input:
            continue

        # ── Command routing ──
        if user_input == "/quit":
            console.print("  [dim]Goodbye.[/]")
            break

        elif user_input == "/help":
            console.print(HELP_TEXT)

        elif user_input == "/tools":
            list_tools()

        elif user_input.startswith("/stream"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("  [yellow]Usage:[/] /stream <task_id>")
            else:
                stream_task(parts[1].strip())

        elif user_input == "/safe-mode":
            safe_mode = not safe_mode
            status = "enabled" if safe_mode else "disabled"
            console.print(f"  [bold green]✓[/] Safe Mode is now [bold]{status}[/]")

        elif user_input.startswith("/"):
            console.print(
                f"  [yellow]Unknown command:[/] {user_input}. " "Type /help for available commands."
            )

        else:
            # Treat as a task instruction
            submit_task(user_input, safe_mode=safe_mode)


if __name__ == "__main__":
    main()
