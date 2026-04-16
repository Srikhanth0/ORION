"""Vision tool category — screen capture + VLM analysis + computer control.

Provides async functions for screenshot capture, Qwen2.5-VL analysis
via a remote Colab server, and pyautogui-based mouse/keyboard control.

Module Contract
---------------
- **Screenshot**: Captures full screen as PNG bytes.
- **Analysis**: Sends screenshot to Colab-hosted Qwen2.5-VL via httpx.
- **Control**: Translates VLM bounding-box output to pyautogui actions.

Environment Variables
---------------------
- ``VISION_API_URL``: ngrok URL to the Colab vision server.
- ``PYAUTOGUI_FAILSAFE``: If "true", pyautogui failsafe stays enabled.

Depends On
----------
- ``httpx`` (async HTTP client)
- ``pyautogui`` (mouse/keyboard control)
- ``PIL`` (screenshot capture)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

VISION_API_URL = os.getenv("VISION_API_URL", "")
_ANALYZE_TIMEOUT = float(os.getenv("VISION_TIMEOUT_SECONDS", "60"))


async def take_screenshot() -> dict[str, Any]:
    """Capture full screen and return base64-encoded PNG.

    Returns:
        Dict with 'image_base64' (str) and 'width'/'height' (int).
    """
    from PIL import ImageGrab

    screenshot = ImageGrab.grab()
    buf = io.BytesIO()
    screenshot.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    logger.info(
        "screenshot_captured",
        size_bytes=len(img_bytes),
        width=screenshot.width,
        height=screenshot.height,
    )

    return {
        "image_base64": base64.b64encode(img_bytes).decode(),
        "width": screenshot.width,
        "height": screenshot.height,
    }


async def analyze_screen(
    prompt: str | None = None,
    image_base64: str | None = None,
) -> dict[str, Any]:
    """Analyze screen content via Qwen2.5-VL vision model.

    Takes a screenshot (or uses provided image) and sends to the
    Colab-hosted VLM for analysis.

    Args:
        prompt: Custom analysis prompt. Defaults to UI element detection.
        image_base64: Pre-captured screenshot. If None, captures fresh.

    Returns:
        Dict with 'result' (str) from the vision model.

    Raises:
        RuntimeError: If VISION_API_URL is not configured.
        httpx.HTTPStatusError: If the vision server returns an error.
    """
    import httpx

    if not VISION_API_URL:
        raise RuntimeError(
            "VISION_API_URL not set. Start the Colab vision server "
            "and set the ngrok URL in your .env file."
        )

    # Capture screenshot if not provided
    if image_base64 is None:
        screenshot = await take_screenshot()
        image_base64 = screenshot["image_base64"]

    default_prompt = (
        "Describe this screen. List all clickable UI elements with their "
        "approximate pixel coordinates as bounding boxes [x1,y1,x2,y2]."
    )

    logger.info(
        "vision_analyze_request",
        prompt_length=len(prompt or default_prompt),
        api_url=VISION_API_URL[:50],
    )

    async with httpx.AsyncClient(timeout=_ANALYZE_TIMEOUT) as client:
        resp = await client.post(
            f"{VISION_API_URL}/analyze",
            json={
                "image_base64": image_base64,
                "prompt": prompt or default_prompt,
            },
        )
        resp.raise_for_status()
        result = resp.json()

    logger.info(
        "vision_analyze_complete",
        result_length=len(result.get("result", "")),
    )

    return result  # type: ignore[no-any-return]


async def click_element(description: str) -> dict[str, Any]:
    """Find a UI element by description and click it.

    Uses the vision model to locate the element on screen,
    then uses pyautogui to perform the click.

    Args:
        description: Natural language description of the element to click.

    Returns:
        Dict with 'success' (bool) and click coordinates or error.
    """
    import pyautogui

    logger.info("vision_click_request", description=description[:100])

    result = await analyze_screen(
        prompt=(
            f"Find the UI element matching: '{description}'. "
            "Return ONLY a JSON object: "
            '{"x": <center_pixel_x>, "y": <center_pixel_y>, "found": true/false, '
            '"element": "<element description>"}'
        )
    )

    # Parse coordinates from VLM output
    text = result.get("result", "")
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        logger.warning("vision_click_no_coords", raw_output=text[:200])
        return {
            "success": False,
            "error": "No coordinates found in model response",
            "raw_output": text[:500],
        }

    try:
        coords = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "error": f"Failed to parse coordinates: {exc}",
            "raw_output": text[:500],
        }

    if not coords.get("found", False):
        return {
            "success": False,
            "error": f"Element '{description}' not found on screen",
            "element": coords.get("element", ""),
        }

    x, y = int(coords["x"]), int(coords["y"])
    pyautogui.click(x, y)

    logger.info("vision_click_executed", x=x, y=y, element=description[:50])

    return {
        "success": True,
        "clicked_at": {"x": x, "y": y},
        "element": coords.get("element", description),
    }


async def type_text(text: str, interval: float = 0.05) -> dict[str, Any]:
    """Type text at the current cursor position.

    Args:
        text: Text to type.
        interval: Seconds between keystrokes (default 0.05).

    Returns:
        Dict with 'success' (bool) and 'typed' (str).
    """
    import pyautogui

    logger.info("vision_type_text", length=len(text))
    pyautogui.write(text, interval=interval)

    return {"success": True, "typed": text}


async def press_key(key: str) -> dict[str, Any]:
    """Press a keyboard key or key combination.

    Supports single keys ('enter', 'tab', 'escape') and
    combinations ('ctrl+c', 'alt+f4', 'ctrl+shift+s').

    Args:
        key: Key or key combination (joined by '+').

    Returns:
        Dict with 'success' (bool) and 'key' (str).
    """
    import pyautogui

    keys = key.split("+")
    logger.info("vision_press_key", key=key, parts=keys)
    pyautogui.hotkey(*keys)

    return {"success": True, "key": key}


# ── Tool metadata for registry auto-discovery ───────────────────────

VISION_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "take_screenshot",
        "description": "Capture full screen as base64-encoded PNG image",
        "fn": take_screenshot,
        "is_destructive": False,
        "params_schema": {},
    },
    {
        "name": "analyze_screen",
        "description": (
            "Analyze screen content via Qwen2.5-VL vision model. "
            "Returns structured description of UI elements."
        ),
        "fn": analyze_screen,
        "is_destructive": False,
        "params_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Custom analysis prompt"},
                "image_base64": {"type": "string", "description": "Pre-captured screenshot"},
            },
        },
    },
    {
        "name": "click_element",
        "description": (
            "Find a UI element by description and click it using vision model "
            "for coordinate detection and pyautogui for click execution."
        ),
        "fn": click_element,
        "is_destructive": True,
        "params_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Element to click"},
            },
            "required": ["description"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text at current cursor position using pyautogui",
        "fn": type_text,
        "is_destructive": True,
        "params_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"},
                "interval": {"type": "number", "description": "Seconds between keystrokes"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": ("Press a keyboard key or combination (e.g. 'enter', 'ctrl+c', 'alt+f4')"),
        "fn": press_key,
        "is_destructive": True,
        "params_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key or combo like 'ctrl+c'"},
            },
            "required": ["key"],
        },
    },
]
