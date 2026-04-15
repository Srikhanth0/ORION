"""Unit tests for vision and computer control tools."""
from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.tools.categories import vision_tools
from orion.tools.registry import ToolCategory, ToolRegistry


class TestVisionTools:
    """Tests for vision_tools functions."""

    @pytest.fixture
    def mock_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock environment variables for vision API."""
        monkeypatch.setenv("VISION_API_URL", "http://test-vision.com")
        monkeypatch.setattr(vision_tools, "VISION_API_URL", "http://test-vision.com")

    @pytest.mark.asyncio
    async def test_take_screenshot(self, mocker: Any) -> None:
        """take_screenshot returns base64 png and dimensions."""
        mock_img = MagicMock()
        mock_img.width = 1920
        mock_img.height = 1080

        # Mock ImageGrab.grab()
        mock_grab = mocker.patch("PIL.ImageGrab.grab", return_value=mock_img)

        # Mock BytesIO.getvalue() indirectly by mocking image save
        def mock_save(buf: Any, format: str) -> None:
            buf.write(b"fake_png_data")

        mock_img.save.side_effect = mock_save

        result = await vision_tools.take_screenshot()

        assert "image_base64" in result
        assert result["width"] == 1920
        assert result["height"] == 1080

        decoded = base64.b64decode(result["image_base64"])
        assert decoded == b"fake_png_data"
        mock_grab.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_screen(self, mock_env: None, mocker: Any) -> None:
        """analyze_screen uses httpx to post to Colab server."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "A button with text Submit"}
        mock_response.raise_for_status = MagicMock()

        # Mock httpx AsyncClient
        mock_post = AsyncMock(return_value=mock_response)

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            post = mock_post

        mocker.patch("httpx.AsyncClient", return_value=MockClient())

        result = await vision_tools.analyze_screen(
            prompt="Find Submit button",
            image_base64="fake_b64",
        )

        assert result["result"] == "A button with text Submit"
        mock_post.assert_awaited_once_with(
            "http://test-vision.com/analyze",
            json={
                "image_base64": "fake_b64",
                "prompt": "Find Submit button",
            },
        )

    @pytest.mark.asyncio
    async def test_analyze_screen_missing_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """analyze_screen raises if VISION_API_URL is missing."""
        monkeypatch.delenv("VISION_API_URL", raising=False)
        vision_tools.VISION_API_URL = ""

        with pytest.raises(RuntimeError, match="VISION_API_URL not set"):
            await vision_tools.analyze_screen(image_base64="asdf")

    @pytest.mark.asyncio
    async def test_click_element_success(self, mock_env: None, mocker: Any) -> None:
        """click_element parses coords and clicks."""
        mock_pyautogui = mocker.patch("pyautogui.click")

        # Mock analyze_screen
        mock_analyze = mocker.patch(
            "orion.tools.categories.vision_tools.analyze_screen",
            new_callable=AsyncMock,
        )
        mock_analyze.return_value = {
            "result": 'Here is the element: {"x": 100, "y": 200, "found": true, "element": "Submit"}'  # noqa: E501
        }

        result = await vision_tools.click_element("Submit button")

        assert result["success"] is True
        assert result["clicked_at"] == {"x": 100, "y": 200}
        mock_pyautogui.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_click_element_not_found(self, mock_env: None, mocker: Any) -> None:
        """click_element returns error if not found."""
        mock_pyautogui = mocker.patch("pyautogui.click")

        mock_analyze = mocker.patch(
            "orion.tools.categories.vision_tools.analyze_screen",
            new_callable=AsyncMock,
        )
        mock_analyze.return_value = {
            "result": '{"x": 0, "y": 0, "found": false}'
        }

        result = await vision_tools.click_element("Submit button")

        assert result["success"] is False
        assert "not found" in result["error"]
        mock_pyautogui.assert_not_called()

    @pytest.mark.asyncio
    async def test_type_text(self, mocker: Any) -> None:
        """type_text uses pyautogui.write."""
        mock_write = mocker.patch("pyautogui.write")

        result = await vision_tools.type_text("hello world")

        assert result["success"] is True
        assert result["typed"] == "hello world"
        mock_write.assert_called_once_with("hello world", interval=0.05)

    @pytest.mark.asyncio
    async def test_press_key(self, mocker: Any) -> None:
        """press_key uses pyautogui.hotkey."""
        mock_hotkey = mocker.patch("pyautogui.hotkey")

        result = await vision_tools.press_key("ctrl+shift+c")

        assert result["success"] is True
        mock_hotkey.assert_called_once_with("ctrl", "shift", "c")


class TestRegistryNativeTools:
    """Tests for native ToolRegistry capabilities."""

    def test_register_native(self) -> None:
        """Registry can register native Python tools."""
        registry = ToolRegistry()

        async def my_tool(x: int) -> int:
            return x * 2

        registry.register_native(
            name="my_vision_tool",
            fn=my_tool,
            description="Does stuff",
            category=ToolCategory.VISION,
        )

        assert registry.is_native_tool("my_vision_tool")

        wrapper = registry.get("my_vision_tool")
        assert wrapper.category == ToolCategory.VISION
        assert wrapper.server_category == "native"

    @pytest.mark.asyncio
    async def test_call_native(self) -> None:
        """Registry can invoke native Python tools."""
        registry = ToolRegistry()

        async def add(a: int, b: int) -> int:
            return a + b

        registry.register_native(
            name="add",
            fn=add,
            category=ToolCategory.SYSTEM,
        )

        result = await registry.call_native("add", {"a": 2, "b": 3})
        assert result == 5

    def test_register_vision_tools_convenience(self, mocker: Any) -> None:
        """register_vision_tools loads definitions from the module."""
        registry = ToolRegistry()
        registry.register_vision_tools()

        assert registry.is_native_tool("take_screenshot")
        assert registry.is_native_tool("analyze_screen")
        assert registry.is_native_tool("click_element")

        wrapper = registry.get("click_element")
        assert wrapper.is_destructive is True
        assert wrapper.category == ToolCategory.VISION
