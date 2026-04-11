"""Browser tool category — typed wrappers for Composio browser actions.

Depends On
----------
- ``orion.tools.mcp_client`` (MCPClient, ToolResult)
"""
from __future__ import annotations

from orion.tools.mcp_client import MCPClient, ToolResult


class BrowserTools:
    """Typed wrappers for browser Composio actions.

    Args:
        client: MCPClient instance for invocation.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def navigate_url(
        self,
        url: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Navigate to a URL.

        Args:
            url: Target URL.
            task_id: Parent task ID.

        Returns:
            ToolResult with page load status.
        """
        return await self._client.invoke(
            "BROWSER_NAVIGATE_URL",
            {"url": url},
            task_id=task_id,
        )

    async def click_element(
        self,
        selector: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Click an element by CSS selector.

        Args:
            selector: CSS selector for the element.
            task_id: Parent task ID.

        Returns:
            ToolResult with click confirmation.
        """
        return await self._client.invoke(
            "BROWSER_CLICK_ELEMENT",
            {"selector": selector},
            task_id=task_id,
        )

    async def get_text(
        self,
        selector: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Get text content of an element.

        Args:
            selector: CSS selector for the element.
            task_id: Parent task ID.

        Returns:
            ToolResult with element text.
        """
        return await self._client.invoke(
            "BROWSER_GET_TEXT",
            {"selector": selector},
            task_id=task_id,
        )

    async def fill_form(
        self,
        selector: str,
        value: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Fill a form field.

        Args:
            selector: CSS selector for the input.
            value: Value to fill.
            task_id: Parent task ID.

        Returns:
            ToolResult with fill confirmation.
        """
        return await self._client.invoke(
            "BROWSER_FILL_FORM",
            {"selector": selector, "value": value},
            task_id=task_id,
        )

    async def take_screenshot(
        self,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Take a screenshot of the current page.

        Args:
            task_id: Parent task ID.

        Returns:
            ToolResult with screenshot data/path.
        """
        return await self._client.invoke(
            "BROWSER_SCREENSHOT",
            {},
            task_id=task_id,
        )

    async def get_page_source(
        self,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Get the current page HTML source.

        Args:
            task_id: Parent task ID.

        Returns:
            ToolResult with HTML source.
        """
        return await self._client.invoke(
            "BROWSER_GET_PAGE_SOURCE",
            {},
            task_id=task_id,
        )

    async def wait_for_element(
        self,
        selector: str,
        timeout: float = 10.0,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Wait for an element to appear on the page.

        Args:
            selector: CSS selector for the element.
            timeout: Wait timeout in seconds.
            task_id: Parent task ID.

        Returns:
            ToolResult with element presence status.
        """
        return await self._client.invoke(
            "BROWSER_WAIT_FOR_ELEMENT",
            {"selector": selector, "timeout": timeout},
            task_id=task_id,
        )
