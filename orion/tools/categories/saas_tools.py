"""SaaS tool category — typed wrappers for MCP SaaS actions.

Depends On
----------
- ``orion.tools.mcp_client`` (MCPClient, ToolResult)
"""

from __future__ import annotations

from orion.tools.mcp_client import MCPClient, ToolResult


class SaaSTools:
    """Typed wrappers for SaaS MCP actions.

    Args:
        client: MCPClient instance for invocation.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def send_slack_msg(
        self,
        channel: str,
        message: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Send a Slack message.

        Args:
            channel: Slack channel name or ID.
            message: Message text.
            task_id: Parent task ID.

        Returns:
            ToolResult with message confirmation.
        """
        return await self._client.invoke(
            "SLACK_SEND_MESSAGE",
            {"channel": channel, "message": message},
            task_id=task_id,
        )

    async def create_linear_issue(
        self,
        title: str,
        description: str = "",
        team_id: str | None = None,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Create a Linear issue.

        Args:
            title: Issue title.
            description: Issue description.
            team_id: Linear team ID.
            task_id: Parent task ID.

        Returns:
            ToolResult with issue URL.
        """
        return await self._client.invoke(
            "LINEAR_CREATE_ISSUE",
            {
                "title": title,
                "description": description,
                "team_id": team_id or "",
            },
            task_id=task_id,
        )

    async def add_notion_page(
        self,
        parent_id: str,
        title: str,
        content: str = "",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Add a Notion page.

        Args:
            parent_id: Parent page/database ID.
            title: Page title.
            content: Page content markdown.
            task_id: Parent task ID.

        Returns:
            ToolResult with page URL.
        """
        return await self._client.invoke(
            "NOTION_ADD_PAGE",
            {
                "parent_id": parent_id,
                "title": title,
                "content": content,
            },
            task_id=task_id,
        )

    async def send_gmail(
        self,
        to: str,
        subject: str,
        body: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Send a Gmail email.

        Args:
            to: Recipient email address.
            subject: Email subject.
            body: Email body text.
            task_id: Parent task ID.

        Returns:
            ToolResult with send confirmation.
        """
        return await self._client.invoke(
            "GMAIL_SEND_EMAIL",
            {"to": to, "subject": subject, "body": body},
            task_id=task_id,
        )

    async def create_calendar_event(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Create a Google Calendar event.

        Args:
            title: Event title.
            start_time: ISO 8601 start time.
            end_time: ISO 8601 end time.
            description: Event description.
            task_id: Parent task ID.

        Returns:
            ToolResult with event URL.
        """
        return await self._client.invoke(
            "CALENDAR_CREATE_EVENT",
            {
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
            },
            task_id=task_id,
        )

    async def search_notion(
        self,
        query: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Search Notion pages.

        Args:
            query: Search query string.
            task_id: Parent task ID.

        Returns:
            ToolResult with search results.
        """
        return await self._client.invoke(
            "NOTION_SEARCH",
            {"query": query},
            task_id=task_id,
        )
