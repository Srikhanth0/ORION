"""GitHub tool category — typed wrappers for MCP GitHub actions.

Each method maps to a specific MCP action name and provides
typed parameters and docstrings for IDE/LLM clarity.

Depends On
----------
- ``orion.tools.mcp_client`` (MCPClient, ToolResult)
"""
from __future__ import annotations

from orion.tools.mcp_client import MCPClient, ToolResult


class GitHubTools:
    """Typed wrappers for GitHub MCP actions.

    Args:
        client: MCPClient instance for invocation.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Create a GitHub issue.

        Args:
            repo: Repository in owner/name format.
            title: Issue title.
            body: Issue body markdown.
            labels: Optional label names.
            task_id: Parent task ID.

        Returns:
            ToolResult with issue URL.
        """
        return await self._client.invoke(
            "GITHUB_CREATE_ISSUE",
            {
                "repo": repo,
                "title": title,
                "body": body,
                "labels": labels or [],
            },
            task_id=task_id,
        )

    async def push_files(
        self,
        repo: str,
        branch: str,
        files: dict[str, str],
        commit_message: str = "Auto-commit by ORION",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Push files to a GitHub repository.

        Args:
            repo: Repository in owner/name format.
            branch: Target branch name.
            files: Dict of filepath → content.
            commit_message: Commit message.
            task_id: Parent task ID.

        Returns:
            ToolResult with commit SHA.
        """
        return await self._client.invoke(
            "GITHUB_PUSH_FILES",
            {
                "repo": repo,
                "branch": branch,
                "files": files,
                "commit_message": commit_message,
            },
            task_id=task_id,
        )

    async def list_prs(
        self,
        repo: str,
        state: str = "open",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """List pull requests for a repository.

        Args:
            repo: Repository in owner/name format.
            state: PR state (open, closed, all).
            task_id: Parent task ID.

        Returns:
            ToolResult with PR list.
        """
        return await self._client.invoke(
            "GITHUB_LIST_PRS",
            {"repo": repo, "state": state},
            task_id=task_id,
        )

    async def get_file_content(
        self,
        repo: str,
        path: str,
        ref: str = "main",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Get file content from a repository.

        Args:
            repo: Repository in owner/name format.
            path: File path within the repo.
            ref: Git ref (branch, tag, SHA).
            task_id: Parent task ID.

        Returns:
            ToolResult with file content.
        """
        return await self._client.invoke(
            "GITHUB_GET_FILE_CONTENT",
            {"repo": repo, "path": path, "ref": ref},
            task_id=task_id,
        )

    async def create_branch(
        self,
        repo: str,
        branch: str,
        from_ref: str = "main",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Create a new branch.

        Args:
            repo: Repository in owner/name format.
            branch: New branch name.
            from_ref: Base ref to branch from.
            task_id: Parent task ID.

        Returns:
            ToolResult with branch info.
        """
        return await self._client.invoke(
            "GITHUB_CREATE_BRANCH",
            {
                "repo": repo,
                "branch": branch,
                "from_ref": from_ref,
            },
            task_id=task_id,
        )

    async def merge_pr(
        self,
        repo: str,
        pr_number: int,
        merge_method: str = "squash",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Merge a pull request.

        Args:
            repo: Repository in owner/name format.
            pr_number: PR number to merge.
            merge_method: Merge method (merge, squash, rebase).
            task_id: Parent task ID.

        Returns:
            ToolResult with merge result.
        """
        return await self._client.invoke(
            "GITHUB_MERGE_PR",
            {
                "repo": repo,
                "pr_number": pr_number,
                "merge_method": merge_method,
            },
            task_id=task_id,
        )

    async def run_workflow(
        self,
        repo: str,
        workflow_id: str,
        ref: str = "main",
        inputs: dict[str, str] | None = None,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Trigger a GitHub Actions workflow.

        Args:
            repo: Repository in owner/name format.
            workflow_id: Workflow file name or ID.
            ref: Git ref to run against.
            inputs: Workflow input parameters.
            task_id: Parent task ID.

        Returns:
            ToolResult with workflow run info.
        """
        return await self._client.invoke(
            "GITHUB_RUN_WORKFLOW",
            {
                "repo": repo,
                "workflow_id": workflow_id,
                "ref": ref,
                "inputs": inputs or {},
            },
            task_id=task_id,
        )
