"""
Async MCP bridge to Blender via BlenderMCP.

Flow: our code → stdio → `uvx blender-mcp` → TCP 9876 → BlenderMCP Blender addon

Prerequisites:
  brew install uv                   # for `uvx blender-mcp`
  pip install "anthropic[mcp]"      # already in requirements
  BlenderMCP addon installed + running in Blender (N-panel → BlenderMCP → Connect)
"""

from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class BlenderBridgeError(Exception):
    pass


class BlenderBridge:
    """Async context manager that talks to Blender via BlenderMCP MCP tools."""

    def __init__(self):
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        params = StdioServerParameters(command="uvx", args=["blender-mcp"])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        tools = await self._session.list_tools()
        print(f"  [Bridge] Connected via BlenderMCP — tools: {[t.name for t in tools.tools]}")
        return self

    async def __aexit__(self, *args):
        if self._stack:
            await self._stack.aclose()

    async def run(self, code: str) -> str:
        """Execute bpy Python code in Blender and return the result string.

        BlenderMCP captures print() output (not the `result` variable), so we
        append a print() call automatically — agents can keep writing `result = ...`.
        """
        wrapped = (
            code.rstrip()
            + "\n\ntry:\n    print(str(result))\nexcept NameError:\n    print('done')"
        )
        try:
            resp = await self._session.call_tool("execute_blender_code", {"code": wrapped})
            for item in resp.content:
                if hasattr(item, "text"):
                    text: str = item.text
                    prefix = "Code executed successfully: "
                    if text.startswith(prefix):
                        return text[len(prefix):].rstrip("\n")
                    return text.rstrip("\n")
            return "done"
        except Exception as e:
            raise BlenderBridgeError(str(e)) from e

    async def render_preview(self) -> str | None:
        """Render the scene; return base64-encoded PNG or None on failure."""
        try:
            resp = await self._session.call_tool("render_preview", {})
            for item in resp.content:
                if hasattr(item, "data"):
                    return item.data
                if hasattr(item, "text") and item.text:
                    return item.text
            return None
        except Exception as e:
            print(f"  [Bridge] render_preview failed: {e}")
            return None

    async def get_viewport_screenshot(self) -> str | None:
        """Take a viewport screenshot; return base64-encoded PNG or None on failure."""
        try:
            resp = await self._session.call_tool("get_viewport_screenshot", {})
            for item in resp.content:
                if hasattr(item, "data"):
                    return item.data
                if hasattr(item, "text") and item.text:
                    return item.text
            return None
        except Exception as e:
            print(f"  [Bridge] get_viewport_screenshot failed: {e}")
            return None
