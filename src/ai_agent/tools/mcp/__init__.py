"""MCP (Model Context Protocol) 工具集成包。

通过 stdio 方式连接 MCP Server，将其提供的工具动态适配为 BaseTool，
上层 Planner / Executor / Runtime 完全感知不到协议差异。

典型用法：
    from ai_agent.tools.mcp import MCPToolProvider, MCPServerConfig

    configs = [
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "C:/workspace"],
        ),
    ]
    provider = await MCPToolProvider.from_configs(configs)
    # 然后把 provider 注入到 CompositeToolProvider
"""

from ai_agent.tools.mcp.config import MCPServerConfig
from ai_agent.tools.mcp.adapter import MCPTool, MCPToolProvider
from ai_agent.tools.mcp.client import MCPClient

__all__ = [
    "MCPServerConfig",
    "MCPClient",
    "MCPTool",
    "MCPToolProvider",
]
