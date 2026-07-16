"""MCP → BaseTool / ToolProvider 适配层。

核心思想：Adapter 模式。

一个 MCP stdio server 会动态提供 N 个工具，我们把每个工具都包装成一个
`MCPTool(BaseTool)`，然后 `MCPToolProvider(ToolProvider)` 聚合所有连接。
上层 Planner / Executor 仍然只看统一的 ToolProvider 接口，协议无感。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_agent.core.provider import ToolProvider
from ai_agent.models.action import Action, tool_action
from ai_agent.tools.base import BaseTool
from ai_agent.tools.mcp.client import MCPClient, MCPError, MCPToolDefinition
from ai_agent.tools.mcp.config import MCPServerConfig


class MCPTool(BaseTool):
    """把 MCP server 提供的一个工具适配为 BaseTool。"""

    def __init__(
        self,
        client: MCPClient,
        prefix: str,
        definition: MCPToolDefinition,
    ):
        self._client = client
        self._prefix = prefix
        self._def = definition
        self._prefixed_name = prefix + definition.name

    @property
    def name(self) -> str:
        return self._prefixed_name

    @property
    def description(self) -> str:
        desc = self._def.description.strip()
        if not desc:
            return f"通过 MCP server({self._client.config.name}) 提供的工具：{self._def.name}"
        return desc

    @property
    def args_schema(self) -> Dict[str, Any]:
        return dict(self._def.input_schema)

    @property
    def timeout(self) -> float:
        return float(self._client.config.timeout_seconds)

    def run(self, args: Dict[str, Any]) -> str:
        """同步调用 MCP 工具（Executor 是同步 run 接口，内部跑事件循环等待）。

        注意：Executor.execute() 内部会捕获异常并包装为 ToolResult，
        这里让 asyncio.run() 抛出即可，上层统一处理。
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._async_call(args))

        # 已经在事件循环中（例如 run_stream 场景），用 run_coroutine_threadsafe
        # 不现实（我们只在单线程 loop 模型下），改用同步桥：
        # 抛给上层转 ToolResult 错误，避免死锁风险。
        # 真正的异步化改造可以在未来把 BaseTool.run 也改成 async。
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    lambda: asyncio.run(self._async_call(args))
                )
                return future.result(timeout=self.timeout + 5)
        except MCPError:
            raise
        except Exception as e:
            raise MCPError(f"调用 MCP 工具 {self.name!r} 失败: {e}") from e

    async def _async_call(self, args: Dict[str, Any]) -> str:
        return await self._client.call_tool(self._def.name, args)


class MCPToolProvider(ToolProvider):
    """聚合多个 MCP server，对外表现为统一的 ToolProvider。

    生命周期：
        provider = MCPToolProvider.from_configs(configs)
        await provider.setup()         # 启动所有子进程 + initialize + 拉取工具列表
        provider.get_tool('fs_read')   # 直接使用
        await provider.teardown()      # 关闭所有连接
    """

    def __init__(
        self,
        configs: List[MCPServerConfig],
        clients: Optional[List[MCPClient]] = None,
        tools: Optional[List[MCPTool]] = None,
    ):
        self._configs = list(configs)
        self._clients: List[MCPClient] = list(clients) if clients else [
            MCPClient(cfg) for cfg in self._configs
        ]
        self._tools: List[MCPTool] = list(tools) if tools else []
        self._tools_by_name: Dict[str, MCPTool] = {}

    # ---------- 便捷构造 ----------

    @classmethod
    def from_configs(cls, configs: List[MCPServerConfig]) -> "MCPToolProvider":
        return cls(configs=list(configs))

    @classmethod
    def from_single(cls, config: MCPServerConfig) -> "MCPToolProvider":
        return cls(configs=[config])

    # ---------- 生命周期 ----------

    async def setup(self) -> None:
        errors: List[str] = []
        for client in self._clients:
            try:
                await client.setup()
                prefix = client.config.resolved_prefix()
                definitions = await client.list_tools()
                for d in definitions:
                    if not d.name:
                        continue
                    wrapped = MCPTool(client, prefix, d)
                    if wrapped.name in self._tools_by_name:
                        errors.append(
                            f"工具名冲突：{wrapped.name!r} 同时由 "
                            f"{self._tools_by_name[wrapped.name]._client.config.name} 和 "
                            f"{client.config.name} 提供，后者被忽略"
                        )
                        continue
                    self._tools.append(wrapped)
                    self._tools_by_name[wrapped.name] = wrapped
            except Exception as e:
                errors.append(f"MCP server {client.config.name!r} 初始化失败: {e}")
        if errors and not self._tools:
            raise MCPError("; ".join(errors))

    async def teardown(self) -> None:
        self._tools_by_name.clear()
        self._tools.clear()
        for client in self._clients:
            try:
                await client.teardown()
            except Exception:
                pass

    async def health(self) -> bool:
        if not self._clients:
            return False
        return any(c.initialized for c in self._clients)

    # ---------- ToolProvider 接口 ----------

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools_by_name.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._tools]

    def as_actions(self) -> List[Action]:
        return [
            tool_action(
                name=t.name,
                args={k: "..." for k in (t.args_schema.get("properties") or {}).keys()},
            )
            for t in self._tools
        ]
