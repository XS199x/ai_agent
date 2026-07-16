"""MCP Server 配置描述。

每个 MCP server 用一个 MCPServerConfig 描述：
- 连接方式：stdio（启动子进程，通过 stdin/stdout 通信）
- 命令、参数、环境变量、工作目录
- 超时控制
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MCPServerConfig:
    """单个 MCP Server 的启动与连接配置。"""

    name: str
    """逻辑名称，用于错误日志和工具名前缀（可选）。"""

    command: str
    """可执行命令。例如：python / npx / uvicorn / .\\tools\\server.exe"""

    args: Optional[List[str]] = None
    """命令参数。例如 ["-y", "@modelcontextprotocol/server-filesystem", "/home"]"""

    env: Optional[Dict[str, str]] = None
    """额外注入的环境变量（会合并到当前进程 env，同名覆盖）。"""

    cwd: Optional[str] = None
    """工作目录。"""

    timeout_seconds: int = 60
    """单个 tools/call 请求的最大等待秒数。"""

    start_timeout_seconds: int = 15
    """启动进程 + 完成 initialize 握手的最大等待秒数。"""

    tool_name_prefix: Optional[str] = None
    """工具名前缀。例如填 'fs_' 则 server 暴露的 read_file 会变 fs_read_file。
    留空则使用 server name + '_' 前缀。若显式传入 '' 字符串则无前缀（可能重名）。"""

    def full_env(self) -> Dict[str, str]:
        base = dict(os.environ)
        if self.env:
            base.update(self.env)
        return base

    def popen_cmd(self) -> Tuple[str, ...]:
        parts = [self.command]
        if self.args:
            parts.extend(self.args)
        return tuple(parts)

    def resolved_prefix(self) -> str:
        if self.tool_name_prefix is not None:
            return self.tool_name_prefix
        return f"{self.name}_"
