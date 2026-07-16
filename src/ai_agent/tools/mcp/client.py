"""MCP stdio 客户端：JSON-RPC 2.0 over stdin/stdout 子进程。

协议要点（LSP 相同分帧方式）：
  - 发送：Content-Length: <bytes>\r\n\r\n<json_bytes>
  - 接收：按同样分帧格式解析，用 Content-Length 精确读取字节数
  - 请求/响应：id 匹配，后台协程持续读 stdout 并唤醒对应 Future

实现侧重：
- 轻量无三方依赖（只用 asyncio / json 标准库）
- setup / teardown 生命周期对齐 ToolProvider 接口
- 请求超时隔离
- 错误标准化（超时、握手失败、工具不存在 → 上层拿到明确错误信息）
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from ai_agent.tools.mcp.config import MCPServerConfig

_MCP_JSONRPC = "2.0"
_HEADERS_TERMINATOR = b"\r\n\r\n"
_CONTENT_LENGTH_HEADER = "content-length"


class MCPError(Exception):
    """MCP 调用错误的统一异常基类。"""

    def __init__(self, message: str, code: Optional[int] = None, raw: Any = None):
        super().__init__(message)
        self.code = code
        self.raw = raw


@dataclass
class MCPToolDefinition:
    """MCP tools/list 返回的单个工具定义（适配后的简化版）。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    raw: Dict[str, Any]


class MCPClient:
    """连接单个 MCP stdio server 的客户端。

    典型生命周期：
        client = MCPClient(config)
        await client.setup()       # 启动子进程 + initialize + initialized
        tools = await client.list_tools()
        out = await client.call_tool("tool_name", {"arg": 1})
        await client.teardown()    # 关 stdin, 等进程退出，杀进程
    """

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._request_id: int = 0
        self._pending: Dict[Any, asyncio.Future] = {}
        self._initialized: bool = False
        self._closed: bool = False

    # ---------- 生命周期 ----------

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    @property
    def initialized(self) -> bool:
        return self._initialized and not self._closed

    async def setup(self) -> None:
        """启动子进程并完成 initialize 握手。"""
        if self._closed:
            raise MCPError("client 已关闭，无法重新 setup")
        if self._proc is not None:
            return

        self._proc = await asyncio.create_subprocess_exec(
            *self._config.popen_cmd(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._config.full_env(),
            cwd=self._config.cwd,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())

        try:
            init_result = await asyncio.wait_for(
                self._request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "ai_agent", "version": "0.1.0"},
                    },
                ),
                timeout=self._config.start_timeout_seconds,
            )
            await self._notify("notifications/initialized", params={})
            # notifications/initialized 是通知（无需等待响应）
            # 简单再 ping 一下确认 server 已进入 ready 状态
            try:
                await asyncio.wait_for(
                    self._request("ping", {}),
                    timeout=5,
                )
            except Exception:
                pass  # 有的 server 不实现 ping，忽略
            self._initialized = True
        except asyncio.TimeoutError as e:
            await self.teardown()
            raise MCPError(
                f"MCP server {self._config.name!r} initialize 超时 "
                f"({self._config.start_timeout_seconds}s)"
            ) from e
        except Exception as e:
            await self.teardown()
            if isinstance(e, MCPError):
                raise
            raise MCPError(f"MCP server {self._config.name!r} 启动失败: {e}") from e

    async def teardown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._initialized = False

        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(MCPError("client closed"))
        self._pending.clear()

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()

        proc = self._proc
        self._proc = None
        if proc is None:
            return

        try:
            if proc.stdin and not proc.stdin.is_closing():
                try:
                    proc.stdin.close()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass

    # ---------- 业务方法 ----------

    async def list_tools(self) -> List[MCPToolDefinition]:
        if not self._initialized:
            raise MCPError("client 尚未 initialize，无法 list_tools")
        payload = await asyncio.wait_for(
            self._request("tools/list", {}),
            timeout=self._config.start_timeout_seconds,
        )
        raw_tools = payload.get("tools", []) if isinstance(payload, dict) else []
        result: List[MCPToolDefinition] = []
        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            schema = raw.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}, "required": []}
            # MCP inputSchema 顶层常常是 {"type":"object","properties":{...},...}
            # 有些 server 只给 properties，这里兜底
            if "type" not in schema:
                schema = {"type": "object", **schema}
            result.append(MCPToolDefinition(
                name=str(raw.get("name") or ""),
                description=str(raw.get("description") or ""),
                input_schema=schema,
                raw=raw,
            ))
        return result

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具，返回给 LLM 看的纯文本结果。

        MCP content 可能是多种 mediaType，这里统一拼接成多段文本，
        结构简单、稳定、好被 LLM 解析。
        """
        if not self._initialized:
            raise MCPError("client 尚未 initialize，无法 call_tool")
        payload = await asyncio.wait_for(
            self._request("tools/call", {"name": name, "arguments": arguments or {}}),
            timeout=self._config.timeout_seconds,
        )
        return self._flatten_content(payload)

    # ---------- 内部：JSON-RPC + 分帧 ----------

    def _flatten_content(self, payload: Any) -> str:
        """把 MCP tools/call 返回的 {content: [...]} / isError 展平成文本。"""
        if not isinstance(payload, dict):
            return str(payload)
        if payload.get("isError"):
            prefix = "[工具返回错误] "
        else:
            prefix = ""
        parts: List[str] = []
        contents = payload.get("content") or []
        if not isinstance(contents, list):
            return prefix + str(contents)
        for item in contents:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            mtype = str(item.get("type") or "")
            if mtype == "text":
                parts.append(str(item.get("text") or ""))
            elif mtype == "image":
                parts.append(f"[image mime={item.get('mimeType','?')} data_length={len(str(item.get('data','')))}]")
            elif mtype == "resource":
                parts.append(f"[resource uri={item.get('resource',{}).get('uri','?')}]")
            else:
                parts.append(str(item))
        text = "\n".join(p for p in parts if p)
        return (prefix + text).strip()

    async def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        if self._proc is None or self._closed:
            raise MCPError("client 未启动")
        envelope = {
            "jsonrpc": _MCP_JSONRPC,
            "method": method,
        }
        if params is not None:
            envelope["params"] = params
        await self._write_json(envelope)

    async def _request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        if self._proc is None or self._closed:
            raise MCPError("client 未启动")
        self._request_id += 1
        rid = self._request_id
        envelope: Dict[str, Any] = {
            "jsonrpc": _MCP_JSONRPC,
            "id": rid,
            "method": method,
        }
        if params is not None:
            envelope["params"] = params
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut
        try:
            await self._write_json(envelope)
        except Exception as e:
            self._pending.pop(rid, None)
            if not fut.done():
                fut.set_exception(e)
            raise
        try:
            return await fut
        finally:
            self._pending.pop(rid, None)

    async def _write_json(self, obj: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + body)
        await self._proc.stdin.drain()

    async def _reader_loop(self) -> None:
        try:
            assert self._proc is not None and self._proc.stdout is not None
            async for msg in self._read_frames():
                try:
                    envelope = json.loads(msg.decode("utf-8"))
                except Exception:
                    continue
                self._dispatch(envelope)
        except asyncio.CancelledError:
            raise
        except Exception:
            # stdout 读完就结束（server 自己关了），不再主动抛
            pass

    async def _read_frames(self) -> AsyncIterator[bytes]:
        assert self._proc is not None and self._proc.stdout is not None
        reader = self._proc.stdout
        buf = bytearray()
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                # 找 header 结束符
                term_idx = buf.find(_HEADERS_TERMINATOR)
                if term_idx < 0:
                    break
                header_bytes = bytes(buf[:term_idx])
                content_length: Optional[int] = None
                for line in header_bytes.split(b"\r\n"):
                    try:
                        text = line.decode("ascii", errors="ignore")
                    except Exception:
                        continue
                    if ":" not in text:
                        continue
                    k, _, v = text.partition(":")
                    if k.strip().lower() == _CONTENT_LENGTH_HEADER:
                        try:
                            content_length = int(v.strip())
                        except ValueError:
                            content_length = None
                        break
                if content_length is None:
                    # 格式不对，跳过这个帧，丢弃 header 终止符之前的内容
                    del buf[: term_idx + len(_HEADERS_TERMINATOR)]
                    continue
                body_start = term_idx + len(_HEADERS_TERMINATOR)
                total_needed = body_start + content_length
                if len(buf) < total_needed:
                    break
                body = bytes(buf[body_start:total_needed])
                del buf[:total_needed]
                yield body

    def _dispatch(self, envelope: Any) -> None:
        if not isinstance(envelope, dict):
            return
        rid = envelope.get("id")
        if rid is None:
            # 通知：忽略（tools/list_changed / logging/message 等先不管）
            return
        fut = self._pending.get(rid)
        if fut is None:
            return
        if fut.done():
            return
        if "error" in envelope:
            err = envelope["error"]
            if isinstance(err, dict):
                msg = str(err.get("message") or "unknown mcp error")
                code = err.get("code")
            else:
                msg, code = str(err), None
            fut.set_exception(MCPError(msg, code=code, raw=err))
            return
        fut.set_result(envelope.get("result") or {})
