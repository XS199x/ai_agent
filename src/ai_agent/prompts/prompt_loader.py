"""集中管理系统 Prompt：从 .txt 文件加载，避免硬编码在代码里。

示例：
    # 加载 Agent Planner 的系统 Prompt
    agent_system = load_prompt("agent_system")

新 Prompt 只需在 prompts/ 目录下新增一个 .txt 文件，不需要改任何 Python 代码。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

# prompts/ 目录的绝对路径 —— 无论从哪里启动都能找到
_PROMPTS_DIR: Path = Path(__file__).resolve().parent

# 内存缓存：避免每次请求都读文件（简单但够用）
_CACHE: Dict[str, str] = {}


def _resolve_path(name: str) -> Path:
    """把 prompt 名解析成文件路径。

    支持两种写法：
        "agent_system"       -> prompts/agent_system.txt
        "agent/system"       -> prompts/agent/system.txt
    """
    if not name.endswith(".txt"):
        name = name + ".txt"
    path = _PROMPTS_DIR / name
    return path


def load_prompt(name: str, default: Optional[str] = None) -> str:
    """加载一个 prompt 文件。

    Args:
        name: prompt 文件的标识（.txt 可省略）
        default: 如果找不到文件时返回的默认值；不传会抛 FileNotFoundError

    Returns:
        prompt 文件的文本内容（尾部空白已 strip）
    """
    if name in _CACHE:
        return _CACHE[name]

    path = _resolve_path(name)
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        _CACHE[name] = text
        return text

    if default is not None:
        return default

    raise FileNotFoundError(
        f"找不到 prompt 文件：{path}。请在 {_PROMPTS_DIR} 目录下创建对应的 .txt 文件。"
    )


def available_prompts() -> list[str]:
    """列出所有可用的 prompt 文件（不带 .txt 后缀）。"""
    result = []
    for f in _PROMPTS_DIR.rglob("*.txt"):
        rel = f.relative_to(_PROMPTS_DIR).as_posix()
        result.append(rel[:-4])  # 去掉 .txt
    return sorted(result)


def reload_all() -> None:
    """清空缓存，下次 load_prompt 会重新从文件读。

    调试时使用：改了 prompt 文件后调这个，不用重启服务。
    """
    _CACHE.clear()
