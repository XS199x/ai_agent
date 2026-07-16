"""Skill 抽象与 SkillManager。

Skill 是一个比"工具"更高层的能力单元。
一个 Skill = 一组工具 + 一段专属的 prompt snippet + 可选的上下文提供者 + 生命周期。

设计原则：
- BaseTool 是"原子能力"（算个数、读个文件），不感知决策
- BaseSkill 是"复合能力"（代码生成），聚合多个 Tool + Prompt + Context，对任务场景建模
- SkillManager 自身实现了 ToolProvider 接口，对 Planner/Executor 完全透明：
  上层只看到一堆 tools 和一段 prompt 片段，不知道 Skill 的存在
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ai_agent.core.provider import Provider, ToolProvider
from ai_agent.models.action import Action, tool_action
from ai_agent.tools.base import BaseTool

# ---------- Skill 基类 ----------


class BaseSkill(ABC):
    """可复用的复合能力单元。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """唯一标识名，用于 enable/disable。"""

    @property
    def description(self) -> str:
        """用户可读说明，用于调试和未来的路由选择。"""
        return ""

    def is_enabled_by_default(self) -> bool:
        """启动时是否默认启用。"""
        return True

    def prompt_snippet(self) -> str:
        """追加到 system prompt 末尾的片段（默认空，按需覆盖）。

        例如代码生成 Skill 可以在这里注入"优先遵守 PEP8；
        写函数前先写类型签名；复杂步骤先出计划后逐段生成"等风格指令。
        """
        return ""

    def tools(self) -> List[BaseTool]:
        """Skill 自带的工具列表（和 LocalToolProvider 里的工具同等地位）。"""
        return []

    def context_provider(self) -> Optional[Provider]:
        """Skill 专属上下文提供者。

        返回一个 Provider 对象（实现 setup/provide/teardown 或具体子接口）。
        SkillManager 会把这些 Provider 和全局 providers 合并交给 ContextManager。
        """
        return None

    async def setup(self) -> None:
        """Skill 启动时执行。子类按需覆盖。"""

    async def teardown(self) -> None:
        """Skill 关闭时执行。子类按需覆盖。"""


# ---------- Skill 管理器：也是一个 ToolProvider ----------


@dataclass
class _SkillState:
    skill: BaseSkill
    enabled: bool
    tools: List[BaseTool] = field(default_factory=list)


class SkillManager(ToolProvider):
    """Skill 生命周期与状态管理。

    作为 ToolProvider 暴露给上层：
    - list_tools()：返回所有 enabled skill 的工具（去重）
    - get_tool(name)：按名查找（去重，name 冲突时先注册的优先）
    - as_actions()：把 enabled 工具转成 Action 列表

    额外的 Skill 专属方法：
    - enable / disable：按名切换启用状态
    - get_system_prompt_snippets()：把 enabled skill 的 prompt 片段拼起来
    - get_context_providers()：返回 enabled skill 的 context_provider 列表
    - setup / teardown：遍历所有 skill 调对应生命周期
    """

    def __init__(self, skills: Iterable[BaseSkill]):
        self._states: Dict[str, _SkillState] = {}
        for s in skills:
            if not isinstance(s, BaseSkill):
                raise TypeError(f"Skill 必须是 BaseSkill 子类，收到 {type(s).__name__}")
            if s.name in self._states:
                raise ValueError(f"Skill 重名：{s.name!r}")
            # 先立即取一次 tools()：对于不需要 setup 就能用的 Skill，
            # 不 await setup() 也能直接 list_tools() / get_tool()。
            try:
                initial_tools = list(s.tools() or [])
            except Exception:
                initial_tools = []
            self._states[s.name] = _SkillState(
                skill=s,
                enabled=s.is_enabled_by_default(),
                tools=initial_tools,
            )

    # ---------- 生命周期 ----------

    async def setup(self) -> None:
        for state in self._states.values():
            try:
                await state.skill.setup()
            except Exception:
                state.enabled = False
                continue
            # setup() 后再刷新一次：有些 Skill 需要登录/初始化后才能拿到工具列表
            try:
                state.tools = list(state.skill.tools() or [])
            except Exception:
                pass

    async def teardown(self) -> None:
        for state in self._states.values():
            try:
                await state.skill.teardown()
            except Exception:
                pass
            state.tools.clear()

    async def health(self) -> bool:
        if not self._states:
            return False
        return any(s.enabled for s in self._states.values())

    # ---------- Skill 开关 ----------

    def enable(self, name: str) -> bool:
        st = self._states.get(name)
        if st is None:
            return False
        st.enabled = True
        return True

    def disable(self, name: str) -> bool:
        st = self._states.get(name)
        if st is None:
            return False
        st.enabled = False
        return True

    def enabled_names(self) -> List[str]:
        return [n for n, s in self._states.items() if s.enabled]

    # ---------- Skill 专属聚合能力 ----------

    def get_system_prompt_snippets(self) -> str:
        parts: List[str] = []
        for state in self._states.values():
            if not state.enabled:
                continue
            snippet = (state.skill.prompt_snippet() or "").strip()
            if not snippet:
                continue
            header = f"# 技能：{state.skill.name}"
            desc = (state.skill.description or "").strip()
            if desc:
                header += f"（{desc}）"
            parts.append(f"{header}\n{snippet}")
        return "\n\n".join(parts)

    def get_context_providers(self) -> List[Provider]:
        result: List[Provider] = []
        for state in self._states.values():
            if not state.enabled:
                continue
            cp = state.skill.context_provider()
            if cp is not None:
                result.append(cp)
        return result

    # ---------- ToolProvider 接口（对上层透明）----------

    def _iter_enabled_tools(self) -> Iterable[BaseTool]:
        seen: set[str] = set()
        for state in self._states.values():
            if not state.enabled:
                continue
            for tool in state.tools:
                if tool.name in seen:
                    continue
                seen.add(tool.name)
                yield tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        for tool in self._iter_enabled_tools():
            if tool.name == name:
                return tool
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._iter_enabled_tools()]

    def as_actions(self) -> List[Action]:
        actions: List[Action] = []
        for t in self._iter_enabled_tools():
            props = t.args_schema.get("properties") or {}
            actions.append(
                tool_action(
                    name=t.name,
                    args={k: "..." for k in props.keys()},
                )
            )
        return actions
