from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ai_agent.core.action_executor import ActionExecutor
from ai_agent.core.agent_runtime import AgentRuntime
from ai_agent.core.application_profile import ApplicationProfile
from ai_agent.core.context_manager import ContextManager
from ai_agent.core.context_provider import (
    ApplicationProvider,
    ConversationProvider,
    MemoryProvider,
    RuntimeProvider,
)
from ai_agent.core.event import EventBus, get_default_bus
from ai_agent.core.handlers import ConversationPersistHandler
from ai_agent.core.planner import Planner
from ai_agent.core.policy import RuntimePolicy
from ai_agent.core.provider import CompositeToolProvider
from ai_agent.llm.base import BaseLLM
from ai_agent.llm.factory import create_llm
from ai_agent.persistence.models import Conversation
from ai_agent.persistence.store import ConversationStore
from ai_agent.skills.base import BaseSkill, SkillManager
from ai_agent.tools.base import ToolRegistry
from ai_agent.tools.calculator import CalculatorTool
from ai_agent.tools.datetime_tool import DateTimeTool
from ai_agent.tools.local_provider import LocalToolProvider
from ai_agent.tools.mcp import MCPServerConfig, MCPToolProvider
from ai_agent.tools.text_stats_tool import TextStatsTool

_project_root = Path(__file__).resolve().parent.parent.parent
_persist_path = _project_root / "data" / "conversations.db"
_max_conversations = 100
_max_agent_iterations = 5


def _default_profiles() -> Dict[str, ApplicationProfile]:
    return {
        "agent": ApplicationProfile.agent_app(
            "agent",
            system_prompt="agent_system",
            tools=["calculator", "datetime", "text_stats"],
            max_iterations=_max_agent_iterations,
        ),
    }


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(TextStatsTool())
    return registry


def build_app_state(
    llm: Optional[BaseLLM] = None,
    bus: Optional[EventBus] = None,
    store: Optional[ConversationStore] = None,
    tool_registry: Optional[ToolRegistry] = None,
    profiles: Optional[Dict[str, ApplicationProfile]] = None,
    skills: Optional[Iterable[BaseSkill]] = None,
    mcp_configs: Optional[Iterable[MCPServerConfig]] = None,
) -> "AppState":
    bus = bus or get_default_bus()

    store = store or ConversationStore(
        max_conversations=_max_conversations,
        persist_path=_persist_path,
    )
    llm = llm or create_llm()
    tool_registry = tool_registry or build_tool_registry()

    bus.subscribe(ConversationPersistHandler(store))

    profiles = profiles or _default_profiles()

    # 全局 Skill / MCP 管理器（应用级生命周期）
    skill_manager = SkillManager(list(skills or []))
    mcp_provider = MCPToolProvider.from_configs(list(mcp_configs or []))
    composite_providers: List[CompositeToolProvider] = []

    agent_runtimes: Dict[str, AgentRuntime] = {}

    for profile in profiles.values():
        system_prompt_text = profile.resolve_system_prompt()

        resolved_tools = profile.resolve_tools(tool_registry)
        local_registry = ToolRegistry()
        for t in resolved_tools:
            local_registry.register(t)
        local_provider = LocalToolProvider(local_registry)

        composite = CompositeToolProvider([
            local_provider,
            skill_manager,
            mcp_provider,
        ])
        composite_providers.append(composite)

        providers: List[Any] = [
            ConversationProvider(store, bus),
            MemoryProvider(),
            *skill_manager.get_context_providers(),
            ApplicationProvider(
                composite,
                extra_prompt_snippets=skill_manager.get_system_prompt_snippets,
            ),
            RuntimeProvider(profile.max_iterations),
        ]
        context_manager = ContextManager(
            providers=providers,
            bus=bus,
        )

        executor = ActionExecutor(
            tool_provider=composite,
            llm=llm,
        )

        planner = Planner(
            llm=llm,
            tool_provider=composite,
            system_prompt=system_prompt_text,
        )

        policy = RuntimePolicy(
            max_iterations=profile.max_iterations,
            timeout_seconds=300.0,
        )
        agent_runtimes[profile.name] = AgentRuntime(
            planner=planner,
            context_manager=context_manager,
            executor=executor,
            bus=bus,
            policy=policy,
        )

    return AppState(
        llm=llm,
        bus=bus,
        store=store,
        tool_registry=tool_registry,
        profiles=profiles,
        agent_runtimes=agent_runtimes,
        skill_manager=skill_manager,
        mcp_provider=mcp_provider,
        composite_providers=composite_providers,
    )


class AppState:
    """所有应用级单例的容器，挂在 app.state.ai 上。"""

    def __init__(
        self,
        llm: BaseLLM,
        bus: EventBus,
        store: ConversationStore,
        tool_registry: ToolRegistry,
        profiles: Dict[str, ApplicationProfile],
        agent_runtimes: Dict[str, AgentRuntime],
        skill_manager: SkillManager,
        mcp_provider: MCPToolProvider,
        composite_providers: List[CompositeToolProvider],
    ) -> None:
        self.llm = llm
        self.bus = bus
        self.store = store
        self.tool_registry = tool_registry
        self.profiles = profiles
        self.agent_runtimes = agent_runtimes
        self.skill_manager = skill_manager
        self.mcp_provider = mcp_provider
        self._composites = list(composite_providers)

    @property
    def agent_runtime(self) -> AgentRuntime:
        if "agent" in self.agent_runtimes:
            return self.agent_runtimes["agent"]
        if not self.agent_runtimes:
            raise RuntimeError(
                "没有注册任何 Agent 模式的应用。请在 _default_profiles() 中添加"
                "一个 ApplicationProfile.agent_app()。"
            )
        return next(iter(self.agent_runtimes.values()))

    async def setup(self) -> None:
        """应用启动时的生命周期：初始化 Skill / MCP / Provider。"""
        errors: List[str] = []
        try:
            await self.skill_manager.setup()
        except Exception as e:
            errors.append(f"skill_manager.setup: {e}")
        try:
            await self.mcp_provider.setup()
        except Exception as e:
            # MCP 启动失败（没有 server、命令行错等）不影响全局运行
            errors.append(f"mcp_provider.setup: {e}")
        for cp in self._composites:
            try:
                await cp.setup()
            except Exception as e:
                errors.append(f"composite.setup: {e}")
        if errors:
            import logging

            logging.getLogger(__name__).warning(
                "AppState.setup warnings: %s", "; ".join(errors)
            )

    async def teardown(self) -> None:
        """应用关闭时的生命周期：关 MCP 子进程、关 Skill、释放 Provider。"""
        import asyncio

        tasks = []
        tasks.append(asyncio.create_task(self.skill_manager.teardown()))
        tasks.append(asyncio.create_task(self.mcp_provider.teardown()))
        for cp in self._composites:
            tasks.append(asyncio.create_task(cp.teardown()))
        for t in tasks:
            try:
                await asyncio.wait_for(t, timeout=5)
            except Exception:
                t.cancel()
                try:
                    await t
                except Exception:
                    pass
