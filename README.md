# AI Agent 架构设计

## 一、项目结构

```
src/ai_agent/
├── app.py                 # FastAPI 应用入口
├── config.py              # Pydantic 配置管理
├── main.py                # 应用启动入口
├── dependencies.py        # 依赖构建（依赖图）
├── schemas.py             # 请求/响应模型
├── routes/                # API 路由
│   ├── __init__.py
│   ├── agent_routes.py    # Agent 聊天路由
│   └── conversation_routes.py  # 会话管理路由
├── core/                  # 核心组件
│   ├── __init__.py
│   ├── agent_runtime.py   # AgentRuntime（运行时：循环控制）
│   ├── action_executor.py # ActionExecutor（动作执行）
│   ├── application_profile.py  # 应用配置
│   ├── context_manager.py     # ContextManager（上下文管理）
│   ├── context_provider.py    # 上下文提供者
│   ├── event.py           # EventBus（事件总线）
│   ├── handlers/          # 事件处理器
│   │   └── __init__.py
│   ├── planner.py         # LLM Planner（决策）
│   ├── policy.py          # RuntimePolicy（策略：迭代/超时/取消）
│   ├── provider.py        # Provider 基类
│   ├── stream.py          # 流式输出处理
│   └── observer.py        # 事件观察者
├── persistence/           # 持久化层
│   ├── __init__.py
│   ├── models.py          # 领域模型（Conversation）
│   └── store.py           # SQLite 存储（ConversationStore）
├── llm/                   # LLM 抽象层
│   ├── base.py            # BaseLLM 抽象接口
│   ├── factory.py         # LLM 实例工厂
│   └── deepseek.py        # DeepSeek 实现
├── models/                # 数据模型
│   ├── action.py          # Action / ToolAction / AnswerAction 定义
│   ├── chat.py            # ChatMessage 定义
│   ├── context.py         # AgentContext（决策上下文快照）
│   └── runtime.py         # ExecutionResult / RuntimeEvent 定义
├── prompts/               # Prompt 模板（集中管理）
│   ├── defaults/system.txt # 默认系统提示词
│   ├── agent_system.txt   # Agent 系统提示词
│   ├── answer.txt         # 回答提示词
│   └── prompt_loader.py   # Prompt 加载器
└── tools/                 # 工具模块
    ├── base.py            # BaseTool 抽象接口 + ToolRegistry
    ├── calculator.py      # 计算器工具
    ├── datetime_tool.py   # 日期时间工具
    ├── local_provider.py  # 本地工具提供者
    └── text_stats_tool.py # 文本统计工具

data/
└── conversations.db       # SQLite 会话数据库（自动创建）
```

## 二、核心架构

### 2.1 架构分层

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                          │
│         (routes: /agent/chat, /conversations)           │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                   AgentRuntime                          │
│     (运行时：循环控制、迭代计数、超时、取消)               │
└────────────────────────────┬────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│              │    │              │    │              │
│  Planner     │    │ ContextManager│    │ ActionExecutor│
│  (AI决策)    │    │ (上下文管理)   │    │ (动作执行)     │
│              │    │              │    │              │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│              资源层（Capability / Service）               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │Conversation  │ │ EventBus     │ │ ToolProvider │     │
│  │   Store      │ │ (事件总线)   │ │              │     │
│  └──────────────┘ └──────────────┘ └──────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心组件职责

#### AgentRuntime
- 控制 Agent 的生命周期
- 管理多步推理循环（Tool → Planner → Tool）
- 管理迭代次数（max_iterations）和超时
- 处理异常和中断
- **不包含业务逻辑，只做控制流**

#### Planner
- 接收 AgentContext 快照
- 调用 LLM 做出决策（回答用户或调用工具）
- 输出：AnswerAction（直接回答）或 ToolAction（调用工具）
- **只负责决策，不执行**

#### ContextManager
- 构建初始 AgentContext（遍历 ContextProvider 收集数据）
- 消费 ExecutionResult 更新上下文（通过 consume() 方法）
- **不包含业务逻辑，只做数据转换和组装**

#### ActionExecutor
- 执行所有类型的 Action（ToolAction / AnswerAction / ErrorAction）
- 返回统一的 ExecutionResult（包含事件列表）
- 处理异常，包装成 ExecutionResult.error()
- 支持 CancellationToken 和 RetryPolicy
- **不包含业务逻辑，只做执行和异常封装**

#### ContextProvider
- 从各种资源提供者获取数据
- 每个 Provider 只负责填充 AgentContext 的一个字段
- 当前实现：ConversationProvider / MemoryProvider / ApplicationProvider / RuntimeProvider

#### ConversationStore
- SQLite 持久化存储会话
- 管理会话的完整生命周期（创建、读取、更新、删除）

#### EventBus
- 事件发布/订阅机制
- 支持同步 emit() 和异步 emit_async()
- 内置处理器：PrintLogHandler、TokenCountHandler、ConversationPersistHandler

## 三、数据流向

### 3.1 多步对话流程

```
用户输入 → API Layer → AgentRuntime.run_stream()
                          ↓
                    ContextManager.build_initial()
                          ↓
                    AgentContext（快照）
                          ↓
                    Planner.plan() → Action
                          ↓
                    ActionExecutor.execute() → ExecutionResult
                          ↓
                    ContextManager.consume() → 新 AgentContext
                          ↓
                    循环直到 Planner 返回 AnswerAction 或达到 max_iterations
                          ↓
                    StreamHandle（流式输出）
                          ↓
                    SSE → 前端
```

### 3.2 AgentContext 数据结构

```python
@dataclass(frozen=True)
class AgentContext:
    conversation: List[ChatMessage]  # 会话历史
    memory: MemorySnapshot  # 长期记忆
    available_actions: List[Action]  # 可用工具列表
    runtime_state: RuntimeState  # 运行时状态
    user_input: str  # 当前用户输入
```

## 四、关键设计原则

### 4.1 封装原则

| 原则 | 说明 |
|------|------|
| 持久化隐藏 | ContextProvider 不关心数据如何持久化，只依赖 ConversationStore 接口 |
| 接口隔离 | Provider 只提供资源，Executor 只执行动作，Planner 只做决策 |
| 控制流与业务分离 | AgentRuntime/ContextManager/ActionExecutor 不包含业务逻辑 |
| 事件驱动 | 通过 EventBus 发布事件，Observer 决定如何处理（日志、持久化等） |

### 4.2 依赖注入

所有核心组件通过 `build_app_state()` 构建依赖图：

```python
def build_app_state():
    # 1. 基础基础设施
    bus = EventBus()
    store = ConversationStore(max_conversations=100)
    llm = create_llm()
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(DateTimeTool())
    
    bus.subscribe(ConversationPersistHandler(store))
    
    # 2. 按 ApplicationProfile 构造 AgentRuntime
    profile = ApplicationProfile.agent_app(
        "agent",
        system_prompt="agent_system",
        tools=["calculator", "datetime"],
    )
    
    # 3. 工具提供者
    tool_provider = LocalToolProvider(tool_registry)
    
    # 4. 上下文提供者
    providers = [
        ConversationProvider(store, bus),
        MemoryProvider(),
        ApplicationProvider(tool_provider),
        RuntimeProvider(max_iterations=5),
    ]
    context_manager = DefaultContextManager(providers=providers, bus=bus)
    
    # 5. 执行器和 Planner
    executor = ActionExecutor(tool_provider=tool_provider, llm=llm)
    planner = LLMPlanner(llm=llm, tool_provider=tool_provider)
    
    # 6. 策略
    policy = RuntimePolicy(max_iterations=5, timeout_seconds=300)
    
    # 7. Agent 实例
    agent_runtime = AgentRuntime(
        planner=planner,
        context_manager=context_manager,
        executor=executor,
        bus=bus,
        policy=policy,
    )
```

### 4.3 扩展方式

| 扩展类型 | 方式 |
|----------|------|
| 新应用 | 新增 ApplicationProfile（配置 system_prompt + tools） |
| 新工具 | 继承 BaseTool，注册到 ToolRegistry |
| 新 LLM | 继承 BaseLLM，在 factory.py 中添加 |
| 新上下文 | 继承 ContextProvider，添加到 DefaultContextManager |
| 新 Action 类型 | 继承 Action，在 ActionExecutor.execute() 中添加分支 |
| 新事件处理器 | 实现 Event handler，订阅到 EventBus |

## 五、配置与运行

### 5.1 环境配置

复制 `.env.example` 为 `.env`：

```
LLM_API_KEY=your_api_key
LLM_API_BASE=https://api.deepseek.com/v1
MAX_CONVERSATIONS=100
```

### 5.2 启动方式

```bash
uv run python -m ai_agent.main
```

## 六、架构约束

以下变更需要讨论后再执行：

- **修改 AgentContext 数据结构**：需要评估对 Planner 和 Provider 的影响
- **新增 Provider 类型**：需要确认是否符合现有分层设计
- **修改 ConversationStore API**：需要评估对 ContextProvider 和 API 层的影响
- **修改 AgentRuntime 核心逻辑**：需要评估对稳定性和性能的影响
- **新增持久化方案**：需要评估与现有 SQLite 方案的兼容性

## 七、开发规范

- 代码遵循 PEP 8
- 使用类型注解
- 避免全局变量，通过依赖注入传递
- Prompt 使用 `.txt` 文件管理，不硬编码在 Python 中
- 新增功能优先考虑扩展，而非修改核心组件
- 核心控制流模块（AgentRuntime/ContextManager/ActionExecutor）不包含业务逻辑
- 文件拆分原则：一个文件一个职责，避免大文件