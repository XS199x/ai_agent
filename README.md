# AI Agent 架构设计

## 一、项目结构

```
src/ai_agent/
├── app.py                 # FastAPI 应用入口，构建依赖图和路由
├── config.py              # Pydantic 配置管理
├── main.py                # 应用启动入口
├── core/                  # 核心组件
│   ├── agent_runtime.py   # AgentRuntime（运行时：循环控制、异常处理）
│   ├── action_dispatcher.py  # ActionDispatcher（动作分发：注册 Handler、分发执行）
│   ├── application_profile.py  # 应用配置文件（system_prompt / tools / max_iterations）
│   ├── context_manager.py     # ContextManager（上下文管理：构建、更新、压缩）
│   ├── context_provider.py    # 上下文提供者（从 Store 获取数据，组装 AgentContext）
│   ├── conversation.py    # 会话模型 + SQLite 持久化存储
│   ├── event.py           # 事件总线（EventBus）
│   ├── executor.py        # 执行器 + Provider 接口定义
│   ├── planner.py         # LLM Planner（根据 Context 决策下一步动作）
│   ├── stream.py          # 流式输出处理（SSE）
│   └── knowledge/         # 知识库模块
│       └── file_knowledge_provider.py  # 基于文件的知识库实现
├── llm/                   # LLM 抽象层
│   ├── base.py            # BaseLLM 抽象接口
│   ├── factory.py         # LLM 实例工厂
│   └── deepseek.py        # DeepSeek 实现
├── models/                # 数据模型
│   ├── action.py          # Action / ToolAction 定义
│   ├── chat.py            # ChatMessage 定义
│   └── context.py         # AgentContext（决策上下文快照）
├── prompts/               # Prompt 模板（集中管理）
│   ├── agent/answer.txt   # 回答生成模板
│   ├── defaults/system.txt # 默认系统提示词
│   └── prompt_loader.py   # Prompt 加载器
└── tools/                 # 工具模块
    ├── base.py            # BaseTool 抽象接口
    ├── calculator.py      # 计算器工具
    ├── datetime_tool.py   # 日期时间工具
    ├── local_provider.py  # 本地工具提供者
    └── text_stats_tool.py # 文本统计工具

data/
├── conversations.db       # SQLite 会话数据库（自动创建）
└── knowledge/             # 知识库文件目录（.md / .txt）
```

## 二、核心架构

### 2.1 架构分层

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                          │
│         (FastAPI routes: /agent/chat, /conversations)   │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                   AgentRuntime                          │
│     (运行时：循环控制、迭代计数、超时、异常、中断)          │
└────────────────────────────┬────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│              │    │              │    │              │
│  Planner     │    │ ContextManager│    │ ActionDispatcher│
│  (AI决策)    │    │ (上下文管理)   │    │ (动作分发)     │
│              │    │              │    │              │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│              资源层（Capability / Service）               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │Conversation  │ │ Knowledge    │ │ ToolProvider │     │
│  │   Store      │ │   Provider   │ │              │     │
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
- 构建初始 AgentContext（从 ContextProvider 获取）
- 更新上下文（将工具执行结果回灌）
- 构建 LLM 消息列表
- 上下文压缩（超长对话时进行摘要）
- **不包含业务逻辑，只做数据转换**

#### ActionDispatcher
- 注册 Action Handler（ToolActionHandler / AnswerActionHandler / ErrorActionHandler）
- 根据 Action 类型分发到对应 Handler 执行
- 新增 Action 类型只需注册新 Handler，不改核心代码
- **不包含业务逻辑，只做分发**

#### ContextProvider
- 从各种资源提供者获取数据
- 组装成 AgentContext（不可变快照）
- 当前实现：SimpleContextProvider

#### ConversationStore
- SQLite 持久化存储会话
- 管理会话的完整生命周期（创建、读取、更新、删除）
- **内部集成知识库检索**：通过 Conversation.retrieve_knowledge() 对外隐藏

#### KnowledgeProvider
- 知识库检索接口
- 当前实现：FileKnowledgeProvider（基于文件的关键词检索）
- **对外隐藏**：通过 Conversation 内部调用，不直接暴露给 API

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
                    ActionDispatcher.dispatch() → Result
                          ↓
                    ContextManager.update() → 新 AgentContext
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
    conversation: List[ChatMessage]      # 会话历史
    memory: MemorySnapshot                # 长期记忆
    knowledge: List[KnowledgeEntry]       # RAG 检索结果
    available_actions: List[Action]       # 可用工具列表
    runtime_state: RuntimeState           # 运行时状态
    user_input: str                       # 当前用户输入
```

## 四、关键设计原则

### 4.1 封装原则

| 原则 | 说明 |
|------|------|
| 知识库隐藏 | KnowledgeProvider 通过 Conversation.retrieve_knowledge() 内部调用，API 层不直接访问 |
| 持久化隐藏 | ContextProvider 不关心数据如何持久化，只依赖 ConversationStore 接口 |
| 接口隔离 | Provider 只提供资源，Executor 只执行动作，Planner 只做决策 |
| 控制流与业务分离 | AgentRuntime/ContextManager/ActionDispatcher 不包含业务逻辑 |

### 4.2 依赖注入

所有核心组件通过 `build_app_state()` 构建依赖图：

```python
def build_app_state():
    # 1. 基础基础设施
    bus = EventBus()
    knowledge_provider = FileKnowledgeProvider()
    store = ConversationStore(knowledge_provider=knowledge_provider)
    
    # 2. 业务组件
    llm = create_llm()
    tool_provider = LocalToolProvider(tool_registry)
    tool_executor = ToolExecutor(tool_provider)
    
    # 3. 上下文管理
    context_provider = SimpleContextProvider(tool_provider, store)
    context_manager = DefaultContextManager(context_provider, llm)
    
    # 4. 动作分发
    dispatcher = ActionDispatcher()
    dispatcher.register_handler(ToolActionHandler(tool_executor))
    dispatcher.register_handler(AnswerActionHandler())
    dispatcher.register_handler(ErrorActionHandler())
    
    # 5. Planner
    planner = LLMPlanner(llm, tool_provider)
    
    # 6. Agent 实例
    agent_runtime = AgentRuntime(planner, context_manager, dispatcher, llm, bus)
```

### 4.3 扩展方式

| 扩展类型 | 方式 |
|----------|------|
| 新应用 | 新增 ApplicationProfile（配置 system_prompt + tools） |
| 新工具 | 继承 BaseTool，注册到 ToolRegistry |
| 新 LLM | 继承 BaseLLM，在 factory.py 中添加 |
| 新知识库 | 继承 KnowledgeProvider，替换 FileKnowledgeProvider |
| 新上下文 | 继承 ContextProvider，替换 SimpleContextProvider |
| 新 Action 类型 | 继承 Action，实现对应 Handler，注册到 ActionDispatcher |

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

### 5.3 知识库

在 `data/knowledge/` 目录下放置 `.md` 或 `.txt` 文件，系统会自动加载并索引。

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
- 核心控制流模块（AgentRuntime/ContextManager/ActionDispatcher）不包含业务逻辑