import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, "src")

from ai_agent.core.action_executor import ActionExecutor
from ai_agent.core.context_manager import ContextManager, ContextProvider
from ai_agent.core.event import Event, EventBus
from ai_agent.core.observer import StreamEventObserver
from ai_agent.core.planner import Planner
from ai_agent.core.policy import CancellationToken, RetryPolicy, RuntimePolicy
from ai_agent.core.provider import CompositeToolProvider, ToolProvider
from ai_agent.core.stream import StreamHandle, StreamItem, item_to_sse_line
from ai_agent.models.action import (
    Action,
    AnswerAction,
    ErrorAction,
    ToolAction,
    answer_action,
    error_action,
    tool_action,
)
from ai_agent.models.context import AgentContext, MemorySnapshot, RuntimeState
from ai_agent.models.runtime import ExecutionOutcome, ExecutionResult, RuntimeEvent

# 1. Event Bus Test
print("=== Test 1: Event Bus ===")
bus = EventBus()
events_received = []
bus.subscribe(lambda e: events_received.append(e))
bus.emit(Event("test", {"key": "value"}))
assert len(events_received) == 1
assert events_received[0].name == "test"
print("PASSED: Event Bus")

# 2. CancellationToken Test
print("\n=== Test 2: CancellationToken ===")
token = CancellationToken()
assert not token.cancelled
token.cancel("test reason")
assert token.cancelled
assert token.reason == "test reason"
try:
    token.raise_if_cancelled()
    assert False, "Should have raised"
except asyncio.CancelledError:
    pass
print("PASSED: CancellationToken")

# 3. RuntimePolicy Test
print("\n=== Test 3: RuntimePolicy ===")
policy = RuntimePolicy(max_iterations=5, timeout_seconds=100.0)
token = CancellationToken()
result = policy.allow_continue(3, 50.0, token)
assert result.allowed

result = policy.allow_continue(5, 50.0, token)
assert not result.allowed
assert "最大迭代" in result.reason or "max" in result.reason.lower()

token.cancel()
result = policy.allow_continue(0, 0.0, token)
assert not result.allowed
print("PASSED: RuntimePolicy")

# 4. RetryPolicy Test
print("\n=== Test 4: RetryPolicy ===")
retry = RetryPolicy(max_retries=3, delay_seconds=0.1)
attempts = list(retry.attempts())
assert len(attempts) == 4  # 0, 1, 2, 3
assert attempts[0].attempt == 0
assert attempts[0].should_retry
assert not attempts[3].should_retry  # Last attempt should not retry
print("PASSED: RetryPolicy")

# 5. StreamHandle Test
print("\n=== Test 5: StreamHandle ===")
handle = StreamHandle()
handle.emit_token("Hello")
handle.emit_token(" World")
assert handle.full_text == "Hello World"
assert handle.token_count == 2

# Test SSE serialization
item = StreamItem(kind="token", delta="test")
sse = item.to_sse_json()
assert sse["choices"][0]["delta"]["content"] == "test"
print("PASSED: StreamHandle")

# 6. ContextManager Test
print("\n=== Test 6: ContextManager ===")


class DummyProvider(ContextProvider):
    async def provide(self, session_id, user_input):
        return {"test_field": "test_value"}


async def test_context_manager():
    cm = ContextManager([DummyProvider()])
    ctx = await cm.build_initial("session1", "hello")
    assert isinstance(ctx, AgentContext)
    assert ctx.runtime_state.session_id == "session1"
    assert ctx.user_input == "hello"
    return ctx


asyncio.run(test_context_manager())
print("PASSED: ContextManager")

# 7. Action Types Test
print("\n=== Test 7: Action Types ===")
ta = tool_action("calculator", {"expr": "2+2"})
assert isinstance(ta, ToolAction)
assert ta.name == "calculator"
assert ta.args == {"expr": "2+2"}

aa = answer_action("Hello")
assert isinstance(aa, AnswerAction)
assert aa.content == "Hello"

ea = error_action("Error occurred")
assert isinstance(ea, ErrorAction)
assert ea.message == "Error occurred"
print("PASSED: Action Types")

# 8. ExecutionResult Test
print("\n=== Test 8: ExecutionResult ===")
result = ExecutionResult.success("output", ExecutionOutcome.STOP)
assert result.success
assert not result.should_continue  # STOP means no continue

result = ExecutionResult.success("output", ExecutionOutcome.CONTINUE)
assert result.should_continue

result = ExecutionResult.from_error("error message")
assert not result.success
assert result.error == "error message"
assert not result.should_continue  # Error should stop
print("PASSED: ExecutionResult")

# 9. Lazy imports from main modules
print("\n=== Test 9: Core Module Imports ===")
from ai_agent.core import (
    ActionExecutor,
    AgentRuntime,
    ContextManager,
    ContextProvider,
    Event,
    EventBus,
    Planner,
    Provider,
    StreamHandle,
    StreamItem,
    ToolProvider,
)

print("PASSED: All core module imports work")

# 10. App State Integration Test
print("\n=== Test 10: App State Integration ===")


async def test_app_state():
    from ai_agent.dependencies import build_app_state

    fake_llm = SimpleNamespace()
    state = build_app_state(llm=fake_llm)
    await state.setup()
    runtime = state.agent_runtime
    assert runtime is not None
    assert isinstance(runtime._planner, Planner)
    assert isinstance(runtime._context_manager, ContextManager)
    await state.teardown()
    print("PASSED: App State Integration")


asyncio.run(test_app_state())

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
