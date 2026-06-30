"""端到端测试：Planner + Calculator + AgentRuntime 通过 FastAPI TestClient 验证。

注意：这会真实调用 LLM API（2 次：1 次 Planner + 1 次 ChatRuntime），需要 API key。
"""

import asyncio
import json
import time

from fastapi.testclient import TestClient
from src.ai_agent.app import app

client = TestClient(app)


def run(name, fn):
    print(f"\n[测试] {name}")
    try:
        fn()
        print(f"    OK")
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()


def test_1_tools_api():
    """GET /tools 应返回 calculator。"""
    r = client.get("/tools")
    assert r.status_code == 200
    data = r.json()
    print(f"    工具列表：{[t['name'] for t in data['tools']]}")
    assert data["count"] >= 1
    assert any(t["name"] == "calculator" for t in data["tools"])


def test_2_agent_chat_math():
    """POST /agent/chat（非流式）：数学问题应触发 calculator。"""
    # 创建会话
    r = client.post("/conversations", json={"title": "agent-math-test"})
    sid = r.json()["conversation"]["session_id"]
    print(f"    session_id: {sid}")

    r = client.post(
        "/agent/chat",
        json={"session_id": sid, "messages": [{"role": "user", "content": "请问 123 × 456 等于多少？"}]},
    )
    print(f"    status: {r.status_code}")
    data = r.json()
    # 应该是 "message": "答案"
    msg = data.get("message", str(data))
    print(f"    回答：{msg[:200]}")
    # 结果应包含 56088
    assert "56088" in str(msg) or "56088" in str(msg), f"预期结果里有 56088，实际是：{msg}"
    # 清理
    client.delete(f"/conversations/{sid}")


def test_3_agent_chat_casual():
    """POST /agent/chat（非流式）：闲聊问题应该不调用工具。"""
    r = client.post("/conversations", json={"title": "agent-chat-test"})
    sid = r.json()["conversation"]["session_id"]

    r = client.post(
        "/agent/chat",
        json={
            "session_id": sid,
            "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
        },
    )
    print(f"    status: {r.status_code}")
    data = r.json()
    msg = data.get("message", str(data))
    print(f"    回答：{msg[:200]}")
    assert msg and len(str(msg)) > 5, "应有有效回答"
    client.delete(f"/conversations/{sid}")


def test_4_agent_stream():
    """POST /agent/chat（stream=true）：流式应返回多个 SSE 行，其中有 token 和 event。"""
    r = client.post("/conversations", json={"title": "agent-stream-test"})
    sid = r.json()["conversation"]["session_id"]

    r = client.post(
        "/agent/chat",
        json={
            "session_id": sid,
            "messages": [{"role": "user", "content": "计算 2^10 的值"}],
            "stream": True,
        },
        stream=True,
    )

    lines = list(r.iter_lines())
    print(f"    收到 {len(lines)} 行 SSE")
    data_lines = [l for l in lines if l.startswith(b"data:")]
    print(f"    其中 data 行 {len(data_lines)} 个")
    # 解析
    has_event = False
    has_token = False
    for l in data_lines:
        try:
            payload = json.loads(l[len(b"data: "):])
        except Exception:
            continue
        if "_event" in payload:
            has_event = True
            print(f"    event: {payload['_event']['name']} = {json.dumps(payload['_event']['payload'], ensure_ascii=False)[:120]}")
        if payload.get("choices") and payload["choices"][0].get("delta", {}).get("content"):
            has_token = True
    assert has_event, "流式应至少有一个 agent.* 事件"
    assert has_token, "流式应至少有一个 token"
    # 看看完整文本里有没有 1024
    full = ""
    for l in data_lines:
        try:
            payload = json.loads(l[len(b"data: "):])
        except Exception:
            continue
        if payload.get("choices"):
            choice = payload["choices"][0]
            delta = choice.get("delta", {}).get("content", "") or ""
            full += delta
    print(f"    完整文本：{full[:200]}")
    assert "1024" in full, f"应提到 1024，实际：{full}"
    client.delete(f"/conversations/{sid}")


if __name__ == "__main__":
    run("1. GET /tools", test_1_tools_api)
    run("2. agent/chat 数学问题（非流式）", test_2_agent_chat_math)
    run("3. agent/chat 闲聊（非流式）", test_3_agent_chat_casual)
    run("4. agent/chat 流式（含事件）", test_4_agent_stream)
    print("\n" + "=" * 60)
    print("完成")
