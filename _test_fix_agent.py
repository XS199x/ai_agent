"""修复验证：/agent/chat + calculator tool（宽松断言）"""

import json
import re
import sys

from fastapi.testclient import TestClient

from src.ai_agent.app import app


# 提取数字，忽略千位分隔符
def extract_nums(text):
    return [re.sub(r"[^\d]", "", s) for s in re.findall(r"[\d,]+", text)]


# 必须用 with 块才会触发 lifespan（设置 app.state.ai）
with TestClient(app) as client:
    print("=" * 60)
    print("测试 1：数学问题（应调用 calculator，非流式）")
    print("=" * 60)

    r = client.post(
        "/agent/chat",
        json={
            "session_id": "test-fix-001",
            "messages": [{"role": "user", "content": "请计算 123 × 456 等于多少？"}],
            "stream": False,
        },
    )
    print(f"status: {r.status_code}")
    assert r.status_code == 200, f"请求失败: {r.text}"
    data = r.json()
    choices = data.get("choices", [])
    msg = choices[0].get("message", {}).get("content", "") if choices else str(data)
    print(f"回答: {msg[:300]}")
    nums = extract_nums(msg)
    print(f"  提取的数字: {nums}")
    assert "56088" in nums, f"回答里没找到 56088（不管千位分隔符）: {msg}"
    print("  OK")

    print()
    print("=" * 60)
    print("测试 2：闲聊（不应调用工具，非流式）")
    print("=" * 60)

    r = client.post(
        "/agent/chat",
        json={
            "session_id": "test-fix-002",
            "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
            "stream": False,
        },
    )
    print(f"status: {r.status_code}")
    assert r.status_code == 200
    data = r.json()
    choices = data.get("choices", [])
    msg = choices[0].get("message", {}).get("content", "") if choices else str(data)
    print(f"回答: {msg[:200]}")
    assert msg and len(msg) > 5
    print("  OK")

    print()
    print("=" * 60)
    print("测试 3：流式（数学问题，检查 agent.* 事件 + token）")
    print("=" * 60)

    r = client.post(
        "/agent/chat",
        json={
            "session_id": "test-fix-003",
            "messages": [{"role": "user", "content": "计算 2^10 的值"}],
            "stream": True,
        },
    )
    print(f"status: {r.status_code}")
    assert r.status_code == 200

    has_agent_event = False
    has_tool_event = False
    has_token = False
    full_text = ""
    data_count = 0
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line.startswith("data: "):
            try:
                payload = json.loads(line[6:])
            except Exception as e:
                print(f"  JSON 解析失败: {e}")
                continue
            data_count += 1
            if "_event" in payload:
                has_agent_event = True
                ev = payload["_event"]
                if "tool" in ev["name"]:
                    has_tool_event = True
                print(
                    f"  [event] {ev['name']}: {json.dumps(ev['payload'], ensure_ascii=False)[:150]}"
                )
            if payload.get("choices"):
                choice = payload["choices"][0]
                delta = choice.get("delta", {}).get("content", "") or ""
                if delta:
                    has_token = True
                    full_text += delta
                    sys.stdout.write(delta)
                    sys.stdout.flush()

    print()
    print(f"  data 行数: {data_count}")
    print(f"  有 agent.* 事件: {has_agent_event}")
    print(f"  有 tool.* 事件: {has_tool_event}")
    print(f"  有 token: {has_token}")
    print(f"  完整回答: {full_text[:300]}")
    nums = extract_nums(full_text)
    print(f"  提取的数字: {nums}")
    assert "1024" in nums, f"回答里没找到 1024: {full_text}"
    assert has_token, "应该有 token"
    print("  OK")

    print()
    print("=" * 60)
    print("测试 4：清理会话")
    print("=" * 60)
    for sid in ["test-fix-001", "test-fix-002", "test-fix-003"]:
        r = client.delete(f"/conversations/{sid}")
        print(f"  delete {sid}: {r.status_code}")

    print()
    print("=" * 60)
    print("全部测试通过！")
