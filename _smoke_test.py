"""快速冒烟测试"""

import asyncio
import json
import sys

from fastapi.testclient import TestClient

from src.ai_agent.app import app


async def main():
    with TestClient(app) as client:
        print("=" * 60)
        print("测试 1：数学问题（非流式）")
        print("=" * 60)
        r = client.post(
            "/agent/chat",
            json={
                "session_id": "smoke-1",
                "messages": [{"role": "user", "content": "123 × 456 等于多少？"}],
                "stream": False,
            },
        )
        print(f"status: {r.status_code}")
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {}).get("content", "")
            print(f"回答: {msg[:300]}")
            assert "56088" in msg or "56,088" in msg, f"没找到结果: {msg}"
            print("  [OK]")
        else:
            print(f"没有 choices: {data}")

        print()
        print("=" * 60)
        print("测试 2：闲聊（非流式）")
        print("=" * 60)
        r = client.post(
            "/agent/chat",
            json={
                "session_id": "smoke-2",
                "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
                "stream": False,
            },
        )
        print(f"status: {r.status_code}")
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {}).get("content", "")
            print(f"回答: {msg[:200]}")
            assert msg and len(msg) > 5
            print("  ✅ OK")

        print()
        print("=" * 60)
        print("测试 3：流式（数学问题 + 检查 agent 事件）")
        print("=" * 60)
        r = client.post(
            "/agent/chat",
            json={
                "session_id": "smoke-3",
                "messages": [{"role": "user", "content": "2^10 是多少？"}],
                "stream": True,
            },
        )
        print(f"status: {r.status_code}")
        assert r.status_code == 200

        event_count = 0
        has_tool_result = False
        full_text = ""
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                except:
                    continue
                if "_event" in payload:
                    ev = payload["_event"]
                    event_count += 1
                    print(
                        f"  [event] {ev['name']} payload={json.dumps(ev['payload'], ensure_ascii=False)[:150]}"
                    )
                    if "tool_result" in ev["name"]:
                        has_tool_result = True
                if payload.get("choices"):
                    choice = payload["choices"][0]
                    delta = choice.get("delta", {}).get("content", "") or ""
                    if delta:
                        full_text += delta
                        sys.stdout.write(delta)
                        sys.stdout.flush()
        print()
        print(f"  agent 事件数: {event_count}")
        print(f"  有 tool_result 事件: {has_tool_result}")
        print(f"  完整回答: {full_text[:200]}")
        assert "1024" in full_text, f"回答里没有 1024: {full_text}"
        assert has_tool_result, "应该有 tool_result 事件"
        print("  ✅ OK")

        print()
        print("=" * 60)
        print("全部测试通过！")


asyncio.run(main())
