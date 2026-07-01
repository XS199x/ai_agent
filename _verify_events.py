"""验证：agent.* 事件是否出现在 EventBus 的日志里"""
import asyncio
import json
from fastapi.testclient import TestClient
from src.ai_agent.app import app


def main():
    with TestClient(app) as client:
        print("=" * 60)
        print("测试 1：/agent/chat stream=False（看 EventBus 打印的 agent.* 事件）")
        print("=" * 60)
        r = client.post(
            "/agent/chat",
            json={
                "session_id": "verify-1",
                "messages": [{"role": "user", "content": "123 × 456 等于多少？"}],
                "stream": False,
            },
        )
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {}).get("content", "")
            print(f"回答: {msg[:200]}")
            assert "56088" in msg or "56,088" in msg, f"没找到 56088: {msg}"
            print("  [OK]")

        print()
        print("=" * 60)
        print("测试 2：/agent/chat stream=True（看 SSE 流里的 agent.* 事件 + token）")
        print("=" * 60)
        r = client.post(
            "/agent/chat",
            json={
                "session_id": "verify-2",
                "messages": [{"role": "user", "content": "2^10 是多少？"}],
                "stream": True,
            },
        )

        event_count = 0
        tool_result_seen = False
        full_text = ""
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "_event" in payload:
                ev = payload["_event"]
                event_count += 1
                print(f"  [SSE-event {event_count}] {ev['name']} payload={json.dumps(ev['payload'], ensure_ascii=False)[:150]}")
                if "tool_result" in ev["name"]:
                    tool_result_seen = True
            if payload.get("choices"):
                choice = payload["choices"][0]
                delta = choice.get("delta", {}).get("content", "") or ""
                if delta:
                    full_text += delta
                    print(delta, end="", flush=True)
        print()
        print(f"  总计 SSE 事件：{event_count}")
        print(f"  有 tool_result：{tool_result_seen}")
        print(f"  完整回答：{full_text[:200]}")
        assert "1024" in full_text, f"回答里没找到 1024: {full_text}"
        assert event_count >= 3, f"SSE 事件太少：{event_count}"
        print("  [OK]")

        print()
        print("=" * 60)
        print("全部通过！")


if __name__ == "__main__":
    main()
