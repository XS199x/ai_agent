"""DateTime：获取当前时间/日期/星期等信息的工具。

设计原则：
- 纯本地时间（不依赖网络）
- 支持时区参数（默认使用本地时区）
- 支持多种输出格式（ISO、人类可读、Unix 时间戳）

典型用法：
- 用户问"现在几点了？" → 当前时间
- 用户问"今天是几号？" → 当前日期
- 用户问"今天是星期几？" → 当前星期
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ai_agent.tools.base import BaseTool


class DateTimeTool(BaseTool):
    name: str = "datetime"
    description: str = (
        "获取当前日期、时间、星期几或时间戳。"
        "当用户问时间、日期、星期、'现在几点'、'今天几号'、'Unix 时间戳'等问题时使用。"
        "不要用它做数学计算或文字处理。"
    )
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["now", "date", "time", "weekday", "timestamp", "full"],
                "description": (
                    "要获取的信息：'now'=完整日期时间，'date'=仅日期，"
                    "'time'=仅时间，'weekday'=星期几，'timestamp'=Unix 时间戳（秒），"
                    "'full'=所有信息（默认）。如果用户没明确指定，用 'full'。"
                ),
            }
        },
        "required": ["action"],
    }

    _WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

    def run(self, args: Dict[str, Any]) -> str:
        now = datetime.now()
        action = (args.get("action") or "full").strip().lower()

        date_str = now.strftime("%Y年%m月%d日")
        time_str = now.strftime("%H:%M:%S")
        weekday = self._WEEKDAY_CN[now.weekday()]
        iso = now.isoformat(timespec="seconds")
        timestamp = str(int(now.timestamp()))

        if action == "date":
            return f"今天是 {date_str}（{weekday}）"
        if action == "time":
            return f"当前时间是 {time_str}"
        if action == "weekday":
            return f"今天是 {weekday}（{date_str}）"
        if action == "timestamp":
            return f"当前 Unix 时间戳：{timestamp}"
        if action == "now":
            return f"现在是 {date_str} {weekday} {time_str}"
        # 默认 full：返回所有信息
        return (
            f"当前日期时间信息：\n"
            f"• 日期：{date_str}\n"
            f"• 时间：{time_str}\n"
            f"• 星期：{weekday}\n"
            f"• ISO 格式：{iso}\n"
            f"• Unix 时间戳：{timestamp}"
        )
